"""Contact-DeepONet training on data/contact_dataset.mat (tool pressing on the membrane).

Builds a ``ContactDeepONet`` on top of a pretrained merged-dataset DeepONet
checkpoint and fine-tunes it on the contact dataset. Branch inputs are the
8x8 sensor grids of (u, v) plus [Contact, CenterX, CenterZ, ToolY]; outputs
are u, v at any (x, y, t) and the contact Force F(t).

Training is autoregressive in the same spirit as the merged curriculum: from the
state at frame k the network predicts the sensor state `frame_stride` frames
later, which is fed back (with ground-truth contact/tool inputs) for
`rollout_horizon` steps. Only sensor values are needed to propagate the state,
so the full 90x90 fields are used just as supervision targets.
"""

import json
import logging
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from .contact_plotting import plot_contact_training_curves, plot_contact_validation
from .models import ContactDeepONet, DualHeadMLP, DualHeadSensorBranch

logger = logging.getLogger("wave_gnn")

N_CONTACT_INPUTS = 4


# --------------------------------------------------------------------------- #
# Data                                                                         #
# --------------------------------------------------------------------------- #
def _resolve_split(script_dir: Path, ccfg: DictConfig, n_total: int) -> tuple[np.ndarray, np.ndarray]:
    split_cfg = ccfg.split
    fraction = float(split_cfg.train_fraction)
    seed = int(split_cfg.seed)
    path = script_dir / "data" / "splits" / f"contact_train{int(fraction * 100)}_test{int(round((1 - fraction) * 100))}_seed{seed}.npz"

    if path.exists():
        split = np.load(path)
        if int(split['n_samples']) != n_total:
            raise ValueError(f"Split {path} was built for {int(split['n_samples'])} samples, dataset has {n_total}")
        logger.info(f"Using existing contact split: {path}")
        return split['train_indices'], split['test_indices']

    perm = np.random.default_rng(seed).permutation(n_total)
    n_train = min(max(int(round(fraction * n_total)), 1), n_total - 1)
    train_ids, test_ids = np.sort(perm[:n_train]), np.sort(perm[n_train:])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, train_indices=train_ids, test_indices=test_ids, seed=np.int64(seed),
             train_fraction=np.float64(fraction), n_samples=np.int64(n_total))
    logger.info(f"Created contact split: {path} (train/test = {len(train_ids)}/{len(test_ids)})")
    return train_ids, test_ids


def _load_samples(h5: h5py.File, sample_ids: np.ndarray, device: torch.device) -> dict:
    """Load selected simulations. Field layout is (n, Nt, d1, d2) with d1 = first MATLAB axis."""

    ids = np.asarray(sample_ids, dtype=np.int64)
    n, (nt, ny, nx) = len(ids), h5['U_hist'].shape[1:]
    u = torch.empty((n, nt, nx, ny), dtype=torch.float32, device=device)
    v = torch.empty_like(u)
    for j, sid in enumerate(ids):
        # HDF5 sample is (Nt, d2, d1); swap to (Nt, d1, d2) to match the `.T` convention of the other datasets.
        u[j] = torch.from_numpy(np.ascontiguousarray(h5['U_hist'][sid].transpose(0, 2, 1))).to(device)
        v[j] = torch.from_numpy(np.ascontiguousarray(h5['V_hist'][sid].transpose(0, 2, 1))).to(device)

    order = np.argsort(ids)
    inverse = np.argsort(order)

    def read(name, shape_tail):
        # h5py fancy indexing requires increasing indices.
        arr = np.asarray(h5[name][ids[order]]).reshape(n, *shape_tail)[inverse]
        return torch.from_numpy(arr).float().to(device)

    return {
        'ids': ids,
        'u': u,
        'v': v,
        'force': read('Force', (nt,)),
        'contact': read('Contact', (nt,)),
        'tool_y': read('ToolY_hist', (nt,)),
        'center_x': read('CenterX', ()),
        'center_z': read('CenterZ', ()),
    }


# --------------------------------------------------------------------------- #
# Model construction                                                           #
# --------------------------------------------------------------------------- #
def build_contact_model(
    cfg: DictConfig,
    pretrained_path: Path | None,
    force_scale: float,
    device: torch.device,
) -> tuple[ContactDeepONet, dict]:
    """Build ContactDeepONet, loading trunk/branch weights from a merged DeepONet checkpoint."""

    ccfg = cfg.contact_training
    n_sensors = int(cfg.sensors.n_sensors)
    ckpt = None
    if pretrained_path is not None:
        ckpt = torch.load(str(pretrained_path), map_location='cpu', weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        trunk_hidden = state['trunk.backbone.0.weight'].shape[0]
        n_modes = state['trunk.head_u.weight'].shape[0]
        trunk_layers = len([k for k in state if k.startswith('trunk.backbone.') and k.endswith('.weight')]) + 1
        branch_hidden = state['branch_ic.mlp.backbone.0.weight'].shape[0]
        branch_layers = len([k for k in state if k.startswith('branch_ic.mlp.backbone.') and k.endswith('.weight')]) + 1
        encoder_channels = state['branch_ic.encoder.0.weight'].shape[0]
        input_channels = state['branch_ic.encoder.0.weight'].shape[1]
        input_scale = float(ckpt.get('input_scale', 1.0) or 1.0)
        normalization = dict(ckpt['normalization'])
    else:
        trunk_hidden, n_modes = int(cfg.networks.trunk.hidden_dim), int(cfg.svd.n_modes)
        trunk_layers = int(cfg.networks.trunk.n_layers)
        branch_hidden, branch_layers = int(cfg.networks.branch.hidden_dim), int(cfg.networks.branch.n_layers)
        encoder_channels, input_channels, input_scale = 8, 3, 1.0
        normalization = {
            'raw_u_min': float(cfg.svd.magnitude_summary.raw_u.min), 'raw_u_max': float(cfg.svd.magnitude_summary.raw_u.max),
            'raw_v_min': float(cfg.svd.magnitude_summary.raw_v.min), 'raw_v_max': float(cfg.svd.magnitude_summary.raw_v.max),
            'raw_f_min': float(cfg.svd.magnitude_summary.raw_f.min), 'raw_f_max': float(cfg.svd.magnitude_summary.raw_f.max),
        }

    trunk = DualHeadMLP(3, trunk_hidden, n_modes, trunk_layers)
    branch = DualHeadSensorBranch(
        n_sensors=n_sensors, hidden_dim=branch_hidden, n_modes=n_modes, n_layers=branch_layers,
        input_scale=input_scale, input_channels=input_channels, encoder_channels=encoder_channels,
    )
    model = ContactDeepONet(
        trunk=trunk,
        branch_ic=branch,
        n_contact_inputs=N_CONTACT_INPUTS,
        contact_hidden_dim=int(ccfg.model.contact_hidden_dim),
        force_trunk_hidden_dim=int(ccfg.model.force_trunk_hidden_dim),
        force_trunk_n_layers=int(ccfg.model.force_trunk_n_layers),
        force_output_scale=force_scale,
    )

    if ckpt is not None:
        missing, unexpected = model.load_state_dict(state, strict=False)
        new_keys = ContactDeepONet.CONTACT_NEW_PREFIXES + ('force_output_scale',)
        bad_missing = [k for k in missing if not k.startswith(new_keys)]
        if bad_missing or unexpected:
            raise RuntimeError(f"Pretrained checkpoint mismatch. missing={bad_missing}, unexpected={unexpected}")
        logger.info(f"Loaded pretrained trunk/branch from {pretrained_path}")
        logger.info(f"  New (randomly initialised) tensors: {len(missing)}")
    else:
        logger.info("No pretrained checkpoint: training ContactDeepONet from scratch")

    arch = {
        'n_modes': n_modes, 'n_sensors': n_sensors,
        'trunk_hidden_dim': trunk_hidden, 'trunk_n_layers': trunk_layers,
        'branch_hidden_dim': branch_hidden, 'branch_n_layers': branch_layers,
        'encoder_channels': encoder_channels, 'input_channels': input_channels, 'input_scale': input_scale,
        'contact_hidden_dim': int(ccfg.model.contact_hidden_dim),
        'force_trunk_hidden_dim': int(ccfg.model.force_trunk_hidden_dim),
        'force_trunk_n_layers': int(ccfg.model.force_trunk_n_layers),
        'n_contact_inputs': N_CONTACT_INPUTS,
    }
    return model.to(device), {'arch': arch, 'normalization': normalization}


def load_contact_checkpoint(path: str | Path, device: torch.device) -> tuple[ContactDeepONet, dict]:
    """Rebuild a ContactDeepONet from a checkpoint written by this module."""

    ckpt = torch.load(str(path), map_location='cpu', weights_only=False)
    a = ckpt['arch']
    trunk = DualHeadMLP(3, a['trunk_hidden_dim'], a['n_modes'], a['trunk_n_layers'])
    branch = DualHeadSensorBranch(
        n_sensors=a['n_sensors'], hidden_dim=a['branch_hidden_dim'], n_modes=a['n_modes'],
        n_layers=a['branch_n_layers'], input_scale=a['input_scale'],
        input_channels=a['input_channels'], encoder_channels=a['encoder_channels'],
    )
    model = ContactDeepONet(
        trunk, branch, a['n_contact_inputs'], a['contact_hidden_dim'],
        a['force_trunk_hidden_dim'], a['force_trunk_n_layers'],
    )
    model.load_state_dict(ckpt['model_state_dict'])
    return model.to(device).eval(), ckpt


# --------------------------------------------------------------------------- #
# Input/target helpers                                                         #
# --------------------------------------------------------------------------- #
def _scale(values: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return 2.0 * (values - lo) / (hi - lo + 1e-10) - 1.0


class _Context:
    """Shared constants for building branch inputs and targets."""

    def __init__(self, ccfg: DictConfig, norm: dict, tool_y_range, n_grid: int, n_sensors: int, device):
        self.device = device
        self.n_grid = n_grid
        self.n_sensors = n_sensors
        self.stride = int(ccfg.frame_stride)
        self.dt_frame = float(ccfg.dt_fixed)
        self.time_scale = float(ccfg.time_scale)
        self.domain_length = float(ccfg.domain_length)
        self.norm = norm
        self.tool_y_range = tool_y_range
        self.scales = {}
        sx = torch.from_numpy(np.linspace(0, n_grid - 1, n_sensors, dtype=np.int64)).to(device)
        self.sensor_idx = sx
        sxx, syy = torch.meshgrid(sx, sx, indexing='ij')
        self.sensor_ix, self.sensor_iy = sxx.reshape(-1), syy.reshape(-1)

    def t_of(self, offset_frames):
        return offset_frames.float() * self.dt_frame / self.time_scale

    def xy_of(self, idx):
        return idx.float() / (self.n_grid - 1)

    def sensor_state(self, data: dict, b: torch.Tensor, k: torch.Tensor):
        u = data['u'][b, k][:, self.sensor_idx][:, :, self.sensor_idx]
        v = data['v'][b, k][:, self.sensor_idx][:, :, self.sensor_idx]
        return u, v

    def branch_sensors(self, u_s: torch.Tensor, v_s: torch.Tensor) -> torch.Tensor:
        n = self.norm
        u_n = _scale(u_s, n['raw_u_min'], n['raw_u_max'])
        v_n = _scale(v_s, n['raw_v_min'], n['raw_v_max'])
        # The static force-field channel of the pretrained model is unused for contact (zero).
        return torch.stack([u_n, v_n, torch.zeros_like(u_n)], dim=1)

    def contact_features(self, data: dict, b: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # Array axis 0 corresponds to CenterZ and axis 1 to CenterX (MATLAB meshgrid convention),
        # so the centre is given in the same (axis0, axis1) order as the trunk coordinates.
        L = self.domain_length
        return torch.stack([
            data['contact'][b, k],
            2.0 * data['center_z'][b] / L - 1.0,
            2.0 * data['center_x'][b] / L - 1.0,
            _scale(data['tool_y'][b, k], *self.tool_y_range),
        ], dim=1)


def _compute_scales(data: dict) -> dict:
    return {
        'u': float(data['u'].std()),
        'v': float(data['v'].std()),
        'force': float(data['force'].pow(2).mean().sqrt()),
    }


# --------------------------------------------------------------------------- #
# Loss over an unrolled window                                                 #
# --------------------------------------------------------------------------- #
def _window_loss(model, data, ctx: _Context, scales: dict, b, k0, horizon: int, n_points: int,
                 noise_std: float, gen=None):
    """Unrolled loss for `horizon` jumps of `frame_stride` frames. Returns per-quantity mean losses."""

    S, ns2 = ctx.stride, ctx.n_sensors ** 2
    bsz = b.shape[0]
    u_s, v_s = ctx.sensor_state(data, b, k0)
    tot = {'u': 0.0, 'v': 0.0, 'force': 0.0}
    k = k0

    for _ in range(horizon):
        sens = ctx.branch_sensors(u_s, v_s)
        if noise_std > 0.0:
            noise = torch.randn(sens.shape, device=sens.device, generator=gen) * noise_std
            noise[:, 2] = 0.0
            sens = torch.clamp(sens + noise, -1.0, 1.0)
        cfeat = ctx.contact_features(data, b, k)

        ix = torch.randint(0, ctx.n_grid, (bsz, n_points), device=ctx.device, generator=gen)
        iy = torch.randint(0, ctx.n_grid, (bsz, n_points), device=ctx.device, generator=gen)
        off = torch.randint(0, S + 1, (bsz, n_points), device=ctx.device, generator=gen)
        # Last ns2 query points: sensor locations at the end of the jump (next state).
        ix = torch.cat([ix, ctx.sensor_ix.expand(bsz, -1)], dim=1)
        iy = torch.cat([iy, ctx.sensor_iy.expand(bsz, -1)], dim=1)
        off = torch.cat([off, torch.full((bsz, ns2), S, device=ctx.device, dtype=off.dtype)], dim=1)

        coords = torch.stack([ctx.xy_of(ix), ctx.xy_of(iy), ctx.t_of(off)], dim=-1)
        u_p, v_p, f_p = model(sens, cfeat, coords)

        frames = k[:, None] + off
        bb = b[:, None]
        u_t, v_t = data['u'][bb, frames, ix, iy], data['v'][bb, frames, ix, iy]
        f_t = data['force'][bb, frames]

        tot['u'] = tot['u'] + F.mse_loss(u_p / scales['u'], u_t / scales['u'])
        tot['v'] = tot['v'] + F.mse_loss(v_p / scales['v'], v_t / scales['v'])
        tot['force'] = tot['force'] + F.mse_loss(f_p / scales['force'], f_t / scales['force'])

        u_s = u_p[:, -ns2:].reshape(bsz, ctx.n_sensors, ctx.n_sensors)
        v_s = v_p[:, -ns2:].reshape(bsz, ctx.n_sensors, ctx.n_sensors)
        k = k + S

    return {name: value / horizon for name, value in tot.items()}


def _combine(losses: dict, weights: dict):
    return sum(weights[name] * losses[name] for name in losses)


@torch.no_grad()
def _evaluate_windows(model, data, ctx, scales, weights, horizon, n_points, seed, batch_size):
    model.eval()
    gen = torch.Generator(device=ctx.device).manual_seed(seed)
    n = data['u'].shape[0]
    max_start = data['u'].shape[1] - 1 - horizon * ctx.stride
    acc, count = {'u': 0.0, 'v': 0.0, 'force': 0.0}, 0
    for start in range(0, n, batch_size):
        b = torch.arange(start, min(start + batch_size, n), device=ctx.device)
        k0 = torch.randint(0, max_start + 1, (len(b),), device=ctx.device, generator=gen)
        losses = _window_loss(model, data, ctx, scales, b, k0, horizon, n_points, 0.0, gen)
        for name in acc:
            acc[name] += float(losses[name]) * len(b)
        count += len(b)
    acc = {name: value / count for name, value in acc.items()}
    acc['total'] = float(_combine(acc, weights))
    return acc


# --------------------------------------------------------------------------- #
# Full-trajectory autoregressive evaluation                                    #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _full_rollout(model, data, ctx: _Context, n_eval: int, batch_size: int = 8, n_snap: int = 4) -> dict:
    """Roll out from frame 0 to the end with ground-truth contact/tool inputs."""

    model.eval()
    S = ctx.stride
    nt = data['u'].shape[1]
    n_steps = (nt - 1) // S
    n = min(n_eval, data['u'].shape[0])
    g = ctx.n_grid
    xs = ctx.xy_of(torch.arange(g, device=ctx.device))
    grid_x, grid_y = torch.meshgrid(xs, xs, indexing='ij')
    grid_xy = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
    force_offsets = torch.arange(1, S + 1, device=ctx.device)

    err_u = torch.zeros(n, n_steps, device=ctx.device)
    err_v = torch.zeros_like(err_u)
    f_pred = torch.zeros(n, n_steps * S, device=ctx.device)
    f_true = torch.zeros_like(f_pred)
    n_snap = min(n_snap, n)
    snaps = {name: torch.zeros(n_snap, n_steps, g, g, device=ctx.device) for name in ('u_pred', 'v_pred', 'u_true', 'v_true')}

    for start in range(0, n, batch_size):
        b = torch.arange(start, min(start + batch_size, n), device=ctx.device)
        bsz = len(b)
        k = torch.zeros(bsz, dtype=torch.long, device=ctx.device)
        u_s, v_s = ctx.sensor_state(data, b, k)
        for step in range(n_steps):
            sens = ctx.branch_sensors(u_s, v_s)
            cfeat = ctx.contact_features(data, b, k)
            t_end = ctx.t_of(torch.full((1,), S, device=ctx.device))
            full = torch.cat([grid_xy, t_end.expand(g * g, 1)], dim=-1).expand(bsz, -1, -1)
            sensor_pts = torch.stack([
                ctx.xy_of(ctx.sensor_ix), ctx.xy_of(ctx.sensor_iy), t_end.expand(ctx.n_sensors ** 2)
            ], dim=-1).expand(bsz, -1, -1)
            f_pts = torch.stack([
                torch.full((S,), 0.5, device=ctx.device), torch.full((S,), 0.5, device=ctx.device),
                ctx.t_of(force_offsets),
            ], dim=-1).expand(bsz, -1, -1)
            u_p, v_p, _ = model(sens, cfeat, full)
            u_sp, v_sp, _ = model(sens, cfeat, sensor_pts)
            _, _, f_p = model(sens, cfeat, f_pts)

            k_next = k + S
            u_t = data['u'][b, k_next].reshape(bsz, -1)
            v_t = data['v'][b, k_next].reshape(bsz, -1)
            err_u[b, step] = (u_p - u_t).norm(dim=1) / (u_t.norm(dim=1) + 1e-8)
            err_v[b, step] = (v_p - v_t).norm(dim=1) / (v_t.norm(dim=1) + 1e-8)
            for name, val in (('u_pred', u_p), ('v_pred', v_p), ('u_true', u_t), ('v_true', v_t)):
                m = b < n_snap
                if m.any():
                    snaps[name][b[m], step] = val[m].reshape(-1, g, g)
            sl = slice(step * S, (step + 1) * S)
            f_pred[b, sl] = f_p
            f_true[b, sl] = data['force'][b[:, None], k[:, None] + force_offsets[None]]

            u_s = u_sp.reshape(bsz, ctx.n_sensors, ctx.n_sensors)
            v_s = v_sp.reshape(bsz, ctx.n_sensors, ctx.n_sensors)
            k = k_next

    return {
        'rel_err_u': err_u.cpu().numpy(), 'rel_err_v': err_v.cpu().numpy(),
        'force_pred': f_pred.cpu().numpy(), 'force_true': f_true.cpu().numpy(),
        'contact': data['contact'][:n, 1:n_steps * S + 1].cpu().numpy(),
        'tool_y': data['tool_y'][:n, 1:n_steps * S + 1].cpu().numpy(),
        'center_x': data['center_x'][:n].cpu().numpy(), 'center_z': data['center_z'][:n].cpu().numpy(),
        'sim_ids': np.asarray(data['ids'][:n]),
        'snapshots': {name: val.cpu().numpy() for name, val in snaps.items()},
    }


def _rollout_summary(roll: dict, ctx: _Context) -> dict:
    fp, ft = roll['force_pred'], roll['force_true']
    rmse = np.sqrt(np.mean((fp - ft) ** 2, axis=1))
    contact = np.abs(ft) > 0
    contact_rmse = np.sqrt(np.sum(((fp - ft) ** 2) * contact, axis=1) / np.maximum(contact.sum(axis=1), 1))
    return {
        'rel_err_u_mean': float(roll['rel_err_u'].mean()),
        'rel_err_u_final': float(roll['rel_err_u'][:, -1].mean()),
        'rel_err_u_p90': float(np.percentile(roll['rel_err_u'].mean(axis=1), 90)),
        'rel_err_v_mean': float(roll['rel_err_v'].mean()),
        'rel_err_v_final': float(roll['rel_err_v'][:, -1].mean()),
        'force_rmse_mean': float(rmse.mean()),
        'force_rmse_contact_mean': float(contact_rmse.mean()),
        'force_true_rms': float(np.sqrt(np.mean(ft ** 2))),
        'n_eval_samples': int(fp.shape[0]),
        'rollout_steps': int(roll['rel_err_u'].shape[1]),
    }


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #
def run_contact_pipeline(cfg: DictConfig, script_dir: Path, output_dir: Path) -> Path | None:
    """Fine-tune ContactDeepONet on contact_dataset.mat. Returns the published checkpoint path."""

    ccfg = cfg.contact_training
    script_dir, output_dir = Path(script_dir), Path(output_dir)
    force_retrain = bool(cfg.pipeline.get('force_retrain', False))

    published = script_dir / str(ccfg.output_checkpoint)
    if published.exists() and not force_retrain:
        logger.info(f"Contact checkpoint exists; skipping training (set pipeline.force_retrain=true): {published}")
        return published

    torch.manual_seed(int(ccfg.seed))
    np.random.seed(int(ccfg.seed))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"CONTACT DEEPONET TRAINING | device: {device}")
    logger.info(f"Config:\n{OmegaConf.to_yaml(ccfg)}")

    dataset_path = script_dir / "data" / str(cfg.datasets.contact.output_file)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Contact dataset not found: {dataset_path}")

    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(dataset_path, 'r') as h5:
        n_total = int(h5['U_hist'].shape[0])
        n_grid = int(h5['U_hist'].shape[-1])
        train_split, test_split = _resolve_split(script_dir, ccfg, n_total)
        train_ids = train_split[: int(ccfg.max_train_samples)]
        test_ids = test_split[: int(ccfg.max_test_samples)]
        logger.info(f"Loading {len(train_ids)} train and {len(test_ids)} test simulations of {n_total}")
        tic = time.time()
        train_data = _load_samples(h5, train_ids, device)
        test_data = _load_samples(h5, test_ids, device)
        tool_y_all = np.asarray(h5['ToolY_hist'][:]).reshape(-1)
        logger.info(f"Loaded in {time.time() - tic:.1f}s")

    tool_y_range = (float(tool_y_all.min()), float(tool_y_all.max()))
    pretrained_path = ccfg.get('pretrained_checkpoint', None)
    pretrained_path = (script_dir / str(pretrained_path)) if pretrained_path else None
    if pretrained_path is not None and not pretrained_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")

    scales = _compute_scales(train_data)
    logger.info(f"Loss scales (std/rms): {scales}; tool_y range: {tool_y_range}")

    model, meta = build_contact_model(cfg, pretrained_path, scales['force'], device)
    ctx = _Context(ccfg, meta['normalization'], tool_y_range, n_grid, int(cfg.sensors.n_sensors), device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"ContactDeepONet parameters: {n_params} (new: {sum(p.numel() for p in model.new_parameters())})")

    weights = {'u': float(ccfg.loss_weights.u), 'v': float(ccfg.loss_weights.v), 'force': float(ccfg.loss_weights.force)}
    lr = float(ccfg.learning_rate)
    optimizer = torch.optim.Adam([
        {'params': model.new_parameters(), 'lr': lr},
        {'params': model.pretrained_parameters(), 'lr': lr * float(ccfg.pretrained_lr_factor)},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=int(ccfg.scheduler_patience)
    )

    n_epochs = int(ccfg.n_epochs)
    steps_per_epoch = int(ccfg.steps_per_epoch)
    batch_size = int(ccfg.batch_size)
    n_points = int(ccfg.points_per_step)
    horizon_final = int(ccfg.rollout_horizon)
    warmup_epochs = int(ccfg.horizon_warmup_epochs)
    freeze_epochs = int(ccfg.freeze_pretrained_epochs)
    noise_std = float(ccfg.input_noise_std)
    eval_horizon = int(ccfg.eval_horizon)
    n_train = train_data['u'].shape[0]
    max_frames = train_data['u'].shape[1]

    best_loss, best_state, history = float('inf'), None, []
    for epoch in range(n_epochs):
        freeze = epoch < freeze_epochs
        for p in model.pretrained_parameters():
            p.requires_grad_(not freeze)
        horizon = 1 if epoch < warmup_epochs else horizon_final

        model.train()
        epoch_losses = {'u': 0.0, 'v': 0.0, 'force': 0.0}
        for _ in range(steps_per_epoch):
            b = torch.randint(0, n_train, (batch_size,), device=device)
            k0 = torch.randint(0, max_frames - horizon * ctx.stride, (batch_size,), device=device)
            losses = _window_loss(model, train_data, ctx, scales, b, k0, horizon, n_points, noise_std)
            loss = _combine(losses, weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(ccfg.grad_clip))
            optimizer.step()
            for name in epoch_losses:
                epoch_losses[name] += float(losses[name].detach()) / steps_per_epoch

        test = _evaluate_windows(model, test_data, ctx, scales, weights, eval_horizon, n_points,
                                 seed=int(ccfg.seed), batch_size=batch_size)
        scheduler.step(test['total'])
        epoch_losses['total'] = sum(float(weights[k]) * epoch_losses[k] for k in ('u', 'v', 'force'))
        history.append({
            'epoch': epoch + 1, 'horizon': horizon, 'frozen_pretrained': bool(freeze),
            'lr': optimizer.param_groups[0]['lr'], 'train': dict(epoch_losses), 'test': test,
        })
        if test['total'] < best_loss:
            best_loss = test['total']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 0 or (epoch + 1) % int(ccfg.log_every) == 0 or epoch + 1 == n_epochs:
            logger.info(
                f"Epoch {epoch + 1:4d}/{n_epochs} H={horizon}{' (frozen pretrained)' if freeze else ''} | "
                f"train u/v/F {epoch_losses['u']:.4e}/{epoch_losses['v']:.4e}/{epoch_losses['force']:.4e} | "
                f"test(H={eval_horizon}) u/v/F {test['u']:.4e}/{test['v']:.4e}/{test['force']:.4e} | "
                f"lr {optimizer.param_groups[0]['lr']:.2e}"
            )

    model.load_state_dict(best_state)
    logger.info(f"Best test loss: {best_loss:.6e}")

    roll = _full_rollout(model, test_data, ctx, int(ccfg.rollout_eval_samples))
    summary = _rollout_summary(roll, ctx)
    logger.info(f"Full-trajectory rollout on test samples: {json.dumps(summary, indent=2)}")
    plot_contact_validation(roll, ctx.dt_frame, ctx.stride, output_dir)
    plot_contact_training_curves(history, output_dir / "contact_training_curves.png")
    with open(output_dir / "contact_metrics.json", 'w') as fh:
        json.dump({'rollout': summary, 'best_test_loss': best_loss, 'history': history}, fh, indent=2)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'arch': meta['arch'],
        'normalization': meta['normalization'],
        'tool_y_range': list(tool_y_range),
        'scales': scales,
        'contact_config': OmegaConf.to_container(ccfg, resolve=True),
        'pretrained_source': str(pretrained_path) if pretrained_path else None,
        'rollout_metrics': summary,
        'train_ids': train_ids.tolist(),
        'test_ids': test_ids.tolist(),
    }
    stage_ckpt = ckpt_dir / "contact_deeponet.pth"
    torch.save(checkpoint, stage_ckpt)
    published.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(stage_ckpt, published)
    logger.info(f"Saved contact checkpoint: {published}")
    return published
