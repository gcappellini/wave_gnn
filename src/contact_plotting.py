"""Validation plots for the ContactDeepONet (field rollout + force validation)."""

from pathlib import Path

import numpy as np

from .plotting import _plot_field_row


def _plot_field_rollout(roll: dict, sample: int, dt: float, stride: int, out_path: Path, n_snap: int = 4) -> None:
    """GT | Pred | |Error| for u and v at a few rollout times (same layout as previous steps)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    snaps = roll['snapshots']
    n_steps = snaps['u_pred'].shape[1]
    step_ids = np.unique(np.linspace(0, n_steps - 1, min(n_snap, n_steps), dtype=int))
    rows = [(kind, sid) for sid in step_ids for kind in ('u', 'v')]

    def limits(kind):
        vals = [snaps[f'{kind}_{w}'][sample, sid] for k, sid in rows if k == kind for w in ('true', 'pred')]
        err = [np.abs(snaps[f'{kind}_pred'][sample, sid] - snaps[f'{kind}_true'][sample, sid])
               for k, sid in rows if k == kind]
        return (min(a.min() for a in vals), max(a.max() for a in vals)), max(max(e.max() for e in err), 1e-12)

    lim = {k: limits(k) for k in ('u', 'v')}
    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 4 * len(rows)))
    for r, (kind, sid) in enumerate(rows):
        t_val = (sid + 1) * stride * dt
        _plot_field_row(
            axes[r], snaps[f'{kind}_true'][sample, sid], snaps[f'{kind}_pred'][sample, sid], kind, t_val,
            field_kind=kind, value_limits=lim[kind][0], error_max=lim[kind][1],
        )
    sim = int(roll['sim_ids'][sample])
    fig.suptitle(
        f"Contact DeepONet rollout (sim {sim}, centre x={roll['center_x'][sample]:.2f}, z={roll['center_z'][sample]:.2f})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 0.84, 0.97])
    for i, kind in enumerate(('u', 'v')):
        for j, (cmap, (lo, hi), name) in enumerate((
            ('seismic' if kind == 'u' else 'viridis', lim[kind][0], f'{kind} value'),
            ('YlOrRd', (0.0, lim[kind][1]), f'|error| {kind}'),
        )):
            cax = fig.add_axes([0.865 + 0.06 * j, 0.80 - 0.4 * i, 0.012, 0.15])
            sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=lo, vmax=hi), cmap=cmap)
            sm.set_array([])
            fig.colorbar(sm, cax=cax).set_label(name, fontsize=9)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_error_accumulation(roll: dict, dt: float, stride: int, out_path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t = np.arange(1, roll['rel_err_u'].shape[1] + 1) * stride * dt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, name in zip(axes, ('rel_err_u', 'rel_err_v'), ('u', 'v')):
        ax.plot(t, roll[key].T, color='C0', alpha=0.25)
        ax.plot(t, roll[key].mean(0), color='k', lw=2, label='mean')
        ax.set(xlabel='time [s]', ylabel='relative L2 error', title=f'Rollout error accumulation ({name})')
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_force_validation(roll: dict, dt: float, out_path: Path, n_curves: int = 6) -> None:
    """Force validation: time series (with contact/tool state), parity, error and peak-force checks."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fp, ft = roll['force_pred'], roll['force_true']
    n, nf = ft.shape
    t = (np.arange(nf) + 1) * dt
    err = fp - ft
    rmse = np.sqrt(np.mean(err ** 2, axis=1))
    # Samples spread over the error ranking: best, median and worst cases are all shown.
    order = np.argsort(rmse)
    picks = order[np.unique(np.linspace(0, n - 1, min(n_curves, n), dtype=int))]

    fig = plt.figure(figsize=(18, 4.2 * (2 + int(np.ceil(len(picks) / 3)))))
    gs = fig.add_gridspec(2 + int(np.ceil(len(picks) / 3)), 3)

    for i, sidx in enumerate(picks):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        ax.plot(t, ft[sidx], 'k', label='true force')
        ax.plot(t, fp[sidx], 'r--', label='predicted force')
        ax.fill_between(t, 0, 1, where=roll['contact'][sidx] > 0.5, color='C0', alpha=0.12,
                        transform=ax.get_xaxis_transform(), label='contact')
        ax2 = ax.twinx()
        ax2.plot(t, roll['tool_y'][sidx], color='C2', alpha=0.6, lw=1)
        ax2.set_ylabel('ToolY', color='C2')
        ax.set(xlabel='time [s]', ylabel='force',
               title=f"sim {int(roll['sim_ids'][sidx])}  RMSE={rmse[sidx]:.3g}  "
                     f"(c=({roll['center_x'][sidx]:.2f},{roll['center_z'][sidx]:.2f}))")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    row = int(np.ceil(len(picks) / 3))
    ax = fig.add_subplot(gs[row, 0])
    ax.scatter(ft.ravel(), fp.ravel(), s=1, alpha=0.2)
    lim = [min(ft.min(), fp.min()), max(ft.max(), fp.max())]
    ax.plot(lim, lim, 'k--')
    r2 = 1 - np.sum(err ** 2) / (np.sum((ft - ft.mean()) ** 2) + 1e-12)
    ax.set(xlabel='true force', ylabel='predicted force', title=f'Force parity (R²={r2:.3f})')
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[row, 1])
    ax.plot(t, np.sqrt(np.mean(err ** 2, axis=0)), label='RMSE over samples')
    ax.plot(t, np.abs(err).mean(0), label='mean |error|')
    ax.plot(t, np.sqrt(np.mean(ft ** 2, axis=0)), 'k:', label='true force RMS')
    ax.set(xlabel='time [s]', ylabel='force error', title='Force error vs time')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[row, 2])
    pk_t, pk_p = ft.max(axis=1), fp.max(axis=1)
    ax.scatter(pk_t, pk_p, s=25)
    lim = [min(pk_t.min(), pk_p.min()), max(pk_t.max(), pk_p.max())]
    ax.plot(lim, lim, 'k--')
    ax.set(xlabel='true peak force', ylabel='predicted peak force', title='Peak force per simulation')
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[row + 1, 0])
    ax.hist(err[np.abs(ft) > 0], bins=60, color='C1')
    ax.set(xlabel='force error (contact frames)', ylabel='count', title='Error distribution (in contact)')

    ax = fig.add_subplot(gs[row + 1, 1])
    no_contact = np.abs(ft) == 0
    leak = np.abs(fp[no_contact]) if no_contact.any() else np.zeros(1)
    ax.hist(leak, bins=60, color='C3')
    ax.set(xlabel='|predicted force| where true force = 0', ylabel='count',
           title=f'Spurious force out of contact (mean {leak.mean():.2g})')
    ax.set_yscale('log')

    ax = fig.add_subplot(gs[row + 1, 2])
    ax.scatter(roll['center_x'], roll['center_z'], c=rmse, cmap='viridis', s=60)
    ax.set(xlabel='CenterX', ylabel='CenterZ', title='Force RMSE vs contact position')
    fig.colorbar(ax.collections[0], ax=ax, label='force RMSE')

    fig.suptitle('Contact DeepONet: force validation (autoregressive rollout)', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_contact_validation(roll: dict, dt: float, stride: int, output_dir: Path, n_field_samples: int = 2) -> None:
    output_dir = Path(output_dir)
    n_field = min(n_field_samples, roll['snapshots']['u_pred'].shape[0])
    for s in range(n_field):
        _plot_field_rollout(roll, s, dt, stride, output_dir / f"validation_contact_rollout_sample{s}.png")
    _plot_error_accumulation(roll, dt, stride, output_dir / "validation_contact_rollout_metrics.png")
    _plot_force_validation(roll, dt, output_dir / "validation_contact_force.png")


def plot_contact_training_curves(history: list, out_path: Path) -> None:
    """Train/test losses (total, u, v, force), learning rate and phase changes across epochs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ep = np.array([h['epoch'] for h in history])
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    panels = [('total', 'Total (weighted)'), ('u', 'u loss'), ('v', 'v loss'), ('force', 'Force loss')]
    for ax, (key, title) in zip(axes.ravel(), panels):
        ax.plot(ep, [h['train'][key] for h in history], label='train')
        ax.plot(ep, [h['test'][key] for h in history], label='test')
        ax.set(xlabel='epoch', ylabel='normalised MSE', title=title, yscale='log')
        ax.grid(alpha=0.3, which='both')
        ax.legend()

    ax = axes[1, 1]
    ax.plot(ep, [h['lr'] for h in history], color='C2')
    ax.set(xlabel='epoch', ylabel='learning rate', title='Learning rate', yscale='log')
    ax.grid(alpha=0.3, which='both')

    ax = axes[1, 2]
    ax.plot(ep, [h['horizon'] for h in history], drawstyle='steps-post', label='rollout horizon')
    ax.plot(ep, [int(h.get('frozen_pretrained', False)) for h in history], drawstyle='steps-post',
            label='pretrained frozen (1=yes)')
    ax.set(xlabel='epoch', title='Training schedule')
    ax.grid(alpha=0.3)
    ax.legend()

    # Mark phase changes (end of freeze / horizon warm-up) on all loss panels.
    for i in range(1, len(history)):
        for key in ('frozen_pretrained', 'horizon'):
            if history[i].get(key) != history[i - 1].get(key):
                for ax in axes.ravel()[:4]:
                    ax.axvline(ep[i], color='gray', ls=':', alpha=0.7)

    fig.suptitle('Contact DeepONet training curves', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
