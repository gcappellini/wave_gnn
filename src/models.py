"""
Neural Network Models for DeepONet

Defines:
- MLP: Simple feedforward network (single output head)
- DualHeadMLP: Shared backbone with two independent output heads (for u and v)
- DeepONet: Operator network (trunk + branch)
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple feedforward MLP with a single output head."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int, input_scale: float = 1.0):
        super().__init__()
        self.input_scale = input_scale
        # n_layers = total number of Linear layers
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 2):  # n_layers - 2 because we have 1 input + 1 output
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x * self.input_scale
        return self.net(x)


class DualHeadMLP(nn.Module):
    """
    Shared-backbone MLP with two independent linear output heads.

    The backbone maps input → hidden representation (Tanh activations throughout).
    head_u and head_v are separate final linear layers that project the shared
    representation to n_modes each, producing independent outputs for u and v.

    Args:
        input_dim:   Dimensionality of the input.
        hidden_dim:  Width of all hidden layers.
        n_modes:     Output size of each head (same for u and v).
        n_layers:    Total number of Linear layers across backbone + one head.
                     Must be >= 2.  The backbone has (n_layers - 1) linear layers
                     (all but the last), and each head adds one linear layer.
        input_scale: Scalar multiplier applied to the input before the backbone.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_modes: int,
        n_layers: int,
        input_scale: float = 1.0,
        u_output_scale: float = 1.0,
        v_output_scale: float = 1.0,
    ):
        super().__init__()
        self.input_scale = input_scale

        # Backbone: (n_layers - 1) linear layers with Tanh, output size = hidden_dim
        backbone_layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 2):
            backbone_layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        self.backbone = nn.Sequential(*backbone_layers)

        # Two independent output heads
        self.head_u = nn.Linear(hidden_dim, n_modes)
        self.head_v = nn.Linear(hidden_dim, n_modes)

        # Optional per-head output scaling (defaults preserve prior behavior magnitude)
        self.register_buffer('u_output_scale', torch.tensor(float(u_output_scale), dtype=torch.float32))
        self.register_buffer('v_output_scale', torch.tensor(float(v_output_scale), dtype=torch.float32))

    def forward(self, x):
        """
        Returns:
            (out_u, out_v): each of shape (B, n_modes)
        """
        x = x * self.input_scale
        h = self.backbone(x)
        out_u = self.head_u(h) * self.u_output_scale
        out_v = self.head_v(h) * self.v_output_scale
        return out_u, out_v


class DualHeadSensorBranch(nn.Module):
    """Dual-head branch with 2-channel sensor encoder followed by DualHeadMLP."""

    def __init__(
        self,
        n_sensors: int,
        hidden_dim: int,
        n_modes: int,
        n_layers: int,
        input_scale: float = 1.0,
        input_channels: int = 2,
        encoder_channels: int = 8,
        pool_kernel: int = 2,
        pool_stride: int = 2,
        u_output_scale: float = 1.0,
        v_output_scale: float = 1.0,
    ):
        super().__init__()
        self.n_sensors = n_sensors
        self.input_channels = int(input_channels)
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride

        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_channels, encoder_channels, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(encoder_channels, 1, kernel_size=3, padding=1),
            # nn.Tanh(),
            # nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, self.input_channels, n_sensors, n_sensors)
            encoded = self.encoder(dummy)
            self.encoded_shape = tuple(encoded.shape[1:])
            encoded_dim = int(encoded.numel())

        self.mlp = DualHeadMLP(
            input_dim=encoded_dim,
            hidden_dim=hidden_dim,
            n_modes=n_modes,
            n_layers=n_layers,
            input_scale=input_scale,
            u_output_scale=u_output_scale,
            v_output_scale=v_output_scale,
        )

    def forward(self, x):
        """Accepts either (B,C,S,S) sensors or flattened (B,C*S*S) inputs."""
        if x.dim() == 2:
            bsz = x.shape[0]
            x = x.view(bsz, self.input_channels, self.n_sensors, self.n_sensors)
        elif x.dim() != 4:
            raise ValueError(
                f"Expected branch input of shape (B,C,S,S) or (B,C*S*S) with C={self.input_channels}, "
                f"got {tuple(x.shape)}"
            )

        features = self.encoder(x).flatten(start_dim=1)
        return self.mlp(features)


class DeepONet(nn.Module):
    """
    DeepONet operator network: trunk + branch (IC or force).

    Free evolution uses DualHeadMLP for both trunk and branch:
      - trunk:  (x, y, t) → (modes_u, modes_v)
      - branch: [u0_sensors, v0_sensors] → (coeffs_u, coeffs_v)
      - forward returns (u, v) tuple, each shape (B,)

    Constant force keeps the original single-output design:
      - trunk:  MLP (x, y, t) → modes
      - branch: MLP force_sensors → coeffs
      - forward returns scalar u, shape (B,)
    """
    
    def __init__(self, 
                 trunk,
                 branch_ic=None,
                 branch_force: MLP = None,
                 problem_type: str = 'free_evolution',
                 wave_speed: float = 1.0,
                 damping: float = 1.0):
        """
        Args:
            trunk:        DualHeadMLP (free_evolution) or MLP (constant_force)
            branch_ic:    DualHeadMLP for IC measurements (free_evolution)
            branch_force: MLP for force measurements (constant_force)
            problem_type: 'free_evolution' or 'constant_force'
            wave_speed:   Wave speed parameter c
            damping:      Damping parameter k
        """
        super().__init__()
        self.trunk = trunk
        self.branch_ic = branch_ic
        self.branch_force = branch_force
        self.problem_type = problem_type
        self.c = wave_speed
        self.k = damping
        
        # Validate problem type
        if problem_type not in ['free_evolution', 'constant_force']:
            raise ValueError(f"problem_type must be 'free_evolution' or 'constant_force', got {problem_type}")
    
    def get_branch(self):
        """Get the active branch based on problem type."""
        if self.problem_type == 'free_evolution':
            if self.branch_ic is None:
                raise RuntimeError("branch_ic is None for free_evolution problem")
            return self.branch_ic
        else:  # constant_force
            if self.branch_force is None:
                raise RuntimeError("branch_force is None for constant_force problem")
            return self.branch_force
    
    def forward(self, measurements: torch.Tensor, coords: torch.Tensor):
        """
        Forward pass.

        Args:
            measurements: IC or force measurements.
                          free_evolution:  shape (B, 2*n_sensors^2)  [u0 || v0 sensors]
                          constant_force:  shape (B, n_sensors^2)
            coords:       Spatial-temporal coordinates (x, y, t), shape (B, 3)

        Returns:
            free_evolution:  (u, v) tuple, each shape (B,)
            constant_force:  u scalar predictions, shape (B,)
        """
        branch = self.get_branch()

        if self.problem_type == 'free_evolution':
            # Both trunk and branch are DualHeadMLP → return tuples
            trunk_u, trunk_v = self.trunk(coords)       # each (B, n_modes)
            branch_u, branch_v = branch(measurements)   # each (B, n_modes)
            u = torch.sum(trunk_u * branch_u, dim=1)    # (B,)
            v = torch.sum(trunk_v * branch_v, dim=1)    # (B,)
            return u, v
        else:
            # constant_force: single-output MLP trunk and branch
            trunk_out = self.trunk(coords)              # (B, n_modes)
            branch_out = branch(measurements)           # (B, n_modes)
            u = torch.sum(trunk_out * branch_out, dim=1)
            return u
    
    def compute_pde_residual(self, 
                            measurements: torch.Tensor, 
                            xyt: torch.Tensor,
                            force_field: torch.Tensor = None) -> torch.Tensor:
        """
        Compute PDE residual for constant_force only.
        
        Wave equation:  u_tt + k*u_t - c^2*(u_xx+u_yy) = f  (or 0)
        
        Args:
            measurements: Force measurements
            xyt:          Coordinates (x, y, t) requiring gradients, shape (B, 3)
            force_field:  Force field f(x, y), shape (B,). If None, free evolution.
        
        Returns:
            residual: PDE residual, shape (B,)
        """
        xyt_grad = xyt.clone().requires_grad_(True)
        result = self.forward(measurements, xyt_grad)
        # For constant_force forward returns a scalar tensor; for free_evolution a tuple
        u = result[0] if isinstance(result, tuple) else result
        
        # First derivatives
        grad_u = torch.autograd.grad(
            u, xyt_grad,
            torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        u_x, u_y, u_t = grad_u[:, 0], grad_u[:, 1], grad_u[:, 2]
        
        # Second derivatives
        u_xx = torch.autograd.grad(u_x, xyt_grad, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0][:, 0]
        u_yy = torch.autograd.grad(u_y, xyt_grad, torch.ones_like(u_y), create_graph=True, retain_graph=True)[0][:, 1]
        u_tt = torch.autograd.grad(u_t, xyt_grad, torch.ones_like(u_t), create_graph=True)[0][:, 2]
        
        residual = u_tt + self.k * u_t - self.c**2 * (u_xx + u_yy)
        
        if force_field is not None:
            residual = residual - force_field
        
        return residual
