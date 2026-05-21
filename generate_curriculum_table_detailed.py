"""
Generate detailed LaTeX table with all curriculum training parameters.
"""

import yaml
from pathlib import Path


def generate_detailed_curriculum_table():
    """Load curriculum config and generate detailed LaTeX table."""
    
    config_path = Path("configs/constant_force/curriculum_total.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    steps = config["curriculum"]["steps"]
    
    # Detailed LaTeX table header
    latex_lines = [
        r"\begin{landscape}",
        r"\begin{table}[h!]",
        r"\centering",
        r"\caption{Detailed Curriculum Training Strategy}",
        r"\label{tab:curriculum_detailed}",
        r"\tiny",
        r"\begin{tabularx}{\linewidth}{|c|l|c|c|c|c|c|c|c|l|}",
        r"\hline",
        r"\textbf{Stage} & \textbf{Step} & \textbf{Trunk} & \textbf{Branch} & \textbf{DeepONet} & \textbf{Finetune} & \textbf{PINN} & \textbf{PDE Weight} & \textbf{Rollout} & \textbf{Additional Config} \\",
        r"\hline",
    ]
    
    for idx, step in enumerate(steps):
        name = step.get("name", f"step_{idx:02d}").replace("_", "\\_")
        overrides = step.get("overrides", {}).get("training", {})
        
        trunk_epochs = overrides.get("trunk_n_epochs", 0)
        branch_epochs = overrides.get("branch_n_epochs", 0)
        deeponet_epochs = overrides.get("deeponet_n_epochs", 0)
        finetune = "Yes" if overrides.get("finetune_deeponet", False) else "No"
        use_pinn = "Yes" if overrides.get("use_pinn_loss", False) else "No"
        pde_weight = overrides.get("pde_loss_weight", 0.0)
        pde_weight_uv = overrides.get("pde_loss_weight_uv", 0.0)
        pde_wave_weight = overrides.get("pde_loss_weight_wave", 0.0)
        
        rollout_cfg = overrides.get("deeponet_rollout", {})
        if rollout_cfg.get("enabled", False):
            horizon = rollout_cfg.get("horizon", "---")
            stride = rollout_cfg.get("step_stride", "---")
            rollout_str = f"H={horizon}, S={stride}"
        else:
            rollout_str = "Disabled"
        
        # Build additional config info
        additional = []
        
        # Check for noise config
        noise_cfg = overrides.get("deeponet_branch_input_noise", {})
        if noise_cfg.get("enabled", False):
            noise_std = noise_cfg.get("std", "---")
            additional.append(f"Noise (std={noise_std})")
        
        # Check for PDE warmup
        if use_pinn == "Yes":
            warmup_epochs = overrides.get("pde_loss_weight_warmup_epochs", 0)
            schedule = overrides.get("pde_loss_weight_schedule", "linear")
            additional.append(f"Warmup={warmup_epochs} ({schedule})")
        
        additional_str = " / ".join(additional) if additional else "---"
        
        # Build pde weight string
        pde_weight_str = f"{pde_weight} (UV: {pde_weight_uv}, Wave: {pde_wave_weight})"
        
        # Build row
        row = (
            f"{idx} & {name} & {trunk_epochs} & {branch_epochs} & {deeponet_epochs} & "
            f"{finetune} & {use_pinn} & {pde_weight_str} & {rollout_str} & {additional_str} \\\\"
        )
        latex_lines.append(row)
    
    # Table footer
    latex_lines.extend([
        r"\hline",
        r"\end{tabularx}",
        r"\vspace{0.5em}",
        r"\begin{tablenotes}",
        r"\tiny",
        r"\item \textbf{Columns:} Trunk/Branch/DeepONet = epochs; Finetune = freeze/update parameters; PINN = Physics-Informed loss; PDE Weight = (overall, UV component, Wave component); Rollout = (H)orizon and (S)tride.",
        r"\item \textbf{Additional Config:} Noise = branch input noise regularization; Warmup = PDE loss weight schedule parameters.",
        r"\item Progressive strategy: Stages 0-2 are supervised pretraining; Stages 3-6 add physics loss with increasing rollout horizons and wave equation weighting.",
        r"\end{tablenotes}",
        r"\end{table}",
        r"\end{landscape}",
    ])
    
    return "\n".join(latex_lines)


if __name__ == "__main__":
    latex_table = generate_detailed_curriculum_table()
    print(latex_table)
    
    # Save to file
    output_path = Path("curriculum_table_detailed.tex")
    with open(output_path, "w") as f:
        f.write(latex_table)
    print(f"\n✓ Saved to {output_path}")
