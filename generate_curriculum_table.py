"""
Generate LaTeX table summarizing curriculum training steps.
"""

import yaml
from pathlib import Path


def generate_curriculum_table():
    """Load curriculum config and generate LaTeX table."""
    
    config_path = Path("configs/constant_force/curriculum_total.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    steps = config["curriculum"]["steps"]
    
    # LaTeX table header
    latex_lines = [
        r"\begin{table}[h!]",
        r"\centering",
        r"\caption{Curriculum Training Strategy: Staged DeepONet Training Pipeline}",
        r"\label{tab:curriculum}",
        r"\small",
        r"\begin{tabularx}{\textwidth}{|c|l|c|c|c|c|c|c|c|}",
        r"\hline",
        r"\textbf{Stage} & \textbf{Step Name} & \textbf{Trunk} & \textbf{Branch} & \textbf{DeepONet} & \textbf{PINN} & \textbf{PDE $w$} & \textbf{Wave $w$} & \textbf{Rollout Config} \\",
        r"\hline",
    ]
    
    for idx, step in enumerate(steps):
        name = step.get("name", f"step_{idx:02d}").replace("_", r"\_")
        overrides = step.get("overrides", {}).get("training", {})
        
        trunk_epochs = overrides.get("trunk_n_epochs", 0)
        branch_epochs = overrides.get("branch_n_epochs", 0)
        deeponet_epochs = overrides.get("deeponet_n_epochs", 0)
        finetune = "Yes" if overrides.get("finetune_deeponet", False) else "No"
        use_pinn = "Yes" if overrides.get("use_pinn_loss", False) else "No"
        pde_weight = overrides.get("pde_loss_weight", 0.0)
        pde_wave_weight = overrides.get("pde_loss_weight_wave", 0.0)
        
        rollout_cfg = overrides.get("deeponet_rollout", {})
        if rollout_cfg.get("enabled", False):
            horizon = rollout_cfg.get("horizon", "---")
            stride = rollout_cfg.get("step_stride", "---")
            rollout_str = f"H={horizon}, S={stride}"
        else:
            rollout_str = "---"
        
        # Build row
        row = (
            f"{idx} & {name} & {trunk_epochs} & {branch_epochs} & {deeponet_epochs} & "
            f"{use_pinn} & {pde_weight} & {pde_wave_weight} & {rollout_str} \\\\"
        )
        latex_lines.append(row)
    
    # Table footer
    latex_lines.extend([
        r"\hline",
        r"\end{tabularx}",
        r"\vspace{0.5em}",
        r"\begin{tablenotes}",
        r"\small",
        r"\item \textbf{Columns:} Trunk/Branch/DeepONet = number of training epochs; PINN = Physics-Informed Neural Network loss enabled; PDE $w$ = PDE loss weight; Wave $w$ = Wave equation loss weight; Rollout Config = (H)orizon and (S)tride for autoregressive rollout.",
        r"\item Each stage initializes from the previous stage's checkpoints for progressive curriculum learning.",
        r"\end{tablenotes}",
        r"\end{table}",
    ])
    
    return "\n".join(latex_lines)


if __name__ == "__main__":
    latex_table = generate_curriculum_table()
    print(latex_table)
    
    # Save to file
    output_path = Path("curriculum_table.tex")
    with open(output_path, "w") as f:
        f.write(latex_table)
    print(f"\n✓ Saved to {output_path}")
