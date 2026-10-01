# Autoregressive DeepONet 2D Wave Equation Experiments

This branch aims at backup the code for the paper "Physics-Informed Autoregressive DeepONet Surrogate for Real-Time Interactive Simulation of Deformable Membranes".
Only the strict code necessary to this implementation will be kept in this branch, organized in order to build on it subsequent implementations and developments.


## Project Structure

```
wave_gnn/
├── main.py                    # Hydra entrypoint for the complete experiment
├── configs/
│   ├── config.yaml             # Experiment, pipeline, and MATLAB parameters
│   ├── curriculum_total.yaml   # Publication curriculum
│   └── curriculum_smoke_test.yaml
├── src/
│   ├── curriculum.py           # Staged training and checkpoint reuse
│   ├── ground_truth_generation.py
│   ├── logging_utils.py        # Console and file logging
│   ├── models.py               # DeepONet model definitions
│   ├── pipeline.py             # Merged-dataset training pipeline
│   ├── plotting.py             # Validation and publication plots
│   ├── reporting.py            # Curriculum metrics and tables
│   ├── svd_analysis.py         # POD basis extraction and analysis
│   └── training.py             # Model training loops
├── data/                       # MATLAB generators and dataset utilities
├── models/                     # Published checkpoints
├── outputs/                    # Timestamped Hydra runs and logs
├── main_ssh.ipynb              # Remote run launcher and monitoring
├── requirements.txt
└── README.md
```

## Configuration System

Hydra manages experiment parameters and pipeline stages via YAML. MATLAB dataset
generators read their per-dataset parameters from `configs/config.yaml` using
MATLAB's `readyaml` function.

Run the publication curriculum or its smoke test through the single entrypoint:

```bash
python main.py --config-name curriculum_total
python main.py --config-name curriculum_smoke_test
```

The `pipeline` section controls dataset generation, analysis, curriculum
training, table generation, and publication rollout plots. Existing stage
checkpoints in `models/` are reused unless `pipeline.force_retrain=true`.

## Output Structure

All outputs are timestamped under `outputs/YYYY-MM-DD/HH-MM-SS/`:


