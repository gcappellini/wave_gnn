# DeepONet 2D Wave Equation Experiments

This branch aims at backup the code for the paper "Physics-Informed Autoregressive DeepONet Surrogate for Real-Time Interactive Simulation of Deformable Membranes".
Only the strict code necessary to this imlementation will be kept in this branch, organized in order to build on it subsequent implementations and developments.


## Project Structure

```
wave_gnn/
├── src/                               # Reusable modules
│   ├── ground_truth_generation.py     # MATLAB wrapper, .mat loading
│   ├── svd_analysis.py                # POD basis extraction, visualization
│   ├── training.py                    # MLP, DeepONet, training loops
│   └── plotting.py                    # Validation plots
│   └── models.py                      # All the models
├── configs/                            # Hydra configuration per problem
│   ├── free_evolution/
│   │   └── config.yaml                # Free evolution hyperparameters
│   └── constant_force/
│       └── config.yaml                # Constant force hyperparameters
├── data/                               # Simulation data and SVD results
│   ├── free_evolution.mat
│   ├── svd_free_evolution.npy
│   ├── constant_force.mat
│   └── svd_constant_force.npy
├── models/                             # Trained model checkpoints
│   ├── trunk_svd_free_evolution.pth
│   ├── branch_svd_free_evolution.pth
│   └── ...
├── outputs/                            # Hydra timestamped outputs
│   └── YYYY-MM-DD/
│       └── HH-MM-SS/
│           ├── .hydra/                # Config snapshots
│           ├── *.log                  # Training logs
│           ├── *_training_curves.png  # Loss curves
│           └── validation_*.png       # Validation plots
├── curriculum_wrapper.py              # Pipeline entire curriculum training
├── run_free_evolution.py              # Pipeline: Free evolution
├── generate_curriculum_table_detailed.py # Some code for plot
├── main_ssh.ipynb                     # Notebook to run on gpu ssh
├── requirements.txt                   
├── run_constant_force.py              # Pipeline: Constant force
└── README.md
```


## Configuration System

Hydra manages all hyperparameters via YAML configs.

## Output Structure

All outputs are timestamped under `outputs/YYYY-MM-DD/HH-MM-SS/`:


