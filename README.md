# Autoregressive DeepONet 2D Wave Equation Experiments

This branch aims at backup the code for the paper "Physics-Informed Autoregressive DeepONet Surrogate for Real-Time Interactive Simulation of Deformable Membranes".
Only the strict code necessary to this implementation will be kept in this branch, organized in order to build on it subsequent implementations and developments.


## Actual project Structure

```
wave_gnn/
├── src/                               # Reusable modules
│   ├── ground_truth_generation.py     # MATLAB wrapper, .mat loading
│   ├── svd_analysis.py                # POD basis extraction, visualization
│   ├── training.py                    # MLP, DeepONet, training loops
│   └── plotting.py                    # Validation plots
│   └── models.py                      # All the models
├── configs/                           # Hydra configuration per problem
│   ├── config.yaml                    # General hyperparameters
│   └── curriculum.yaml                # Training-specific hyperparameters
├── data/                              # Scripts to generate the dataset
│   └── ...
├── models/                             # Trained model checkpoints
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

## Target project Structure

In the main folder I would like to have only one script, main.py. all the curriculum wrapping, plotting scripts, pipeline scripts, and table geenrators should be moved in src and distributed in the existing / new scripts that should logically pertain to.
The entire experiment should be set from config files. 
Also MATLAB dataset generation scripts should load parameters from this yaml file, to keep a better trace overall.

## Configuration System

Hydra manages all hyperparameters via YAML configs.

## Output Structure

All outputs are timestamped under `outputs/YYYY-MM-DD/HH-MM-SS/`:


