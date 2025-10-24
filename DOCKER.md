# Docker Usage for Wave GNN

## Prerequisites

- Docker installed
- NVIDIA Docker runtime (for GPU support)
- NVIDIA drivers installed on host

## Quick Start

### Build the image
```bash
docker build -t wave-gnn:latest .
```

### Run with docker-compose (recommended)
```bash
# Start training with default config
docker-compose up

# Run with custom Hydra overrides
docker-compose run wave-gnn python main.py training=fast_training model.dropout=0.3

# Interactive shell
docker-compose run wave-gnn bash
```

### Run with docker command
```bash
# Basic training
docker run --gpus all -v $(pwd)/outputs:/app/outputs wave-gnn:latest

# Interactive mode
docker run --gpus all -it -v $(pwd)/outputs:/app/outputs wave-gnn:latest bash

# Custom config
docker run --gpus all -v $(pwd)/outputs:/app/outputs wave-gnn:latest \
    python main.py model.dropout=0.2 training.epochs=500
```

## Volume Mounts

The following directories are mounted for persistent storage:
- `./outputs` → `/app/outputs` (training outputs, logs, checkpoints)
- `./figures` → `/app/figures` (plots and visualizations)
- `./matlab` → `/app/matlab` (MATLAB export files)
- `./configs` → `/app/configs` (configuration files)

## GPU Configuration

By default, the container uses GPU 0. To use a different GPU:

```bash
# Set CUDA_VISIBLE_DEVICES
docker-compose run -e CUDA_VISIBLE_DEVICES=1 wave-gnn python main.py

# Or modify docker-compose.yml
```

## Common Commands

### View logs from running container
```bash
docker-compose logs -f
```

### Stop container
```bash
docker-compose down
```

### Test the model
```bash
docker-compose run wave-gnn python test_gcn.py
```

### Clean up old images
```bash
docker image prune -a
```

## Troubleshooting

### GPU not detected
```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Permission issues with mounted volumes
```bash
# Run with current user ID
docker-compose run --user $(id -u):$(id -g) wave-gnn python main.py
```

### Out of memory
- Reduce batch size in configs
- Use smaller model (model.hidden_channels)
- Enable gradient checkpointing if available
