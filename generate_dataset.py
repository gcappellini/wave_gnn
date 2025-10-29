#!/usr/bin/env python3
"""
Generate HDF5 dataset for wave GNN training.

Usage:
    python generate_dataset.py                    # Default: 100 sims × 500 steps
    python generate_dataset.py --fast             # Fast: 20 sims × 50 steps
    python generate_dataset.py --num-sims 50      # Custom: 50 sims × 500 steps
    python generate_dataset.py --num-steps 300    # Custom: 100 sims × 300 steps
"""

import argparse
from pathlib import Path
from dataset import generate_h5_simulations

def main():
    parser = argparse.ArgumentParser(description='Generate HDF5 dataset for wave GNN')
    parser.add_argument('--output', type=str, default='data/simulations.h5',
                        help='Output HDF5 file path (default: data/simulations.h5)')
    parser.add_argument('--num-sims', type=int, default=100,
                        help='Number of simulations to generate (default: 100)')
    parser.add_argument('--num-steps', type=int, default=500,
                        help='Number of timesteps per simulation (default: 500)')
    parser.add_argument('--seed', type=int, default=2025,
                        help='Base random seed (default: 2025)')
    parser.add_argument('--fast', action='store_true',
                        help='Fast mode: 20 sims × 50 steps (overrides num-sims and num-steps)')
    parser.add_argument('--no-overwrite', action='store_true',
                        help='Do not overwrite existing file')
    
    args = parser.parse_args()
    
    # Fast mode overrides
    if args.fast:
        num_sims = 20
        num_steps = 50
        print("Fast mode enabled: 20 simulations × 50 timesteps")
    else:
        num_sims = args.num_sims
        num_steps = args.num_steps
    
    # Check if file exists
    output_path = Path(args.output)
    if output_path.exists() and args.no_overwrite:
        print(f"File already exists: {output_path}")
        print("Use --overwrite flag or remove --no-overwrite to regenerate")
        return
    
    # Estimate file size
    est_size_mb = (num_sims * num_steps * 100 * 3 * 8) / (1024 * 1024)  # Rough estimate
    print(f"\nGenerating HDF5 dataset:")
    print(f"  Output: {args.output}")
    print(f"  Simulations: {num_sims}")
    print(f"  Timesteps per sim: {num_steps}")
    print(f"  Estimated size: ~{est_size_mb:.1f} MB")
    print(f"  Base seed: {args.seed}")
    print()
    
    try:
        out_file = generate_h5_simulations(
            out_path=args.output,
            num_samples=num_sims,
            num_steps=num_steps,
            base_seed=args.seed,
            overwrite=not args.no_overwrite,
        )
        print(f"\n✓ Successfully saved dataset to: {out_file}")
        
        # Print actual file size
        actual_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✓ Actual file size: {actual_size_mb:.1f} MB")
        
    except Exception as e:
        print(f"\n✗ Dataset generation failed: {e}")
        raise

if __name__ == "__main__":
    main()
