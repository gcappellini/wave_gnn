#!/usr/bin/env python3
"""
Quick example script showing how to run experiments programmatically.
"""

import subprocess
import sys


def run_command(cmd, description):
    """Run a command and print its description."""
    print("\n" + "="*60)
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print("="*60)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Warning: Command failed with return code {result.returncode}")
    return result.returncode


def main():
    """Run example experiments."""
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║       GCN Wave Equation - Example Experiments              ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    examples = [
        {
            "cmd": "python main.py --help",
            "desc": "Show available configuration options"
        },
        {
            "cmd": "python main.py training=fast dataset.num_graphs=10",
            "desc": "Quick test run (10 graphs, 10 epochs)"
        },
        {
            "cmd": "python main.py model=gcn training.epochs=30",
            "desc": "Standard GCN training (30 epochs)"
        },
        {
            "cmd": "python main.py model=deep_gcn training.epochs=30",
            "desc": "Deep GCN training (30 epochs)"
        },
    ]
    
    print("\nAvailable examples:")
    for i, ex in enumerate(examples, 1):
        print(f"\n{i}. {ex['desc']}")
        print(f"   {ex['cmd']}")
    
    print("\n" + "="*60)
    choice = input("\nEnter example number to run (1-{}, or 'q' to quit): ".format(len(examples)))
    
    if choice.lower() == 'q':
        print("Exiting...")
        return 0
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(examples):
            example = examples[idx]
            return run_command(example["cmd"], example["desc"])
        else:
            print(f"Invalid choice. Please enter a number between 1 and {len(examples)}")
            return 1
    except ValueError:
        print("Invalid input. Please enter a number or 'q'")
        return 1


if __name__ == "__main__":
    sys.exit(main())
