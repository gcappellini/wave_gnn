#!/usr/bin/env python3
"""
Verification script to check if the reorganization was successful.
Run this to verify all components are properly set up.
"""

import sys
from pathlib import Path


def check_file_exists(filepath, description):
    """Check if a file exists."""
    path = Path(filepath)
    exists = path.exists()
    status = "✓" if exists else "✗"
    print(f"  {status} {description}: {filepath}")
    return exists


def check_directory_structure():
    """Verify the directory structure."""
    print("\n" + "="*60)
    print("CHECKING DIRECTORY STRUCTURE")
    print("="*60)
    
    checks = [
        ("configs/config.yaml", "Main config"),
        ("configs/dataset/default.yaml", "Dataset config"),
        ("configs/model/gcn.yaml", "GCN model config"),
        ("configs/model/deep_gcn.yaml", "Deep GCN model config"),
        ("configs/training/default.yaml", "Training config"),
        ("configs/training/fast.yaml", "Fast training config"),
        ("configs/experiment/quick_test.yaml", "Quick test experiment"),
        ("configs/experiment/production.yaml", "Production experiment"),
        ("main.py", "Main script"),
        ("train.py", "Training module"),
        ("test_gcn.py", "Testing module"),
        ("import_mesh.py", "Dataset creation"),
        ("requirements.txt", "Dependencies"),
        ("README.md", "Documentation"),
        (".gitignore", "Git ignore file"),
        ("REORGANIZATION.md", "Reorganization summary"),
    ]
    
    all_exist = True
    for filepath, desc in checks:
        if not check_file_exists(filepath, desc):
            all_exist = False
    
    return all_exist


def check_imports():
    """Check if required imports work."""
    print("\n" + "="*60)
    print("CHECKING IMPORTS")
    print("="*60)
    
    imports_to_check = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("matplotlib", "Matplotlib"),
        ("networkx", "NetworkX"),
        ("hydra", "Hydra"),
        ("omegaconf", "OmegaConf"),
    ]
    
    all_imports_ok = True
    for module_name, description in imports_to_check:
        try:
            __import__(module_name)
            print(f"  ✓ {description} ({module_name})")
        except ImportError:
            print(f"  ✗ {description} ({module_name}) - NOT INSTALLED")
            all_imports_ok = False
    
    return all_imports_ok


def check_config_syntax():
    """Check if config files have valid YAML syntax."""
    print("\n" + "="*60)
    print("CHECKING CONFIG FILE SYNTAX")
    print("="*60)
    
    try:
        from omegaconf import OmegaConf
        
        config_files = [
            "configs/config.yaml",
            "configs/dataset/default.yaml",
            "configs/model/gcn.yaml",
            "configs/training/default.yaml",
        ]
        
        all_valid = True
        for config_file in config_files:
            try:
                cfg = OmegaConf.load(config_file)
                print(f"  ✓ {config_file}")
            except Exception as e:
                print(f"  ✗ {config_file} - ERROR: {str(e)}")
                all_valid = False
        
        return all_valid
    except ImportError:
        print("  ⚠ OmegaConf not installed - skipping syntax check")
        return True


def print_next_steps(all_checks_passed):
    """Print next steps based on verification results."""
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    if all_checks_passed:
        print("\n✓ All checks passed! Your project is properly set up.\n")
        print("Next steps:")
        print("  1. Install dependencies (if not done):")
        print("     pip install -r requirements.txt")
        print("\n  2. Run a quick test:")
        print("     python main.py experiment=quick_test")
        print("\n  3. View available options:")
        print("     python main.py --help")
        print("\n  4. Read the documentation:")
        print("     cat README.md")
    else:
        print("\n✗ Some checks failed. Please address the issues above.\n")
        print("To install missing dependencies:")
        print("  pip install -r requirements.txt")


def main():
    """Run all verification checks."""
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║           Project Reorganization Verification              ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    structure_ok = check_directory_structure()
    imports_ok = check_imports()
    config_ok = check_config_syntax()
    
    all_ok = structure_ok and imports_ok and config_ok
    
    print_next_steps(all_ok)
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
