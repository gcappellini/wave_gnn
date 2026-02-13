"""
Quick test to verify validation plotting functions work correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Try importing
try:
    from plotting import plot_validation_basic, plot_trunk_validation, plot_deeponet_validation
    print("✓ All validation functions imported successfully")
    print("\nAvailable functions:")
    print("  - plot_validation_basic")
    print("  - plot_trunk_validation")
    print("  - plot_deeponet_validation")
    
    # Check function signatures
    import inspect
    
    print("\n" + "=" * 70)
    print("plot_trunk_validation signature:")
    print(inspect.signature(plot_trunk_validation))
    
    print("\nplot_deeponet_validation signature:")
    print(inspect.signature(plot_deeponet_validation))
    print("=" * 70)
    
    print("\n✓ Validation module ready!")
    
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
