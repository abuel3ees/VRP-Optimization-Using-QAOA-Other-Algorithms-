#!/usr/bin/env python3
"""
Check if QAOA (Qiskit) is available and test with Recursive from Notebook
"""
import sys

try:
    import qiskit
    import qiskit_aer
    import qiskit_algorithms
    import qiskit_optimization
    print("✓ QAOA (Qiskit) is AVAILABLE")
    print(f"  - qiskit version: {qiskit.__version__}")
    print(f"  - qiskit-aer version: {qiskit_aer.__version__}")
    print(f"  - qiskit-algorithms version: {qiskit_algorithms.__version__}")
    print(f"  - qiskit-optimization version: {qiskit_optimization.__version__}")
    sys.exit(0)
except ImportError as e:
    print("✗ QAOA (Qiskit) is NOT available")
    print(f"  Error: {e}")
    sys.exit(1)
