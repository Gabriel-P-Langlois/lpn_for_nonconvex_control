#!/usr/bin/env python3
"""
Test script to verify environment and GPU setup
"""
import os
import sys
import torch

print("="*50)
print("Environment Test")
print("="*50)

# Check Python version
print(f"Python: {sys.version}")

# Check PyTorch
print(f"PyTorch version: {torch.__version__}")

# Check CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Test GPU computation
    print("\nTesting GPU computation...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print(f"GPU computation successful! Result shape: {z.shape}")
    except Exception as e:
        print(f"GPU computation failed: {e}")
else:
    print("WARNING: CUDA not available!")

# Check environment variables
print(f"\nCUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

# Test imports from the project
print("\nTesting project imports...")
sys.path.insert(0, '/project/mang/users/tosin/GLP/lpn_for_nonconvex_control')
sys.path.insert(0, '/project/mang/users/tosin/GLP/lpn_for_nonconvex_control/exps')

try:
    from network import LPN
    print("✓ Successfully imported LPN from network")
except ImportError as e:
    print(f"✗ Failed to import LPN: {e}")

try:
    from lib.utils import cvx
    print("✓ Successfully imported cvx from lib.utils")
except ImportError as e:
    print(f"✗ Failed to import cvx: {e}")

try:
    from lib.invert import invert
    print("✓ Successfully imported invert from lib.invert")
except ImportError as e:
    print(f"✗ Failed to import invert: {e}")

print("\n" + "="*50)
print("All tests completed")
print("="*50)
