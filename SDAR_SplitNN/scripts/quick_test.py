"""
Quick smoke test for SDAR pipeline using synthetic data.
Verifies that all model components initialize and train correctly
without needing to download the full CIFAR-10 dataset.
"""
import os
import sys
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.append('../src')
import tensorflow as tf

print("=" * 60)
print("SDAR Quick Smoke Test")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")

# Create small synthetic dataset (fake 32x32x3 images, 10 classes)
NUM_SAMPLES = 256
NUM_CLASSES = 10
IMG_SHAPE = (32, 32, 3)

print(f"\n[1/5] Creating synthetic dataset ({NUM_SAMPLES} samples, {IMG_SHAPE})...")
x_client = np.random.rand(NUM_SAMPLES, *IMG_SHAPE).astype(np.float32)
y_client = np.random.randint(0, NUM_CLASSES, (NUM_SAMPLES, 1))
x_server = np.random.rand(NUM_SAMPLES, *IMG_SHAPE).astype(np.float32)
y_server = np.random.randint(0, NUM_CLASSES, (NUM_SAMPLES, 1))

client_ds = tf.data.Dataset.from_tensor_slices((x_client, y_client)).shuffle(100, seed=42)
server_ds = tf.data.Dataset.from_tensor_slices((x_server, y_server)).shuffle(100, seed=42)
print("   OK - Synthetic datasets created.")

# Import SDAR components
print("\n[2/5] Importing SDAR modules...")
from sdar.sdar import SDARAttacker
from util.util import load_config
print("   OK - All modules imported.")

# Initialize SDAR attacker
print("\n[3/5] Initializing SDAR attacker (ResNet, level 7, vanilla SL)...")
config = load_config("sdar", "resnet", "cifar10", u_shape=False)
print(f"   Config: {config}")
sdar_attacker = SDARAttacker(client_ds, server_ds, num_class=NUM_CLASSES, batch_size=32)
print("   OK - Attacker initialized.")

# Run for just 10 iterations
NUM_ITER = 10
print(f"\n[4/5] Running SDAR for {NUM_ITER} iterations...")
history = sdar_attacker.run(
    level=7,
    num_iter=NUM_ITER,
    config=config,
    u_shape=False,
    conditional=True,
    model_type="resnet",
    width="standard",
    verbose_freq=5
)
print("   OK - Training loop completed.")

# Test the attack (reconstruction)
print("\n[5/5] Testing attack (image reconstruction)...")
eval_ds = client_ds.batch(32).take(1)
for (x, y) in eval_ds:
    x_recon, mse = sdar_attacker.attack(x, y)
    print(f"   Reconstruction MSE: {mse:.4f}")
    print(f"   Input shape: {x.shape}, Reconstructed shape: {x_recon.shape}")
    break

print("\n" + "=" * 60)
print("ALL TESTS PASSED - SDAR pipeline works correctly!")
print("=" * 60)
