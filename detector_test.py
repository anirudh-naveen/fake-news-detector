# Test script to isolate the bus error issue
print("Testing imports...")

try:
    import numpy as np
    print("✓ NumPy imported successfully")
except Exception as e:
    print(f"✗ NumPy import failed: {e}")

try:
    import pandas as pd
    print("✓ Pandas imported successfully")
except Exception as e:
    print(f"✗ Pandas import failed: {e}")

try:
    import tensorflow as tf
    print("✓ TensorFlow imported successfully")
    print(f"TensorFlow version: {tf.__version__}")
except Exception as e:
    print(f"✗ TensorFlow import failed: {e}")
    print("This is likely the cause of the bus error on macOS")

try:
    from sklearn.model_selection import train_test_split
    from sklearn import preprocessing
    print("✓ Scikit-learn imported successfully")
except Exception as e:
    print(f"✗ Scikit-learn import failed: {e}")

print("\nTesting CSV read...")
try:
    data = pd.read_csv("news.csv")
    print(f"✓ CSV read successfully! Shape: {data.shape}")
    print("\nFirst 5 rows:")
    print(data.head())
except Exception as e:
    print(f"✗ CSV read failed: {e}")

