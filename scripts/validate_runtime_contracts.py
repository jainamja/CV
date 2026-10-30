import os
import sys
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("=" * 70)
print("STAGE 4: LOCAL RUNTIME CONTRACT & COMPATIBILITY VALIDATION")
print("=" * 70)

# 1. Imports
print("1. Validating Module Imports:")
try:
    import cv2
    print(f"   [OK] OpenCV imported (version {cv2.__version__})")
except Exception as e:
    print(f"   [FAIL] OpenCV import error: {e}")

try:
    import numpy as np
    print(f"   [OK] NumPy imported (version {np.__version__})")
except Exception as e:
    print(f"   [FAIL] NumPy import error: {e}")

try:
    import tensorflow as tf
    from tensorflow import keras
    print(f"   [OK] TensorFlow / Keras imported (TF {tf.__version__}, Keras {keras.__version__})")
except Exception as e:
    print(f"   [FAIL] TensorFlow import error: {e}")

try:
    import streamlit as st
    print(f"   [OK] Streamlit imported (version {st.__version__})")
except Exception as e:
    print(f"   [FAIL] Streamlit import error: {e}")

try:
    import sympy as sp
    print(f"   [OK] SymPy imported (version {sp.__version__})")
except Exception as e:
    print(f"   [FAIL] SymPy import error: {e}")

try:
    import joblib
    print(f"   [OK] joblib imported (version {joblib.__version__})")
except Exception as e:
    print(f"   [FAIL] joblib import error: {e}")

try:
    from PIL import Image
    print(f"   [OK] PIL / Pillow imported")
except Exception as e:
    print(f"   [FAIL] PIL import error: {e}")

# 2. Model Loading
print("\n2. Validating Model Loading:")
cnn_path = "models/cnn_mathvision.keras"
if os.path.exists(cnn_path):
    m = keras.models.load_model(cnn_path)
    print(f"   [OK] Successfully loaded {cnn_path} (Inputs: {m.input_shape}, Outputs: {m.output_shape})")
else:
    print(f"   [FAIL] Missing model file at {cnn_path}")

svm_path = "models/svm_baseline.joblib"
if os.path.exists(svm_path):
    svm = joblib.load(svm_path)
    print(f"   [OK] Successfully loaded {svm_path}")
else:
    print(f"   [NOTE] Optional SVM file missing at {svm_path} (CNN remains primary)")

# 3. Required Runtime Asset Paths
print("\n3. Validating Required Runtime Asset Paths:")
assets = [
    ("app.py", os.path.exists("app.py")),
    ("ui/styles.css", os.path.exists("ui/styles.css")),
    ("ui/components.py", os.path.exists("ui/components.py")),
    (".streamlit/config.toml", os.path.exists(".streamlit/config.toml")),
    ("models/cnn_mathvision.keras", os.path.exists("models/cnn_mathvision.keras")),
    ("models/test_split.npz", os.path.exists("models/test_split.npz")),
]
for name, exists in assets:
    status = "[OK]" if exists else "[FAIL]"
    print(f"   {status} {name}")

samples = sorted(glob.glob("data/samples/*.png"))
print(f"   {'[OK]' if len(samples) >= 5 else '[FAIL]'} data/samples/*.png ({len(samples)} sample images found)")

# 4. Streamlit config validation
print("\n4. Validating Streamlit Configuration:")
with open(".streamlit/config.toml") as f:
    cfg_text = f.read()
    print("   .streamlit/config.toml contents:")
    for line in cfg_text.strip().splitlines():
        print(f"     {line}")

# 5. Pipeline End-to-End Test on Sample 01
print("\n5. Validating End-to-End Pipeline Execution on sample_01.png:")
from src.pipeline import run_pipeline
res = run_pipeline("data/samples/sample_01.png")
print(f"   Pipeline Status : {res.get('status')}")
print(f"   Expression      : {res.get('expression_parsed')}")
print(f"   Solution Answer : {res.get('solution', {}).get('final_answer')}")
assert res.get("status") == "success"
print("   [OK] Pipeline executed with zero exceptions.")

print("\nSTAGE 4 RUNTIME CONTRACT VALIDATION COMPLETE.")
