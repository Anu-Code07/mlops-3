import joblib
import torch
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os

os.makedirs("models", exist_ok=True)

# Load the trained scikit-learn model
print("Loading scikit-learn model...")
sk_model = joblib.load("models/sklearn_model.joblib")

# Extract parameters
coef = sk_model.coef_
intercept = sk_model.intercept_

print(f"Original coefficients shape: {coef.shape}")
print(f"Original intercept shape: {intercept.shape}")
print(f"Original intercept value: {intercept}")

# Store unquantized parameters
unquant_params = {
    "coef": coef,
    "intercept": intercept
}
joblib.dump(unquant_params, "models/unquant_params.joblib")
print("✅ Saved unquantized parameters")

# Manual quantization to 8-bit unsigned integer
print("\nPerforming manual quantization...")

# Quantize coefficients
coef_min, coef_max = coef.min(), coef.max()
coef_scale = 255.0 / (coef_max - coef_min)
coef_zero_point = -coef_min * coef_scale

quantized_coef = np.round(coef * coef_scale + coef_zero_point).astype(np.uint8)

# For intercept (scalar), use a simple scaling approach
# Since intercept is a single value, we'll scale it relative to the coefficient range
intercept_scale = coef_scale  # Use same scale as coefficients
intercept_zero_point = coef_zero_point  # Use same zero point

quantized_intercept = np.round(intercept * intercept_scale + intercept_zero_point).astype(np.uint8)

# Store quantized parameters
quant_params = {
    "coef": quantized_coef,
    "intercept": quantized_intercept,
    "coef_scale": coef_scale,
    "coef_zero_point": coef_zero_point,
    "intercept_scale": intercept_scale,
    "intercept_zero_point": intercept_zero_point
}
joblib.dump(quant_params, "models/quant_params.joblib")
print("✅ Saved quantized parameters")

# Load test data
print("\nLoading test data...")
X, y = fetch_california_housing(return_X_y=True)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# De-quantize parameters
print("De-quantizing parameters...")
dequant_coef = (quantized_coef.astype(float) - coef_zero_point) / coef_scale
dequant_intercept = (quantized_intercept.astype(float) - intercept_zero_point) / intercept_scale

# Create PyTorch model with de-quantized weights
print("Creating PyTorch model...")
model = torch.nn.Linear(X_test.shape[1], 1)
model.weight.data = torch.tensor(dequant_coef.reshape(1, -1), dtype=torch.float32)
model.bias.data = torch.tensor([dequant_intercept], dtype=torch.float32)

# Perform inference
print("Running inference...")
model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_pred = model(X_test_tensor).numpy().flatten()

# Calculate R² score
r2_quantized = r2_score(y_test, y_pred)

# Also calculate R² for original sklearn model for comparison
y_pred_original = sk_model.predict(X_test)
r2_original = r2_score(y_test, y_pred_original)

print(f"\n{'='*50}")
print("MODEL COMPARISON RESULTS")
print(f"{'='*50}")
print(f"Original Sklearn Model R² Score: {r2_original:.6f}")
print(f"Quantized Model R² Score: {r2_quantized:.6f}")

# Calculate file sizes
import os
unquant_size = os.path.getsize("models/unquant_params.joblib") / 1024  # KB
quant_size = os.path.getsize("models/quant_params.joblib") / 1024  # KB

print(f"\nOriginal Model Size: {unquant_size:.2f} KB")
print(f"Quantized Model Size: {quant_size:.2f} KB")
print(f"Compression Ratio: {unquant_size/quant_size:.2f}x")

print(f"\n{'='*50}")
print("QUANTIZATION SUMMARY")
print(f"{'='*50}")
print(f"✅ Unquantized parameters saved: models/unquant_params.joblib")
print(f"✅ Quantized parameters saved: models/quant_params.joblib")
print(f"✅ Manual quantization completed successfully")
print(f"✅ PyTorch model created with de-quantized weights")
