import joblib
import torch
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os

os.makedirs("models", exist_ok=True)

sk_model = joblib.load("models/sklearn_model.joblib")
params = {"coef": sk_model.coef_, "intercept": sk_model.intercept_}
joblib.dump(params, "models/unquant_params.joblib")

scale = 255 / (params["coef"].max() - params["coef"].min())
zero_point = -params["coef"].min() * scale
quantized_coef = np.round(params["coef"] * scale + zero_point).astype(np.uint8)
quantized_intercept = np.round(params["intercept"] * scale + zero_point).astype(np.uint8)

quant_params = {
    "coef": quantized_coef,
    "intercept": quantized_intercept,
    "scale": scale,
    "zero_point": zero_point,
}
joblib.dump(quant_params, "models/quant_params.joblib")

X, y = fetch_california_housing(return_X_y=True)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dequant_coef = (quantized_coef.astype(float) - zero_point) / scale
dequant_intercept = (quantized_intercept.astype(float) - zero_point) / scale

model = torch.nn.Linear(X_test.shape[1], 1)
model.weight.data = torch.tensor(dequant_coef.reshape(1, -1), dtype=torch.float32)
model.bias.data = torch.tensor([dequant_intercept], dtype=torch.float32)

y_pred = model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()
r2 = r2_score(y_test, y_pred)

print("Quantized Model R² Score:", r2)
