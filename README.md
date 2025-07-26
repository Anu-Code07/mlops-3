# MLOps Assignment 3

End-to-end MLOps pipeline using:
- Scikit-learn Linear Regression
- Manual quantization
- PyTorch conversion
- Docker + GitHub Actions CI/CD

## Branches
- main
- dev
- docker_ci
- quantization

## Setup

```bash
pip install -r requirements.txt
python src/train.py
python src/predict.py
python src/quantize.py
```

## Docker

```bash
docker build -t mlops-assignment-3 .
docker run mlops-assignment-3
```

## CI/CD

GitHub Actions workflow is defined in `.github/workflows/ci.yml`.
