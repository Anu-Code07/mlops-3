# MLOps Assignment 3

End-to-end MLOps pipeline using:
- Scikit-learn Linear Regression
- Manual quantization
- PyTorch conversion
- Docker + GitHub Actions CI/CD

## Branches
- **main**: Initial setup with README.md and .gitignore
- **dev**: Model development with train.py
- **docker_ci**: Docker automation and CI/CD
- **quantization**: Model conversion and optimization

## Project Structure
```
mlops_assignment_3_code/
├── .github/workflows/ci.yml    # CI/CD workflow
├── src/
│   ├── train.py               # Train scikit-learn model
│   ├── predict.py             # Model prediction and verification
│   └── quantize.py            # Manual quantization
├── models/                    # Saved models
├── Dockerfile                 # Container configuration
├── docker_simulator.py        # Docker simulation script
├── docker.sh                  # Shell-based Docker simulator
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Setup

### Prerequisites
```bash
pip install -r requirements.txt
```

### Training
```bash
python src/train.py
```

### Prediction
```bash
python src/predict.py
```

### Quantization
```bash
python src/quantize.py
```

## Docker Simulation

Since this is a local development setup, we provide Docker simulation tools that mimic real Docker operations:

### Using Python Simulator
```bash
# Full pipeline
python docker_simulator.py

# Individual commands
python docker_simulator.py build
python docker_simulator.py run
python docker_simulator.py push
```

### Using Shell Script (Linux/Mac)
```bash
# Make executable
chmod +x docker.sh

# Full pipeline
./docker.sh full

# Individual commands
./docker.sh build
./docker.sh run
./docker.sh push
```

### Using PowerShell (Windows)
```bash
# Full pipeline
python docker_simulator.py

# Individual commands
python docker_simulator.py build
python docker_simulator.py run
python docker_simulator.py push
```

## CI/CD

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs on pushes to the `docker_ci` branch and:

1. Sets up Python environment
2. Installs dependencies
3. Trains the model
4. Simulates Docker build
5. Simulates Docker run
6. Simulates Docker push
7. Runs quantization

## Model Comparison

After running quantization, you can compare the original and quantized models:

| Metric | Original Sklearn Model | Quantized Model |
|--------|----------------------|-----------------|
| R² Score | [Your result] | [Your result] |
| Model Size | [size of unquant_params.joblib] KB | [size of quant_params.joblib] KB |

## Branch Workflow

1. **main**: Contains initial setup
2. **dev**: Contains model training code (do not merge to main)
3. **docker_ci**: Contains Docker and CI/CD setup (do not merge to main)
4. **quantization**: Contains quantization implementation (do not merge to main)

## Files Generated

- `models/sklearn_model.joblib`: Trained scikit-learn model
- `models/unquant_params.joblib`: Unquantized parameters
- `models/quant_params.joblib`: Quantized parameters

## Notes

- This setup simulates Docker operations locally for development purposes
- The CI/CD workflow uses the Docker simulator instead of real Docker
- All branches remain separate as per assignment requirements
- Quantization is performed manually without using PyTorch's built-in quantization
