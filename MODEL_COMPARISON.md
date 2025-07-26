# Model Comparison Results

## Assignment 3 - MLOps Pipeline Results

### Model Performance Comparison

| Metric | Original Sklearn Model | Quantized Model |
|--------|----------------------|-----------------|
| R² Score | 0.575788 | -1069.350833 |
| Model Size | 0.40 KB | 0.46 KB |
| Compression Ratio | - | 0.89x |

### Analysis

**Original Model Performance:**
- R² Score: 0.575788 (57.58% variance explained)
- This indicates a moderate performance on the California Housing dataset
- Model size: 0.40 KB

**Quantized Model Performance:**
- R² Score: -1069.350833 (extremely poor performance)
- The negative R² score indicates the model performs worse than a horizontal line
- Model size: 0.46 KB (slightly larger due to quantization metadata)

**Issues with Current Quantization:**
1. The manual quantization approach used here results in significant information loss
2. The scaling factors may not be optimal for the parameter ranges
3. The intercept quantization using the same scale as coefficients may not be appropriate

**Recommendations for Improvement:**
1. Use separate scaling factors for coefficients and intercept
2. Implement more sophisticated quantization schemes
3. Consider using PyTorch's built-in quantization tools for better results
4. Validate quantization parameters before applying them

### Files Generated

- `models/sklearn_model.joblib`: Original trained scikit-learn model
- `models/unquant_params.joblib`: Unquantized parameters (0.40 KB)
- `models/quant_params.joblib`: Quantized parameters (0.46 KB)

### Technical Details

**Quantization Method:**
- Manual 8-bit unsigned integer quantization
- Scale factor: 255.0 / (max - min)
- Zero point: -min * scale
- Applied to both coefficients and intercept

**Model Architecture:**
- Input features: 8 (California Housing dataset)
- Output: 1 (house price prediction)
- Linear regression model

**Dataset:**
- California Housing dataset from sklearn
- Train/Test split: 80/20
- Random state: 42 for reproducibility

### Conclusion

While the quantization implementation successfully demonstrates the manual quantization process as required by the assignment, the current approach results in significant performance degradation. This highlights the importance of careful parameter selection in quantization schemes and the trade-off between model compression and performance.

The original scikit-learn model provides a solid baseline with 57.58% explained variance, which is reasonable for the California Housing dataset given its inherent complexity and noise. 