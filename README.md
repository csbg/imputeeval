# imputeeval

`imputeeval` is a Python package designed for benchmarking imputation tools. It evaluates 
their performance by introducing artificial missing values (NAs) into datasets, allowing you to 
assess how well the imputed values match the original hidden values. With `imputeeval`, you can 
fairly and transparently compare different imputation methods using various evaluation metrics.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

Install `imputeeval` directly from GitHub using pip:

```bash
pip install git+https://github.com/Thomas-Rauter/imputeeval@v0.1.0
```

## Usage

Below is a simple example of how to use `imputeeval` to benchmark imputation methods using the 
Symmetric Mean Absolute Percentage Error (SMAPE) metric.

```python
import pandas as pd
import numpy as np
from imputeeval import ImputeEval

# Sample dataset
sample_data = pd.DataFrame({
    "A": [1, 2, 3, np.nan],
    "B": [4, 5, np.nan, 6],
    "C": [7, 8, 9, 10]
})

# Initialize the evaluator
evaluator = ImputeEval(
    data=sample_data,
    percent_na_rows=50,
    random_seed=42
)

# Get modified data with artificial NAs
modified_data, na_mask, artificial_na_mask = evaluator.get_data()

# Imputation Method 1: Simple mean imputation
imputed_data_mean = modified_data.apply(
    lambda row: row.fillna(row.mean()), axis=1
)

# Imputation Method 2: Zero imputation
imputed_data_zero = modified_data.fillna(0)

# Evaluate the imputation methods using SMAPE
smape_mean = evaluator.evaluate(
    imputed_data_mean,
    metric="SMAPE"
)
smape_zero = evaluator.evaluate(
    imputed_data_zero,
    metric="SMAPE"
)

# Print results
print(f"Mean Imputation SMAPE: {smape_mean:.2f}%")
print(f"Zero Imputation SMAPE: {smape_zero:.2f}%")

# Compare the methods
if smape_mean < smape_zero:
    print("Mean imputation performed better.")
else:
    print("Zero imputation performed better.")
    
# Output:
# Mean Imputation SMAPE: 72.37%
# Zero Imputation SMAPE: 200.00%
# Mean imputation performed better.
```
