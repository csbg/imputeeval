import pytest
import pandas as pd
import numpy as np
from imputeeval import ImputeEval  # Replace with your package import path
import random

# Fixture for sample data
@pytest.fixture
def sample_data():
    """
     Provides a sample DataFrame for testing purposes.

     The DataFrame contains a mix of numerical values and NaNs to test
     the functionality of imputation methods, mask generation, and
     evaluation workflows.

     Returns
     -------
     pd.DataFrame
         A DataFrame with the following structure:
             A    B     C
         0  1.0  4.0   7.0
         1  2.0  5.0   8.0
         2  3.0  NaN   9.0
         3  NaN  6.0  10.0
     """
    return pd.DataFrame({
        "A": [1, 2, 3, np.nan],
        "B": [4, 5, np.nan, 6],
        "C": [7, 8, 9, 10]
    })


# Tests for API methods ----------------------------------------------------------------------------


@pytest.mark.usefixtures("sample_data")
def test_get_data(sample_data):
    # Test for valid case with artificial NAs introduced
    evaluator = ImputeEval(
        data=sample_data,
        percent_na_rows=60,
        random_seed=42
    )

    modified_data, na_mask, artificial_na_mask = evaluator.get_data()

    # Assert modified_data has NAs introduced
    assert modified_data.isna().any().any(), "Artificial NAs should be introduced."

    # Assert na_mask correctly identifies all NAs
    assert na_mask.equals(sample_data.isna()), "na_mask should match original NA positions."

    # Assert artificial_na_mask only marks artificially introduced NAs
    assert artificial_na_mask.values.sum() > 0, "Artificial NA mask should have True values."


@pytest.mark.usefixtures("sample_data")
def test_evaluate(sample_data):
    evaluator = ImputeEval(
        data=sample_data,
        percent_na_rows=100,
        random_seed=42
    )

    # Get modified data
    modified_data, _, artificial_na_mask = evaluator.get_data()

    # Perform simple mean imputation
    imputed_data = modified_data.apply(
        lambda row: row.fillna(row.mean()), axis=1
    )

    # List of supported metrics
    metrics = ["SMAPE", "MAE", "RMSE", "NRMSE", "RAE"]

    # Iterate over each metric and evaluate
    for metric in metrics:
        avg_error = evaluator.evaluate(imputed_data, metric=metric)

        # Assert the error is computed correctly
        assert isinstance(avg_error, float), f"Evaluation with {metric} should return a float."
        assert avg_error >= 0, f"{metric} should be non-negative."


# Tests for internal methods -----------------------------------------------------------------------


@pytest.mark.usefixtures("sample_data")
def test_introduce_nas(sample_data):
    mask = pd.DataFrame(
        False,
        index=sample_data.index,
        columns=sample_data.columns
    )

    rng = random.Random(42)

    # Call _introduce_nas directly
    ImputeEval._introduce_nas(
        current_data=sample_data,
        current_mask=mask,
        total_rows=len(sample_data),
        rng=rng,
        percent=100  # Introduce NAs into all rows
    )

    # Assert NAs are introduced in all rows
    assert mask.values.sum() > 0, "Artificial NAs should be introduced."
    assert sample_data.isna().any().any(), "Data should contain NAs."

    # Assert that at least one column per row is still non-NA
    for _, row in sample_data.iterrows():
        assert row.dropna().shape[0] >= 1, "Each row should have at least one non-NA value."


@pytest.mark.usefixtures("sample_data")
def test_inject_artificial_na(sample_data):
    evaluator = ImputeEval(
        data=sample_data,
        percent_na_rows=150,
        random_seed=42
    )

    # Call _inject_artificial_na directly
    modified_data, artificial_na_mask = evaluator._inject_artificial_na()

    # Assert NAs are introduced
    assert artificial_na_mask.values.sum() > 0, "Artificial NAs should be introduced."
    assert modified_data.isna().any().any(), "Modified data should contain NAs."

    # Assert at least one column per row is non-NA
    for _, row in modified_data.iterrows():
        assert row.dropna().shape[0] >= 1, "Each row should have at least one non-NA value."

    # Assert correct percentage of rows have artificial NAs
    num_rows_with_nas = (artificial_na_mask.sum(axis=1) > 0).sum()
    expected_rows_with_nas = int(len(sample_data) * (150 / 100))  # 150% means all rows + 50%
    assert num_rows_with_nas >= len(sample_data), "All rows should have at least one NA."
    assert num_rows_with_nas <= expected_rows_with_nas, "Extra rows should not exceed 50% more."
