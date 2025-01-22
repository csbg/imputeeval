"""
This module provides the `ImputeEval` class, designed to evaluate imputation methods
by introducing artificial missing values into datasets and assessing their performance
using various metrics such as SMAPE, MAE, RMSE, NRMSE, and RAE. The class supports
controlled NA injection, mask generation, and evaluation of imputation quality in
a consistent and interpretable manner.
"""

# External imports ---------------------------------------------------------------------------------
import random
from typing import Tuple, Optional
import pandas as pd
import numpy as np


# Exported class definition ------------------------------------------------------------------------


class ImputeEval:
    """
    A class to evaluate imputation methods by introducing artificial NAs
    and calculating percentage error on the imputed values.

    Attributes
    ----------
    data : pd.DataFrame
        Original input data.
    na_mask : pd.DataFrame
        Boolean DataFrame indicating all missing (NA) values.
    artificial_na_mask : pd.DataFrame
        Boolean DataFrame indicating positions of artificially introduced NAs.
    modified_data : pd.DataFrame
        The input data with artificially introduced NAs.
    """

    def __init__(
            self,
            data: pd.DataFrame,
            percent_na_rows: int | float = 50,
            random_seed: Optional[int] = None
    ):
        """
        Initializes the ImputeEval class by introducing artificial NAs.

        Parameters
        ----------
        data : pd.DataFrame
            The input dataframe.

        percent_na_rows : float
            Percentage of rows where random non-NA values will be replaced
            by artificial NA values. If greater than 100, multiple rounds of
            NA introduction will be performed.

        random_seed : int, optional
            Seed for reproducibility. Default is None.
        """
        self._validate_init_inputs(
            data=data,
            percent_na_rows=percent_na_rows,
            random_seed=random_seed
        )

        self.data = data
        self.percent_na_rows = percent_na_rows
        self.random_seed = random_seed

        # Inject artificial NAs and create masks
        self.modified_data, self.artificial_na_mask = self._inject_artificial_na()
        self.na_mask = self.data.isna()  # Create a Boolean DataFrame mask for all NAs


    def get_data(self) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame
    ]:
        """
        Returns the modified data with artificial NAs and the masks.

        Returns
        -------
        pd.DataFrame
            Data with artificial NAs introduced.

        pd.DataFrame
            Boolean mask for all NA values.

        pd.DataFrame
            Boolean mask for artificially introduced NAs.
        """
        na_mask = self.data.isna()  # Create a Boolean mask for all NAs
        return self.modified_data, na_mask, self.artificial_na_mask

    def evaluate(
            self,
            imputed_data: pd.DataFrame,
            metric: str = "SMAPE"
    ) -> float:
        """
        Evaluates the imputation performance using a specified metric.

        Parameters
        ----------
        imputed_data : pd.DataFrame
            The dataframe with imputed values.

        metric : str, optional
            The evaluation metric to use. Supported metrics include:

            - "SMAPE" (default): Symmetric Mean Absolute Percentage Error.
              Scale-independent. Formula:
              SMAPE = (|predicted - true| / ((|predicted| + |true|) / 2)) * 100

            - "MAE": Mean Absolute Error.
              Scale-dependent. Formula:
              MAE = mean(|predicted - true|)

            - "RMSE": Root Mean Squared Error.
              Scale-dependent. Formula:
              RMSE = sqrt(mean((predicted - true)^2))

            - "NRMSE": Normalized Root Mean Squared Error.
              Scale-independent. Formula:
              NRMSE = RMSE / (max(true) - min(true))

            - "RAE": Relative Absolute Error (compares with mean imputation).
              Scale-independent. Formula:
              RAE = sum(|predicted - true|) / sum(|mean(true) - true|)

        Returns
        -------
        float
            The calculated error or evaluation score.

        Raises
        ------
        ValueError
            If the provided metric is not supported.
        """
        self._validate_evaluate_method_inputs(
            imputed_data=imputed_data,
            metric=metric
        )

        # Create a mapping of true values and predicted values
        true_values = self.data[self.artificial_na_mask].values.flatten()
        predicted_values = imputed_data[self.artificial_na_mask].values.flatten()

        # Remove any NaN values from true or predicted
        valid_indices = ~pd.isna(true_values) & ~pd.isna(predicted_values)
        true_values = true_values[valid_indices]
        predicted_values = predicted_values[valid_indices]

        # Map metric names to their corresponding methods
        metrics_map = {
            "SMAPE": self._compute_smape,
            "MAE": self._compute_mae,
            "RMSE": self._compute_rmse,
            "NRMSE": self._compute_nrmse,
            "RAE": self._compute_rae
        }
        metric_func = metrics_map.get(metric)    # Get the selected metric function
        return metric_func(true_values, predicted_values) # Compute and return the selected metric


    # Level 1 methods ------------------------------------------------------------------------------

    @staticmethod
    def _validate_init_inputs(
            data: pd.DataFrame,
            percent_na_rows: int | float,
            random_seed: Optional[int]
    ) -> None:
        """
        Validates the inputs provided to the ImputeEval class.

        Parameters
        ----------
        data : pd.DataFrame
            The input dataframe to be validated.

        percent_na_rows : int | float
            Percentage of rows where artificial NAs will be introduced.

        random_seed : int, optional
            Seed for reproducibility, which must be an integer or None.

        Raises
        ------
        TypeError
            If any input is of the wrong type.

        ValueError
            If any input value is invalid.
        """
        # Validate `data`
        if not isinstance(data, pd.DataFrame):
            raise TypeError("`data` must be a Pandas DataFrame.")
        if not all(data.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x) or pd.isna(x))):
            raise ValueError("`data` must only contain numeric values or NA values.")

        # Validate `percent_na_rows`
        if not isinstance(percent_na_rows, (int, float)):
            raise TypeError("`percent_na_rows` must be an integer or float.")
        if percent_na_rows <= 0:
            raise ValueError("`percent_na_rows` must be greater than 0.")

        # Validate `random_seed`
        if random_seed is not None and not isinstance(random_seed, int):
            raise TypeError("`random_seed` must be an integer or None.")

    def _inject_artificial_na(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Injects artificial NAs into the data based on percent_na_rows.

        Returns
        -------
        pd.DataFrame
            Data with artificial NAs introduced.

        pd.DataFrame
            Boolean DataFrame indicating positions of artificial NAs.
        """
        rng = random.Random(self.random_seed)
        modified_data = self.data.copy()
        artificial_na_mask = pd.DataFrame(
            False,
            index=self.data.index,
            columns=self.data.columns
        )

        total_rows = len(self.data)
        num_na_rounds = int(self.percent_na_rows // 100)
        remaining_percent = self.percent_na_rows % 100

        # Perform full rounds of NA introduction
        for _ in range(num_na_rounds):
            self._introduce_nas(
                modified_data,
                artificial_na_mask,
                total_rows,
                rng, percent=100
            )

        # Introduce remaining NAs
        if remaining_percent > 0:
            self._introduce_nas(
                modified_data,
                artificial_na_mask,
                total_rows,
                rng,
                percent=remaining_percent
            )

        return modified_data, artificial_na_mask

    def _validate_evaluate_method_inputs(
            self,
            imputed_data: pd.DataFrame,
            metric: str
    ) -> None:
        """
        Validates the inputs provided to the `evaluate` method.

        Parameters
        ----------
        imputed_data : pd.DataFrame
            The dataframe with imputed values to be validated.

        metric : str
            The evaluation metric to use.

        Raises
        ------
        TypeError
            If the input types are incorrect.

        ValueError
            If the input values are invalid.
        """
        # Validate `imputed_data`
        if not isinstance(imputed_data, pd.DataFrame):
            raise TypeError("`imputed_data` must be a Pandas DataFrame.")
        if not all(imputed_data.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x))):
            raise ValueError("`imputed_data` must only contain numeric values.")

        # Ensure `imputed_data` aligns with the original data
        if not imputed_data.shape == self.data.shape:
            raise ValueError("`imputed_data` must have the same shape as the original data.")

        # Validate `metric`
        if not isinstance(metric, str):
            raise TypeError("`metric` must be a string.")
        supported_metrics = ["SMAPE", "MAE", "RMSE", "NRMSE", "RAE"]
        if metric not in supported_metrics:
            raise ValueError(
                f"Unsupported metric: {metric}. Supported metrics are: {', '.join(supported_metrics)}."
            )

    @staticmethod
    def _compute_smape(
            true_values: np.ndarray,
            predicted_values: np.ndarray
    ) -> float:
        """
        Computes the Symmetric Mean Absolute Percentage Error (SMAPE).
        """
        numerator = np.abs(predicted_values - true_values)
        denominator = (np.abs(predicted_values) + np.abs(true_values)) / 2
        epsilon = 1e-8  # Avoid division by zero
        smape = (numerator / (denominator + epsilon)) * 100
        return smape.mean()

    @staticmethod
    def _compute_mae(
            true_values: np.ndarray,
            predicted_values: np.ndarray
    ) -> float:
        """
        Computes the Mean Absolute Error (MAE).
        """
        return np.abs(predicted_values - true_values).mean()

    @staticmethod
    def _compute_rmse(
            true_values: np.ndarray,
            predicted_values: np.ndarray
    ) -> float:
        """
        Computes the Root Mean Squared Error (RMSE).
        """
        return np.sqrt(((predicted_values - true_values) ** 2).mean())

    @staticmethod
    def _compute_nrmse(
            true_values: np.ndarray,
            predicted_values: np.ndarray
    ) -> float:
        """
        Computes the Normalized Root Mean Squared Error (NRMSE).
        """
        data_range = true_values.max() - true_values.min()
        rmse = np.sqrt(((predicted_values - true_values) ** 2).mean())
        return rmse / data_range

    @staticmethod
    def _compute_rae(
            true_values: np.ndarray,
            predicted_values: np.ndarray
    ) -> float:
        """
        Computes the Relative Absolute Error (RAE).
        """
        baseline_error = np.abs(true_values - true_values.mean()).sum()
        model_error = np.abs(predicted_values - true_values).sum()
        return model_error / baseline_error

    # Level 2 methods ------------------------------------------------------------------------------

    @staticmethod
    def _introduce_nas(
            current_data: pd.DataFrame,
            current_mask: pd.DataFrame,
            total_rows: int,
            rng: random.Random,
            percent: float
    ) -> None:
        """
        Introduces artificial NAs into a percentage of rows in the dataframe.

        Parameters
        ----------
        current_data : pd.DataFrame
            The data into which NAs will be introduced.

        current_mask : pd.DataFrame
            The mask indicating where artificial NAs are introduced.

        total_rows : int
            Total number of rows in the dataset.

        rng : random.Random
            A random generator for reproducibility.

        percent : float
            Percentage of rows to introduce NAs into.
        """
        num_na_rows = int((percent / 100) * total_rows)
        eligible_rows = [idx for idx in current_data.index if
                         not current_data.loc[idx].isna().all()]
        modified_rows = rng.sample(eligible_rows, min(num_na_rows, len(eligible_rows)))

        for row_idx in modified_rows:
            non_na_columns = current_data.loc[row_idx].dropna().index.tolist()

            if len(non_na_columns) > 1:  # Ensure there are at least 2 non-NA values
                col_idx = rng.choice(non_na_columns)
                current_data.loc[row_idx, col_idx] = float('nan')
                current_mask.loc[row_idx, col_idx] = True
