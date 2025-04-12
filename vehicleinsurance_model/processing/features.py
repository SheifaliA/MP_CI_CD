from typing import List
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

class Mapper(BaseEstimator, TransformerMixin):
    """
    Ordinal categorical variable mapper.
    Treats the specified column as an ordinal categorical variable and assigns numerical values accordingly.
    """

    def __init__(self, variable: str, mappings: dict):
        if not isinstance(variable, str):
            raise ValueError("The variable name should be a string.")

        self.variable = variable
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Required for compatibility with sklearn pipelines."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies mapping transformation to the specified column."""
        X = X.copy()
        X[self.variable] = X[self.variable].map(self.mappings)
        return X


class ColumnStandardScalar(BaseEstimator, TransformerMixin):
    """Custom transformer to apply standard scaling to numerical features."""

    def __init__(self, variable: list):
        if not isinstance(variable, list):
            raise ValueError("Column names should be provided as a list.")

        self.variable = variable
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        """Fits the scaler to the specified numerical columns."""
        X = X.copy()
        if self.variable:
            self.scaler.fit(X[self.variable])
        return self

    def transform(self, X):
        """Applies standard scaling to the specified columns."""
        X = X.copy()
        if self.variable:
            X[self.variable] = self.scaler.transform(X[self.variable])
        return X


class AnnualPremiumMinMaxScalar(BaseEstimator, TransformerMixin):
    """Custom transformer to apply MinMax scaling to the Annual Premium column."""

    def __init__(self, variable: list):
        if not isinstance(variable, list):
            raise ValueError("AnnualPremium should be provided as a list.")

        self.variable = variable
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        """Fits the scaler to the specified numerical columns."""
        X = X.copy()
        if self.variable:
            X_subset = X[self.variable].copy()
            self.scaler.fit(X_subset)
        return self

    def transform(self, X):
        """Applies MinMax scaling to the specified columns."""
        X = X.copy()
        if self.variable:
            X[self.variable] = self.scaler.transform(X[self.variable])
        return X


class ColumnOneHotEncoder(BaseEstimator, TransformerMixin):
    """Custom transformer to apply one-hot encoding to categorical features."""

    def __init__(self, variable: list):
        if not isinstance(variable, list):
            raise ValueError("Columns should be provided in a list.")

        self.variable = variable
        self.encoder = OneHotEncoder(sparse_output=False)

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fits the encoder to categorical columns."""
        X = X.copy()
        if self.variable:
            self.encoder.fit(X[self.variable])
            self.encoded_features_names = self.encoder.get_feature_names_out()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies one-hot encoding and replaces the original categorical columns."""
        X = X.copy()
        encoded_cols = self.encoder.transform(X[self.variable])

        # Append encoded features and remove original column
        X[self.encoded_features_names] = encoded_cols
        X.drop(self.variable, axis=1, inplace=True)
        return X


class RenameColumnsTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer to rename specific columns and ensure integer types for dummy variables."""

    def __init__(self):
        self.rename_map = {
            "Vehicle_Age_var_< 1 Year": "Vehicle_Age_var_lt_1_Year",
            "Vehicle_Age_var_> 2 Years": "Vehicle_Age_var_gt_2_Years"
        }
        self.int_columns = ["Vehicle_Age_var_lt_1_Year", "Vehicle_Age_var_gt_2_Years", "Vehicle_Damage_var_Yes"]

    def fit(self, X, y=None):
        """Fit method (required but not used in this case)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Renames columns and ensures integer type for selected categorical variables."""
        X = X.copy()
        X = X.rename(columns=self.rename_map)
        for col in self.int_columns:
            if col in X.columns:
                X[col] = X[col].astype('int')
        return X


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer to drop specified columns from the dataset."""

    def __init__(self, variable=None):
        self.variable = variable

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Drops the specified columns from the dataset, avoiding errors if columns don't exist."""
        X = X.copy()
        if self.variable:
            X = X.drop(columns=self.variable, errors="ignore")
        return X