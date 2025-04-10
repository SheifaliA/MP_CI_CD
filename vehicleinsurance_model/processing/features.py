from typing import List
import sys
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

class Mapper(BaseEstimator, TransformerMixin):
    """
    Ordinal categorical variable mapper:
    Treat column as Ordinal categorical variable, and assign values accordingly
    """

    def __init__(self, variable:str, mappings:dict):
        #Type of column name
        if not isinstance(variable, str):
            raise ValueError("Gender should be a string")

        self.variable = variable
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # has_nan = X[self.variable].isna().any()  # Check for NaN
        # has_inf = np.isinf(X[self.variable]).any() 
        # print(":"+has_nan+","+has_inf)
        X[self.variable] = X[self.variable].map(self.mappings)

        return X

class ColumnStandardScalar(BaseEstimator, TransformerMixin):
    """Custom transformer to apply standard scaling to numerical features."""
    
    def __init__(self, variable:list):
        #Type of column name
        if not isinstance(variable, list):
            raise ValueError("Columns should be a list")
              
        self.variable = variable
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        """Fit the scaler to the numerical columns."""
        X = X.copy()
        if self.variable:
            # print(f"Type of input data: {type(X[self.variable])}")
            self.scaler.fit(X[self.variable])
        return self

    def transform(self, X):
        """Apply standard scaling to specified columns."""
        X=X.copy()
        if self.variable:
            X[self.variable] = self.scaler.transform(X[self.variable])
        return X
    
class AnnualPremiumMinMaxScalar(BaseEstimator, TransformerMixin):
    """Custom transformer to apply MinMax scaling to numerical features."""
    
    def __init__(self, variable:list):
        #Type of column name
        if not isinstance(variable, list):
            raise ValueError("AnnualPremium should be a str")
                
        self.variable = variable
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        """Fit the scaler to the numerical columns."""
        X = X.copy()
        print(f"fitType of self.variable: {type(self.variable)}")
        print(f"Value of self.variable: {self.variable}")

        if self.variable:
            X_subset= X[self.variable].copy()
            print(f"Type of input data: {type(X_subset)}")

            self.scaler.fit(X_subset)
        return self

    def transform(self, X):
        """Apply standard scaling to specified columns."""
        X=X.copy()
        print(f"transType of self.variable: {type(self.variable)}")
        print(f"Value of self.variable: {self.variable}")

        if self.variable:
            X[self.variable] = self.scaler.transform(X[self.variable])
        return X  
      
class ColumnOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encode weekday column """

    def __init__(self, variable:list):

        if not isinstance(variable, list):
            raise ValueError("Columns passed should be in a  list")
        self.variable= variable
        self.encoder = OneHotEncoder(sparse_output=False)

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        X = X.copy()
        if self.variable:
            self.encoder.fit(X[self.variable])
        # Get encoded feature names
            self.encoded_features_names = self.encoder.get_feature_names_out()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()      
        print(f"Type of self.variable: {type(self.variable)}")
        print(f"Value of self.variable: {self.variable}")
          
        encoded_cols = self.encoder.transform(X[self.variable])
        # Append encoded features to X
        print("Feature Names Learned:", self.encoder.get_feature_names_out())
        print("Variable Passed:", self.variable)

        X[self.encoded_features_names] = encoded_cols
        # drop original column after encoding
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
        """Apply renaming and integer conversion to specified columns."""
        X=X.copy()
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
        X=X.copy()
        """Drop the specified columns from the dataset."""
        if self.variable:
            X = X.drop(columns=self.variable, errors="ignore")  # Avoid errors if columns don't exist
        
        return X    