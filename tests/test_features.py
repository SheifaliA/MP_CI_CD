from sklearn.preprocessing import MinMaxScaler
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from vehicleinsurance_model.config.core import config
from vehicleinsurance_model.processing.features import AnnualPremiumMinMaxScalar, Mapper

# Resolve file paths dynamically to support imports
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]

# Add parent directory to sys.path to allow module imports
sys.path.append(str(root))


@pytest.fixture
def sample_df():
    """Fixture to provide a sample dataset for scaling tests."""
    return pd.DataFrame({"Annual_Premium": [1000, 5000, 10000, 15000, 20000]})


def test_minmax_scaling(sample_df, sample_input_data):
    """Test Min-Max Scaling to verify correct transformation."""

    # Instantiate custom MinMaxScaler for the "Annual_Premium" column
    scaler = AnnualPremiumMinMaxScalar(variable=["Annual_Premium"])

    # Apply MinMaxScaler transformation
    transformed_df = scaler.fit(sample_df.copy()).transform(sample_df.copy())

    # Instantiate expected MinMaxScaler from sklearn
    expected_scaler = MinMaxScaler()

    # Fit expected scaler to sample input data and transform
    expected_scaler.fit(sample_input_data[0][["Annual_Premium"]])
    expected_values = expected_scaler.transform(sample_input_data[0][["Annual_Premium"]])

    # Ensure transformed values are close to expected ones
    assert np.allclose(transformed_df["Annual_Premium"].values, expected_values), \
        "MinMax scaling does not match expected values!"


def test_gender_variable_mapper(sample_input_data):
    """Test feature mapping for the 'Gender' variable."""

    # Given: Initialize the Mapper with configuration settings
    mapper = Mapper(
        variable=config.model_config_.Gender_var,
        mappings=config.model_config_.Gender_mappings
    )

    # Verify that row 4 initially contains 'Male'
    assert sample_input_data[0].iloc[4]["Gender"] == 'Male', \
        "Expected Gender to be 'Male' before transformation."

    # When: Apply the mapping transformation
    subject = mapper.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then: Verify that 'Male' is correctly mapped to numerical representation (1)
    assert subject.iloc[4]["Gender"] == 1, \
        "Expected mapped Gender to be 1 after transformation."