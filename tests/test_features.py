
"""
Note: These tests will fail if you have not first trained the model.
"""
from sklearn.preprocessing import MinMaxScaler
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
import pandas as pd
from vehicleinsurance_model.config.core import config
from vehicleinsurance_model.processing.features import AnnualPremiumMinMaxScalar,Mapper
import pytest

@pytest.fixture
def sample_df():
    return pd.DataFrame({"Annual_Premium": [1000, 5000, 10000, 15000, 20000]})

def test_minmax_scaling(sample_df,sample_input_data):
    scaler = AnnualPremiumMinMaxScalar(variable=["Annual_Premium"])
    transformed_df = scaler.fit(sample_df.copy()).transform(sample_df.copy())
    
    expected_scaler = MinMaxScaler()
    # print(type(sample_input_data["Annual_Premium_var"]))
    # print(sample_input_data["Annual_Premium_var"])
    expected_scaler.fit(sample_input_data[0][["Annual_Premium"]])
    expected_values = expected_scaler.transform(sample_input_data[0][["Annual_Premium"]])
    assert np.allclose

def test_gender_variable_mapper(sample_input_data):
    # Given
    mapper = Mapper(variable = config.model_config_.Gender_var, 
                    mappings = config.model_config_.Gender_mappings)
    # print("Row 4 data:", sample_input_data[0].iloc[4]["Gender"])
    assert sample_input_data[0].iloc[4]["Gender"] == 'Male'

    # When
    subject = mapper.fit(sample_input_data[0]).transform(sample_input_data[0])
    # print("gender"+str(subject.iloc[4]["Gender"]))
    # Then
    assert subject.iloc[4]["Gender"]==1



