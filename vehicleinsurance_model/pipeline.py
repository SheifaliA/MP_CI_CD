import sys
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from vehicleinsurance_model.config.core import config
from vehicleinsurance_model.processing.features import (
    Mapper,
    ColumnStandardScalar,
    AnnualPremiumMinMaxScalar,
    ColumnOneHotEncoder,
    RenameColumnsTransformer,
    DropColumnsTransformer,
)

# Dynamically resolve file paths for module imports
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]

# Add parent directory to Python's module search path
sys.path.append(str(root))


# Define the machine learning pipeline
vehicleinsurance_pipe = Pipeline([
    ######### Ordinal Encoding ###########
    ('map_gender', Mapper(
        variable=config.model_config_.Gender_var,
        mappings=config.model_config_.Gender_mappings
    )),  # Converts 'Male' and 'Female' into numerical values

    ########## Feature Scaling ##########
    ('scale_cols', ColumnStandardScalar(variable=[
        config.model_config_.Age_var, config.model_config_.Vintage_var
    ])),  # Standard scales Age and Vintage features

    ('scale_annualpremium', AnnualPremiumMinMaxScalar(variable=[
        config.model_config_.Annual_Premium_var
    ])),  # Scales Annual Premium using MinMaxScaler

    ######## One-Hot Encoding ########
    ('encode_cols', ColumnOneHotEncoder(variable=[
        config.model_config_.Vehicle_Age_var, config.model_config_.Vehicle_Damage_var
    ])),  # Converts categorical Vehicle Age & Damage features into numerical representation

    ########## Column Renaming ##########
    ('renamecolumnstransformer', RenameColumnsTransformer()),  # Renames specific categorical columns for consistency

    ############## Column Dropping #############
    ('dropcolumnstransformer', DropColumnsTransformer(
        variable=config.model_config_.id_var
    )),  # Drops ID column to prevent data leakage

    ########## Machine Learning Model ##########
    ('model_rf', RandomForestClassifier(
        n_estimators=config.model_config_.N_ESTIMATORS,  # Number of decision trees
        max_depth=config.model_config_.MAX_DEPTH,  # Maximum tree depth
        random_state=config.model_config_.RANDOM_STATE,  # Ensures reproducibility
        min_samples_split=config.model_config_.MIN_SAMPLES_SPLIT,  # Minimum samples needed for a split
        min_samples_leaf=config.model_config_.MIN_SAMPLES_LEAF,  # Minimum samples per leaf node
        criterion=config.model_config_.CRITERION  # Decision tree split criterion ('entropy' or 'gini')
    ))
])