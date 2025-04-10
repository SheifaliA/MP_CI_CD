import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from vehicleinsurance_model.config.core import config
from vehicleinsurance_model.processing.features import Mapper
from vehicleinsurance_model.processing.features import ColumnStandardScalar
from vehicleinsurance_model.processing.features import AnnualPremiumMinMaxScalar
from vehicleinsurance_model.processing.features import ColumnOneHotEncoder
from vehicleinsurance_model.processing.features import RenameColumnsTransformer
from vehicleinsurance_model.processing.features import DropColumnsTransformer

vehicleinsurance_pipe = Pipeline([     
    ######### Mapper ###########
    ('map_gender', Mapper(variable = config.model_config_.Gender_var, mappings = config.model_config_.Gender_mappings)),
    ##########Scaling##########
    ('scale_cols', ColumnStandardScalar(variable =[config.model_config_.Age_var,config.model_config_.Vintage_var])),
    ('scale_annualpremium', AnnualPremiumMinMaxScalar(variable = [config.model_config_.Annual_Premium_var])),
    ######## One-hot encoding ########
    ('encode_cols', ColumnOneHotEncoder(variable=[config.model_config_.Vehicle_Age_var, config.model_config_.Vehicle_Damage_var])),
    ##########Rename#########################
    ('renamecolumnstransformer', RenameColumnsTransformer()),
    ##############Drop#############
    ('dropcolumnstransformer', DropColumnsTransformer(variable = config.model_config_.id_var)),
    # Regressor
    ('model_rf', RandomForestClassifier(n_estimators = config.model_config_.N_ESTIMATORS, 
                                        max_depth = config.model_config_.MAX_DEPTH,
                                        random_state = config.model_config_.RANDOM_STATE,
                                        min_samples_split= config.model_config_.MIN_SAMPLES_SPLIT, 
                                        min_samples_leaf=  config.model_config_.MIN_SAMPLES_LEAF,
                                        criterion = config.model_config_.CRITERION))
    
    ])
