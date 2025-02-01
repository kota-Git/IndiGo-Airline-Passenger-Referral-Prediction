# code part  writing in  constant file
# Import necessary libraries
import os
from datetime import date
from pathlib import Path
from dotenv import load_dotenv
load_dotenv() 

DATABASE_NAME="Referal-Prediction_DB"

COLLECTION_NAME="Passengers_refer_data"

MONGODB_URL_KEY="MONGODB_URL"

PIPELINE_NAME:str ="src"  # NOT A SRC its pipeline name

ARTIFICAT_DIR:str="artifacat"

MODEL_FILE_NAME="model.pkl"

TARGET_COLUMN="recommended"

PREPROCESSING_OBJECT_FILE_NAME="preprocessing.pkl"


FILE_NAME:str="Passengers_referal_data.csv"

TRAIN_FILE_NAME:str="train.csv"

TEST_FILE_NAME:str="test.csv"

SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")


"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "Passengers_refer_data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2





# code for constant file for datavalidation
"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"


# code for constant_file for data_transformation

"""
Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"



# code for constant_file for model_trainer

"""
MODEL TRAINER related constant start with MODEL_TRAINER var name
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")