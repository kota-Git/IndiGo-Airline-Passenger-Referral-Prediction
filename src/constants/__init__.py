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


"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "Passengers_refer_data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2