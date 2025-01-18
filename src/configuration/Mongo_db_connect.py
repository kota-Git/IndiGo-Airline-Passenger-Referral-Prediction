# code in configuration mongo_db_connection.py

import sys
import os
import pymongo
import certifi
from src.exception import referal_pred_Exception
from src.logger import logging
from src.constants import DATABASE_NAME, MONGODB_URL_KEY

ca = certifi.where()

class MongoDBClient:
    """
    Class Name :   export_data_into_feature_store
    Description :   This method exports the dataframe from mongodb feature store as dataframe

    Output      :   connection to mongodb database
    On Failure  :   raises an exception
    """
    client = None

    def __init__(self, database_name=DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)
                if mongo_db_url is None:
                    raise Exception(f"Environment key: {MONGODB_URL_KEY} not set.")
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url,tlsCAFile=ca)
            self.client= MongoDBClient.client
            self.database = self.client[database_name]  # incase error
            self.database_name = database_name
            logging.info("Connected to MongoDB database successfull")
        except Exception as e:
            raise referal_pred_Exception(e,sys)