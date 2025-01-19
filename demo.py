# testing of code of data_ingestin using  demo.py
import sys
from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig
from src.logger import logging
from src.exception import referal_pred_Exception

def main():
    try:
        # Initialize DataIngestionConfig
        data_ingestion_config = DataIngestionConfig()

        # Initialize DataIngestion
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)

        # Start the data ingestion process
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        # Log the results
        logging.info(f"Data ingestion completed successfully.")
        logging.info(f"Training file path: {data_ingestion_artifact.trained_file_path}")
        logging.info(f"Testing file path: {data_ingestion_artifact.test_file_path}")

        # Print dataset shapes for confirmation
        import pandas as pd
        train_data = pd.read_csv(data_ingestion_artifact.trained_file_path)
        test_data = pd.read_csv(data_ingestion_artifact.test_file_path)
        print(f"Training dataset shape: {train_data.shape}")
        print(f"Testing dataset shape: {test_data.shape}")
    except Exception as e:
        logging.error(f"Error occurred during data ingestion: {e}")
        print(f"Error occurred during data ingestion: {e}")

if __name__ == "__main__":
    main()