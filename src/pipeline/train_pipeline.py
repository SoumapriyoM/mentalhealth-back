import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    try:
        logging.info("Starting the training pipeline")

        data_ingestion = DataIngestion()
        train_data, test_data = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr, test_arr = data_transformation.initiate_data_transformation(train_data, test_data)

        model_trainer = ModelTrainer()
        model_trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info("Training pipeline completed successfully")
    except Exception as e:
        logging.error(f"Error in training pipeline: {e}")
        raise CustomException(e, sys)
