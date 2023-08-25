import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split

import os,sys
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path   = os.path.join('artifacts','raw.csv')
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path  = os.path.join('artifacts','test.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Initiate Data Ingestion Process.")
            dataset = pd.read_csv(os.path.join('/config/workspace/Cement_Strength_Prediction/notebooks/data','cement_data_preprocessed.csv'))
            logging.info("Dataset read as pandas DataFrame.")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            dataset.to_csv(self.ingestion_config.raw_data_path,index=False)

            logging.info("Train-Test Split of Dataset.")
            train_data_set, test_data_set = train_test_split(dataset,test_size=0.20,random_state=42)

            train_data_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_data_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data Ingestion Completed Successfully.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            logging.info("Error occured in Data Ingesiton Process.")
            raise CustomException(e, sys)






