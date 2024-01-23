import sys
import os
from exception import customException
from logger import logging
from components.data_transformation import DataTransformation
from components.model_trainer import ModelTrainer
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class dataIngestionConfig:
    train_data_path = os.path.join('artifacts',"train.csv")
    test_data_path = os.path.join('artifacts',"test.csv")
    raw_data_path = os.path.join('artifacts',"raw.csv")

class dataIngestion:
    def __init__(self):
        self.ingestion_config = dataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")

        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            logging.info("Train test split initiated")
            train_set,test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Data ingestion is completed")

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)

        except Exception as e:
            raise customException(e,sys)
        
if __name__ == "__main__":
    obj = dataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_arr, test_arr)