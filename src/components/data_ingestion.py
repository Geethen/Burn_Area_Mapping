import sys
import os
from exception import customException
from logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class dataIngestionConfig:
    train_data_path = os.path.join('artifacts',"train.csv")
    test_data_path = os.path.join('artifacts',"test.csv")
    calibration_data_path = os.path.join('artifacts',"calibration.csv")
    raw_data_path = os.path.join('artifacts',"raw.csv")

class dataIngestion:
    def __init__(self):
        self.ingestion_config = dataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")

        try:
            # Get a list of all CSV files in the directory
            csv_files = [file for file in os.listdir(os.getcwd()) if file.startswith('extract_') and file.endswith('.csv')]

            # Initialize an empty DataFrame to store the merged data
            df = pd.DataFrame()

            # Iterate over each CSV file and merge its data into the merged DataFrame
            for file in csv_files:
                file_path = os.path.join(os.getcwd(), file)
                dfi = pd.read_csv(file_path)
                df = pd.concat([df, dfi], ignore_index=True)
                
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            logging.info("Train-test split initiated")
            train_set,test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set, calibration_set = train_test_split(train_set, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            calibration_set.to_csv(self.ingestion_config.calibration_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Data ingestion is completed")

            return (self.ingestion_config.train_data_path,
                     self.ingestion_config.calibration_data_path,
                       self.ingestion_config.test_data_path)

        except Exception as e:
            raise customException(e,sys)
    