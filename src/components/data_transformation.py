import sys
import os
import ee

from src.exception import customException
from src.utils import save_object

try:
    service_account = 'bam-981@ee-geethensingh.iam.gserviceaccount.com'
    credentials = ee.ServiceAccountCredentials(service_account, 'secret.json')
    ee.Initialize(credentials)
except Exception as e:
    ee.Authenticate()
    ee.Initialize()
    customException = customException(e, sys)

from src.logger import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

@dataclass
class dataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class dataTransformation:
    def __init__(self):
        self.data_transformation_config = dataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="median")),
                    ('scaler', StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="most_frequent")),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
                )
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            preprocessor = ColumnTransformer([
            ('num_pipeline', num_pipeline, numerical_columns),
            ('cat_pipeline', cat_pipeline, categorical_columns)
            ])
            
            return preprocessor       

        except Exception as e:
            raise customException(e,sys)

    def initiate_data_transformation(self,train_path, calibration_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            calibration_df = pd.read_csv(calibration_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train, calibration and test data completed")

            logging.info("Obtaining preprocessing object")

            # preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "label"
            dropCols = "scenes"

            input_feature_train_df = train_df.drop(columns=[target_column_name, dropCols],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_cal_df = calibration_df.drop(columns=[target_column_name, dropCols],axis=1)
            target_feature_cal_df = calibration_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name, dropCols],axis=1)
            target_feature_test_df = test_df[target_column_name]

            # logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            # input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            # input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_df.values, np.array(target_feature_train_df)]
            calibration_arr = np.c_[input_feature_cal_df.values, np.array(target_feature_cal_df)]
            test_arr = np.c_[input_feature_test_df.values, np.array(target_feature_test_df)]

            # logging.info("Saved preprocessing object.")
            # save_object(file_path=self.data_transformation_config.preprocessed_object_file_path, obj = preprocessing_obj)

            return (train_arr, calibration_arr, test_arr)

        except Exception as e:
            raise customException(e, sys)