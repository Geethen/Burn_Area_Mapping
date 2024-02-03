import sys
import os
import ee
try:
    ee.Initialize()
except:
    ee.Authenticate()
    ee.Initialize()

from src.components.exception import customException
from utils import save_object

from logger import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Create fire events

class DataTransformation:
    def __init__(self, collection) -> None:
        self.collection = collection

    # def initiate_Xdata_transformation(self, X_data_path):

    def initiate_ydata_transformation(self, fire_data_path):
        try:
            df = pd.read_csv(fire_data_path)
            df = df.dropna()
            # Add a column for fire risk
            risk_dict = {7692: 'Extreme', 7805:'High', 2903:'Medium', 6555:'Low'}
            df['risk'] = df['fireRisk'].astype(int).map(risk_dict)

            # Combine 'year' and 'day_of_year' columns to create a new datetime column
            df['date'] = pd.to_datetime(df['BurnYear'].astype(str) + df['BurnDate'].astype(str), format='%Y%j')
        
        except Exception as e:
            raise customException(e, sys)



@dataclass
class dataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
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

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")
            save_object(file_path=self.data_transformation_config.preprocessed_object_file_path, obj = preprocessing_obj)

            return (train_arr,test_arr,self.data_transformation_config.preprocessed_object_file_path)

        except Exception as e:
            raise customException(e, sys)