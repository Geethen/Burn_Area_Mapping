import os
import sys
import datetime

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.model_selection import (GridSearchCV)
from mapie.classification import MapieClassifier

from exception import customException

def delete_old_logs(repo_path, specified_date=None):
    # If specified_date is not provided, keep logs for the past month
    if specified_date is None:
        specified_date = datetime.datetime.now() - datetime.timedelta(days=30)
    
    # Iterate over all directories and files in the repository recursively
    for root, dirs, files in os.walk(repo_path):
        for file_name in files:
            # Check if the file name matches the log file format
            if file_name.endswith('.log'):
                try:
                    # Extract the date from the file name
                    file_date_str = file_name.split('.')[0]
                    file_date = datetime.datetime.strptime(file_date_str, '%m_%d_%Y_%H_%M_%S')
                    
                    # Check if the file date is before the specified date
                    if file_date < specified_date:
                        # If yes, delete the file
                        os.remove(os.path.join(root, file_name))
                        print(f"Deleted file: {os.path.join(root, file_name)}")
                except ValueError:
                    # Handle invalid file names or dates
                    print(f"Ignored invalid file name: {os.path.join(root, file_name)}")

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise customException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise customException(e, sys)
        
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):

            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_, random_state=42)
            model.fit(X_train,y_train)

            y_test_pred = model.predict(X_test)

            test_model_score = f1_score(y_test, y_test_pred, average= 'weighted')

            report[list(models.keys())[i]] = test_model_score
        return report
    
    except Exception as e:
            raise customException(e, sys)