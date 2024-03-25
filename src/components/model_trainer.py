import os
import sys
from dataclasses import dataclass

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import mode
from catboost import CatBoostClassifier
from sklearn.ensemble import (AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, f1_score, accuracy_score
from dvclive import Live

from src.utils import save_object, evaluate_models
from src.logger import logging
from exception import customException

@dataclass
class modelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', "model.pkl")

class modelTrainer:
    def __init__(self):
        self.model_trainer_config = modelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test data input data")
            X_train, y_train, X_test, y_test = train_array[:, :-1], train_array[:, -1], test_array[:, :-1], test_array[:, -1]

            # Get naive accuracy based on mode prediction
            # Predict the mode class for all instances
            mode_class = mode(y_train).mode
            baseline_predictions = [mode_class] * len(y_test)

            # Calculate accuracy of the baseline
            accuracy = accuracy_score(y_test, baseline_predictions)

            print(f"Naive Baseline Accuracy: {accuracy:.2f}")

            models = {
                #  "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),                
                # "Gradient Boosting": HistGradientBoostingClassifier(),
                "CatBoost": CatBoostClassifier(verbose = False),
                # "XGBRegressor": XGBClassifier(),
                # "AdaBoost Regressor": AdaBoostClassifier()
            }

            params={
                # "Decision Tree": {
                #     'criterion':['log_loss', 'gini', 'entropy'],
                #     # 'splitter':['best','random'],
                #     # 'max_features':['sqrt','log2'],
                # },
                "Random Forest":{
                    # 'criterion':['log_loss', 'gini', 'entropy'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                # "Gradient Boosting":{
                #     # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                #     'learning_rate':[.1,.01,.05,.001],
                #     'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                #     # 'criterion':['squared_error', 'friedman_mse'],
                #     # 'max_features':['auto','sqrt','log2'],
                #     'n_estimators': [8,16,32,64,128,256]
                # },
                "CatBoost":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                # "AdaBoost":{
                #     'learning_rate':[.1,.01,0.5,.001],
                #     # 'loss':['linear','square','exponential'],
                #     'n_estimators': [8,16,32,64,128,256]
                # }
            }

            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, param=params)

            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                print('No best model found, metric < 0.6')
                # raise customException("No best model found", sys)
            logging.info("Best found model on both training and testing set")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            # dvc register model for tracking, and log metrics and confusion metric
            with Live() as live:
                # live.log_artifact(
                #     self.model_trainer_config.trained_model_file_path,
                #     type="model",
                #     name="BurnArea-classification",
                #     desc="This is a Scene-level classification model to discern satellite images with burn areas.",
                #     labels=["scene-level", "classification", "satellite-images"],
                # )
                live.log_metric("test/f1", f1_score(y_test, predicted, average="weighted"), plot=False)
                live.log_metric("test/mcc", matthews_corrcoef(y_test, predicted), plot=False)
                live.log_sklearn_plot(
                "confusion_matrix", y_test, predicted, name="test/confusion_matrix",
                title="Test Confusion Matrix")

            f1 = f1_score(y_test, predicted)
            print("classification report", classification_report(y_test, predicted))

            print('Confusion Matrix:')
            # Create a DataFrame for the confusion matrix
            confusion_df = pd.DataFrame(confusion_matrix(y_test, predicted), index=['Normal', 'Fire'], columns=['Pred normal', 'Pred fire'])
            # Plot the confusion matrix using Seaborn's heatmap
            plt.figure(figsize=(6, 4))
            sns.heatmap(confusion_df, annot=True, fmt='d', cmap='Blues', cbar=False)
            
            return f1
        except Exception as e:
            raise customException(e, sys)