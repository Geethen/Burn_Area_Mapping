import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import mode
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from mapie.classification import MapieClassifier
from mapie.metrics import classification_coverage_score
from mapie.metrics import classification_mean_width_score

from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, f1_score, accuracy_score
from dvclive import Live

from src.utils import save_object, evaluate_models
from src.logger import logging
from src.exception import customException

@dataclass
class modelTrainerConfig:
    trained_model_file_path = os.path.join(Path.cwd().parent,'components/artifacts', "model.pkl")

class modelTrainer:
    def __init__(self):
        self.model_trainer_config = modelTrainerConfig()

    def initiate_model_trainer(self, train_array, calibration_array, test_array):
        try:
            logging.info("Splitting train and test data input data")
            X_train, y_train, X_cal, y_cal, X_test, y_test = train_array[:, :-1], train_array[:, -1],\
            calibration_array[:, :-1], calibration_array[:, -1], \
            test_array[:, :-1], test_array[:, -1]

            # Get naive accuracy based on mode prediction
            # Predict the mode class for all instances
            mode_class = mode(y_train).mode
            baseline_predictions = [mode_class] * len(y_test)

            # Calculate accuracy of the baseline
            accuracy = accuracy_score(y_test, baseline_predictions)

            print(f"Naive Baseline Accuracy: {accuracy:.2f}")

            models = {
                "Random Forest": RandomForestClassifier(),
                "CatBoost": CatBoostClassifier(verbose = False)
            }

            params={
                "Random Forest":{
                    # 'criterion':['log_loss', 'gini', 'entropy'],
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoost":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                }
            }

            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, param=params)

            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            mapie_score = MapieClassifier(estimator = best_model, cv="prefit", method="lac", random_state= 42)
            mapie_score.fit(X_cal, y_cal)

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = mapie_score
            )

            predicted, pred_sets = mapie_score.predict(X_test, alpha = 0.05)
            pred_sets = np.squeeze(pred_sets)

            if best_model_score < 0.6:
                print('No best model found, metric < 0.6')
            logging.info("Best found model on both training and testing set")

            # dvc register model for tracking, and log metrics and confusion metric
            with Live() as live:
                live.log_metric("test/f1", f1_score(y_test, predicted, average="weighted"), plot=False)
                live.log_metric("test/mcc", matthews_corrcoef(y_test, predicted), plot=False)
                live.log_sklearn_plot("confusion_matrix", y_test, predicted, name="test/confusion_matrix", title="Test Confusion Matrix")
                live.log_metric("test/empirical coverage", classification_coverage_score(y_test.astype('int'), pred_sets), plot = False)
                live.log_metric("test/average set size", classification_mean_width_score(pred_sets), plot = False)

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