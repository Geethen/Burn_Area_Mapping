"""
Contains various utility functions for PyTorch model training and saving.
"""
import os
import sys
import datetime

import torch
from pathlib import Path
import rasterio as rio
from torchgeo.datasets import RasterDataset
from typing import List
import dill
from sklearn.metrics import f1_score
from sklearn.model_selection import (GridSearchCV)

from exception import customException

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)

class MyNormalize(torch.nn.Module):
    def __init__(self, mean: List[float], stdev: List[float]):
        super().__init__()

        self.mean = torch.Tensor(mean)[:, None, None]
        self.std = torch.Tensor(stdev)[:, None, None]

    def forward(self, inputs: dict):

        x = inputs[..., : len(self.mean), :, :]

        # if batch
        if inputs.ndim == 4:
            x = (x - self.mean[None, ...]) / self.std[None, ...]

        else:
            x = (x - self.mean) / self.std

        inputs[..., : len(self.mean), :, :] = x

        return inputs
    def revert(self, inputs: dict):
        """
        De-normalize the batch.

        Args:
            inputs (dict): Dictionary with the 'image' key
        """

        x = inputs[..., : len(self.mean), :, :]

        # if batch
        if x.ndim == 4:
            x = inputs[:, : len(self.mean), ...]
            x = x * self.std[None, ...] + self.mean[None, ...]
        else:
            x = x * self.std + self.mean

        inputs[..., : len(self.mean), :, :] = x

        return inputs
def calc_statistics(dset: RasterDataset):
        """
        Calculate the statistics (mean and std) for the entire dataset
        Warning: This is an approximation. The correct value should take into account the
        mean for the whole dataset for computing individual stds.
        For correctness I suggest checking: http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
        """

        # To avoid loading the entire dataset in memory, we will loop through each img
        # The filenames will be retrieved from the dataset's rtree index
        files = [item.object for item in dset.index.intersection(dset.index.bounds, objects=True)]

        # Reseting statistics
        accum_mean = 0
        accum_std = 0

        for file in files:
            img = rio.open(file).read()/65535 #type: ignore
            accum_mean += img.reshape((img.shape[0], -1)).mean(axis=1)
            accum_std += img.reshape((img.shape[0], -1)).std(axis=1)

        # at the end, we shall have 2 vectors with lenght n=chnls
        # we will average them considering the number of images
        return accum_mean / len(files), accum_std / len(files)

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