import os
import ee
import sys
import numpy as np
import mapie

from datetime import datetime
from dataclasses import dataclass
import geedim as gd

from data_extraction import extractInferenceDataset
from utils import save_object, load_object
from logger import logging
from exception import customException

try:
    service_account = 'bam-981@ee-geethensingh.iam.gserviceaccount.com'
    credentials = ee.ServiceAccountCredentials(service_account, 'secret.json')
    ee.Initialize(credentials)
except Exception as e:
    ee.Authenticate()
    ee.Initialize()
    customException = customException(e, sys)


supportedSensors = {'Sentinel-2': ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED"),
                    'LANDSAT_4': ee.ImageCollection("LANDSAT/LT04/C02/T1_L2"),
                    'LANDSAT_5': ee.ImageCollection("LANDSAT/LT05/C02/T1_L2"),
                    'LANDSAT_7': ee.ImageCollection("LANDSAT/LE07/C02/T1_L2"),
                    'LANDSAT_8': ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"),
                    'LANDSAT_9': ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")}

@dataclass
class modelInferenceConfig:
    model_path = os.path.join('artifacts',"model.pkl")
    dateChecked_path = os.path.join('artifacts',"lastCheckedDate.pkl")
    downloadList_path = os.path.join('artifacts',"downloadList.pkl")

class Inference:
    def __init__(self):
        self.model_config = modelInferenceConfig()

    def collectionUpdates(self, imageCollection: ee.ImageCollection, country: str)->bool:
        """
        checks if new scenes have been made available

        Args:
            imageCollection (ee.ImageCollection):
            country (str): Name of the country to check for new scenes

        Returns:
            Boolean: True if new scenes have been made available, False otherwise.
        
        """
        logging.info("Checking for new scenes")
        countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
        self.country = countries.filter(ee.Filter.eq('country_na', country))
        # Get current date
        current_date = datetime.now().date()
        # Format the current date as "YYYY-MM-DD"
        current_formattedDate = ee.Date(current_date.strftime("%Y-%m-%d"))
        try:
            last_checked = load_object(self.model_config.dateChecked_path)
        except Exception:
            last_checked = "2024-03-01"
            save_object(self.model_config.dateChecked_path, last_checked)
        nScenes = imageCollection.filterBounds(self.country.geometry()).filterDate(ee.Date(last_checked), current_formattedDate).size().getInfo()
        print("Number of new scenes:", nScenes)
        return nScenes>0

    def getPredictions(self, df):
        model = load_object(self.model_config.model_path)
        dfPredict = df.drop(['scenes'], axis=1)
        _, pred_sets = model.predict(dfPredict, alpha = 0.05)
        def create_mask(arr):
            mask = ~np.all(arr == [True, False], axis=1)
            return mask
        mask = create_mask(np.squeeze(pred_sets))
        if mask.sum() > 1:
            # Select rows where the mask is True
            sceneIds = df[mask]['scenes'].apply(lambda x: x[-1])
            return sceneIds
        else:
            print("There are no new scenes with fire to proceed.")
        

    def initiate_inference_pipeline(self, sensor: str, country: str, startDate:str = None, endDate:str = None)->list:
        """
        Conducts inference pipeline for fire detection.

        Args:
            sensor (str): Type of sensor used for data collection.
            country (str): Country for which the inference is performed.
            startDate (str): A date str formatted as 'YYYY-MM-DD'. If specified, overrides last checked date
            endDate (str): A date str formatted as 'YYYY-MM-DD'. If specified, overrides last current date

        Returns:
            list: List of scenes with fires.
        """
        logging.info("Entering inference pipeline")
        try:
            if self.collectionUpdates(supportedSensors.get(sensor), country= country):
                # extract covariates and prepare data for inference.
                if endDate is None:
                    # Get current date
                    current_date = datetime.now().date()
                    # Format the current date as "YYYY-MM-DD"
                    current_formattedDate = ee.Date(current_date.strftime("%Y-%m-%d"))
                else:
                    current_date = endDate
                    # Format the end date as "YYYY-MM-DD"
                    current_formattedDate = ee.Date(endDate)

                if startDate is None:
                    last_checked = load_object(self.model_config.dateChecked_path)
                else:
                    last_checked = ee.Date(startDate)
                print("input dates", last_checked, current_date)
                df = extractInferenceDataset(sensor, country, ee.Date(last_checked), current_formattedDate)
                last_checked = datetime.now().date()
                # make prediction
                downloadList = self.getPredictions(df)
                # overwrite last_checked with current date
                save_object(self.model_config.dateChecked_path, current_formattedDate)
            else:
                downloadList = []
                print("There are no new scenes")
            # Save list of scenes to be downloaded to disk
            save_object(self.model_config.downloadList_path, downloadList)
            
            # logging.info(f"There are {len(downloadList)} new scenes")
            return downloadList
        except Exception as e:
            raise customException(e, sys)