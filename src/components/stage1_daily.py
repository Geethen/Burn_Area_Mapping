import ee
import os
from src.components.inference import Inference

service_account = 'bam-981@ee-geethensingh.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, 'secret.json')
ee.Initialize(credentials)

# Get the current working directory
current_directory = os.getcwd()
print(current_directory)

# Define the relative path to the target directory
relative_path = "src/components"

# Join the current directory with the relative path to get the absolute path
target_directory = os.path.join(current_directory, relative_path)

# Change to the target directory
os.chdir(target_directory)
inference_pipeline = Inference()
sceneList = inference_pipeline.initiate_inference_pipeline('LANDSAT_8', 'South Africa', '2016-01-01', '2017-01-01')
sceneList