import ee
import os
from inference import Inference

my_secret = os.environ.get('EE_SERVICE_ACCOUNT_KEY')

service_account = 'github-action@ee-geethensingh.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, my_secret)
ee.Initialize(credentials)

inference_pipeline = Inference()
sceneList = inference_pipeline.initiate_inference_pipeline('LANDSAT_8', 'South Africa')
sceneList