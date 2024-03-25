import ee
import os
from inference import Inference

service_account = 'bam-981@ee-geethensingh.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, 'secret.json')
ee.Initialize(credentials)

inference_pipeline = Inference()
sceneList = inference_pipeline.initiate_inference_pipeline('LANDSAT_8', 'South Africa')
sceneList