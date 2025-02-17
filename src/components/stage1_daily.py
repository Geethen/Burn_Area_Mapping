import sys
import ee
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.components.inference import Inference

service_account = 'bam-981@ee-geethensingh.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, 'secret.json')
ee.Initialize(credentials)

inference_pipeline = Inference()
# Western Cape - fynbos and renosterveld
ecoregions = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017")
aoi = ee.FeatureCollection(ecoregions.filter(ee.Filter.inList('ECO_ID', [89,90]))).bounds()
sceneList = inference_pipeline.initiate_inference_pipeline('LANDSAT_8', aoi, '2016-01-01', '2017-01-01')
sceneList