import ee
try:
    ee.Initialize()
except:
    ee.Authenticate()
    ee.Initialize()

from logger import logging
from geeml.extract import extractor

def extractFire(collection, region, scale):
    """For each pixel in the fire image collection (burn areas) extract coordinates,
      date and fire risk.
    
    Args:
        collection (ee.ImageCollection):
        region ()
      
      """
    
    coords = ee.Image.pixelCoordinates('EPSG:4326')
    # Risk of fires
    fireRisk = ee.FeatureCollection("projects/ee-geethensingh/assets/postdoc/VeldFire_Risk_2010")
    fireRisk = fireRisk.reduceToImage(properties = ['COUNT'], reducer = ee.Reducer.first()).rename('fireRisk')

    # Initialise extractor
    size = collection.size()
    fire = collection.toList(collection.size())
    for i in range(0, size.getInfo()):
        inData = ee.Image(ee.ImageCollection(fire.slice(i, i+1)).first())
        inDate = inData.get('system:index').getInfo()
        fireExt = extractor(covariates = inData.addBands([fireRisk, coords]), target = inData,
                            aoi = region, scale = scale, dd= 'data', spcvGridSize= 30000)

        # Extract data in batches of 30 000 points
        fireExt.extractPoints(gridSize = 50000, batchSize = 30000, filename = f'fire_{inDate}.csv')