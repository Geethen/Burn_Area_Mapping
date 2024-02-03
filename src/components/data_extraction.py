import ee
try:
    ee.Initialize()
except:
    ee.Authenticate()
    ee.Initialize()

from src.logger import logging
from geeml.extract import extractor

def extractFire(collection: ee.ImageCollection, region: ee.Feature, scale:int) -> None:
    """For each pixel in the fire image collection (burn areas) extract coordinates,
      date and fire risk. Data is exported to the data folder in the format "fire_{date}.csv".
      The date corresponds to the first day of the month.
    
    Args:
        collection (ee.ImageCollection): Preprocessed imageCollections from the data_preprocessing stage
        region (ee.Feature): The extent of the area to extract data.
        scale (int): The scale at which the dat should be extracted.      
      """
    
    coords = ee.Image.pixelCoordinates('EPSG:4326')
    # Risk of fires
    fireRisk = ee.FeatureCollection("projects/ee-geethensingh/assets/postdoc/VeldFire_Risk_2010")
    fireRisk = fireRisk.reduceToImage(properties = ['COUNT'], reducer = ee.Reducer.first()).rename('fireRisk')

    # Initialise extractor
    size = collection.size()
    fire = collection.toList(size)
    logging.info('Extracting fire data...')
    for i in range(0, size.getInfo()):
        inData = ee.Image(ee.ImageCollection(fire.slice(i, i+1)).first())
        inDate = inData.get('system:index').getInfo()
        fireExt = extractor(covariates = inData.addBands([fireRisk, coords]), target = inData,
                            aoi = region, scale = scale, dd= r'src\notebooks\data', spcvGridSize= 30000)

        # Extract data in batches of 30 000 points
        fireExt.extractPoints(gridSize = 50000, batchSize = 30000, filename = f'fire_{inDate}.csv')