import ee
try:
    ee.Initialize()
except:
    ee.Authenticate()
    ee.Initialize()

from src.logger import logging
from geeml.extract import extractor
from src.data_extraction import preProcessXCollection

def extractXy (Xcollection: ee.ImageCollection, yCollection: ee.FeatureCollection, Xweeks: int):
    """Extracts x and y data from the given collections"""
    logging.info("Extracting x and y data")

    ySize = yCollection.size().getInfo()
    fire = yCollection.toList(ySize)
    for idx in range(ySize):
        fireEvent = ee.Feature(fire.get(idx))
        startDate = fireEvent.get('BurnDate').first().getInfo()
        endDate = fireEvent.get('BurnDate').last().getInfo()

        # Define datasets
        L8filtered =  preProcessXCollection(collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2'), region = fireEvent, startDate = startDate, endDate = endDate)
        # L9filtered = preProcessXCollection(collection = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'), region = fireEvent, startDate = startDate, endDate = endDate)
        # S2filtered = preProcessXCollection(collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2'), region = fireEvent, startDate = startDate, endDate = endDate)

        # Compute NBR temporal percentiles
        reducer = ee.Reducer.percentile([5, 25, 50, 75, 95])
        L8percentiles = L8filtered.reduce(reducer)
        # S2percentiles = S2filtered.reduce(reducer)
        # Compute temporal variance
        L8variance = L8percentiles.reduce(ee.Reducer.variance()).rename('temporal_variance')
        # S2variance = S2percentiles.reduce(ee.Reducer.variance())

        # Compute image spatial stats
        L8stats = L8variance.reduceRegion(reducer = reducer,
                                              geometry = L8variance.geometry(),
                                              scale = 1000).get('temporal_variance')
        # S2stats = S2variance.reduceRegion(reducer = reducer,
        #                                       geometry = S2variance.geometry(),
        #                                       scale = 1000)
        # convert to geodataframe
        ee.data.computeFeatures(L8stats, L8stats.geometry())

        # Properties to extract
        # Area
        # startdate
        # enddate
        # duration
        

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