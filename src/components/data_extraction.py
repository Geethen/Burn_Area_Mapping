import pandas as pd
import ee
import sys
from tqdm.auto import tqdm

from logger import logging
from src.exception import customException
from geeml.extract import extractor
from data_preprocessing import preProcessXCollection

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

def getStats(imageCollection: ee.ImageCollection)->pd.DataFrame:
    """
    computes the (spatial) percentiles (5, 25, 50, 75, 95) for the temporal variance across the temporal percentiles

    Args:
        imageCollection (ee.ImageCollection): A imagecollection that contains a Normalised Burn-Ratio band called 'nbr'

    Returns:
        pd.DataFrame containing the 5 percentiles and a column ('scenes') containing a list of scene id's used for the computation
    """
    scenes = [imageCollection.aggregate_array('system:index').getInfo()]
    # Compute NBR temporal percentiles
    reducer = ee.Reducer.percentile([5, 25, 50, 75, 95])
    percentiles = imageCollection.select('nbr').reduce(reducer)
    # Compute temporal variance
    variance = percentiles.reduce(ee.Reducer.variance()).rename('temporal_variance')
    # Compute image spatial stats
    stats = variance.reduceRegion(reducer = reducer,
                                            geometry = imageCollection.geometry(),
                                            scale = 1000)
    row = pd.DataFrame([stats.getInfo()])
    row['scenes'] = scenes
    return row

def getImages(image: ee.Image, featureCollection: ee.FeatureCollection)-> ee.ImageCollection:
    """ 
    1) Checks if image is spatio-temporally within a fire event, labels image accordingly and
    2) Gets all prior images associated with creating a feature.
    """
    # get all firevents that overlap the scene and that started at most two months prior to currentn image 
    # We use 2 months prior -nMonths largely irrelevant because we check if image falls with a dateRange.
    # Only consideration is for compute
    fireEvents = featureCollection.filterBounds(image.geometry()).filterDate(image.date().advance(-2, 'month'), image.date().advance(1, 'day'))
    startDate = fireEvents.aggregate_min('system:time_start')
    endDate = fireEvents.aggregate_max('system:time_end')
    # If there are no fire events from two months prior to image, start and end date will be assigned None, overwrite this with
    # current image date. This will return False for the dateRange function.
    if startDate.getInfo() is None or endDate.getInfo() is None:
        startDate = image.date()
        endDate = image.date()

    dateRange = ee.DateRange(startDate, endDate)

    # check if it falls within dateRange of a fire event- set label property
    yImage = ee.Algorithms.If(dateRange.contains(image.date()), image.set('label', 1).copyProperties(image), image.set('label', 0).copyProperties(image))

    # Define datasets, get xweeks nImages prior to image (i.e., get time series info)
    xImages =  preProcessXCollection(image = image, nImages = 4, returnInterval = 16)
    return xImages, yImage

def extractDataset(sensor, country, startDate, endDate, fireEvents, Xweeks: int, filename)-> pd.DataFrame:
    """Extracts the dataset from the given image collection
    
    Args:
        sensor (str): 
        country (ee.Geometry): 
        startDate (ee.date): 
        endDate (ee.Date): 

    Returns:
        pd.DataFrame
    
    """
    logging.info("Extracting x and y data (extractDataset)")
    countryGeom = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(ee.Filter.eq('country_na', country)).geometry()
    # filterImages to country of interest
    filteredImages = supportedSensors.get(sensor).filterBounds(countryGeom).filterDate(startDate, endDate)
    
    xSize = filteredImages.size().getInfo()
    xListImages = filteredImages.toList(xSize)

    outdf = pd.DataFrame()
    for idx in tqdm(range(xSize)):
        xImage = ee.Image(xListImages.get(idx))
        # get Ximages and labelled y image
        xImages, yImage = getImages(xImage, fireEvents)
        # convert to geodataframe
        try:
            row = getStats(xImages)
            row['label'] = ee.Image(yImage).get('label').getInfo()
            outdf = pd.concat([outdf, row])
            # Write the results to a file.
            with open(filename, 'w', newline='') as file:
                # Use the to_csv method to write the DataFrame to the CSV file
                outdf.to_csv(file, index=False)
        except Exception as e:
            raise customException(e, sys)
        
def getInferenceImages(image: ee.Image)-> ee.ImageCollection:
    """ 
    1) Gets all prior images associated with creating a feature.
    """

    # Define datasets, get nImages prior to image (i.e., get time series info)
    xImages =  preProcessXCollection(image = image, nImages = 4, returnInterval = 16)
    return xImages
        
def extractInferenceDataset(sensor, country, startDate, endDate)-> pd.DataFrame:
    """Extracts the dataset from the given image collection
    
    Args:
        sensor (str): 
        country (ee.Geometry): 
        startDate (ee.Date): 
        endDate (ee.Date): 

    Returns:
        pd.DataFrame
    
    """
    logging.info("Extracting x inference data (extractInferenceDataset)")
    countryGeom = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(ee.Filter.eq('country_na', country)).geometry()
    # filterImages to country of interest
    filteredImages = supportedSensors.get(sensor).filterBounds(countryGeom).filterDate(startDate, endDate)
    
    xSize = filteredImages.size().getInfo()
    xListImages = filteredImages.toList(xSize)

    outdf = pd.DataFrame()
    for idx in tqdm(range(xSize)):
        xImage = ee.Image(xListImages.get(idx))
        # get Ximages and current image
        xImages = getInferenceImages(xImage)
        # convert to geodataframe
        try:
            row = getStats(xImages)
            outdf = pd.concat([outdf, row])
        except Exception as e:
            raise customException(e, sys)
    return outdf

def extractXy (yCollection: ee.FeatureCollection, Xweeks: int, filename):
    """Extracts x and y data from the given collections"""
    logging.info("Extracting x and y data")

    ySize = yCollection.size().getInfo()
    fire = yCollection.toList(ySize)

    outdf = pd.DataFrame()
    for idx in tqdm(range(ySize)):
        fireEvent = ee.Feature(fire.get(idx))
        endDate = ee.Date(fireEvent.get('ig_date'))
        startDate = endDate.advance(ee.Number(Xweeks*-7),'day')

        # Define datasets
        L8filtered =  preProcessXCollection(collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2'), region = fireEvent.geometry(), startDate = startDate, endDate = endDate)
        # L9filtered = preProcessXCollection(collection = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'), region = fireEvent, startDate = startDate, endDate = endDate)
        # S2filtered = preProcessXCollection(collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2'), region = fireEvent, startDate = startDate, endDate = endDate)
        
        # convert to geodataframe
        try:
            row = getStats(L8filtered)
            outdf = pd.concat([outdf, row])
            # Write the results to a file.
            with open(filename, 'w', newline='') as file:
                # Use the to_csv method to write the DataFrame to the CSV file
                outdf.to_csv(file, index=False)
        except Exception as e:
            raise customException(e, sys)
            # print(f"Error occurred for index {idx}")
            # continue

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