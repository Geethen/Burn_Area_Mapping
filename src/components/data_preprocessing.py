import ee
import sys
import math

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

def cloudMask(sensor: str, image : ee.Image)-> ee.Image:
    """
    This function performs scaling to reflectance and cloud masking on Landsat 4, 5, 7, 8 and 9 and Sentinel-2.
    Landsat uses the QA bands and Sentinel-2 uses the CloudScore+ dataset.

    Args:
        sensor (str): Specifies a Lansat mission. One of [].
        image (ee.Image): A landat image collection

    Returns:
        image (ee.Image) with masked clouds
    """
    if sensor in ['LANDSAT_8', 'LANDSAT_9']:
        # Landsat 8 and 9 SR
        #  QA_PIXEL band (CFMask) to mask unwanted pixels.

        #   Bit 0 - Fill
        #   Bit 1 - Dilated Cloud
        #   Bit 2 - Cirrus
        #   Bit 3 - Cloud
        #   Bit 4 - Cloud Shadow
        clouds = image.select('QA_PIXEL').bitwiseAnd(0b1000).eq(0)
        cirrus = image.select('QA_PIXEL').bitwiseAnd(0b1100).eq(0)
        saturationMask = image.select('QA_RADSAT').eq(0)

        #   Apply the scaling factors to the appropriate bands.
        opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0)

        # Compute normalised burn ratio (NBR)
        nbr = image.normalizedDifference(['SR_B5', 'SR_B7']).multiply(-1).rename('nbr')

        #  Replace the original bands with the scaled ones and apply the masks.
        return image.addBands([opticalBands, thermalBands, nbr], None, True)\
            .updateMask(clouds)\
            .updateMask(cirrus)\
            .updateMask(saturationMask)
            
            
    # Landsat 4,5,7 surface reflectance
    if sensor in ['LANDSAT_4', 'LANDSAT_5', 'LANDSAT_7']:
        # Bit 0 - Fill
        # Bit 1 - Dilated Cloud
        # Bit 2 - Unused
        # Bit 3 - Cloud
        # Bit 4 - Cloud Shadow
        clouds = image.select('QA_PIXEL').bitwiseAnd(math.pow(2, 10)).eq(0)
        cirrus = image.select('QA_PIXEL').bitwiseAnd(math.pow(2, 11)).eq(0)
        saturationMask = image.select('QA_RADSAT').eq(0)

        # Apply the scaling factors to the appropriate bands.
        opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        thermalBand = image.select('ST_B6').multiply(0.00341802).add(149.0)

        # Compute normalised burn ratio (NBR)
        nbr = image.normalizedDifference(['SR_B4', 'SR_B7']).multiply(-1).rename('nbr')

        # Replace the original bands with the scaled ones and apply the masks.
        return image.addBands(opticalBands, None, True)\
            .addBands(thermalBand, None, True)\
            .updateMask(clouds)\
            .updateMask(cirrus)\
            .updateMask(saturationMask)
            
    
    # Sentinel-2
    if sensor == 'Sentinel-2':
        #  Cloud Score+ image collection. Note Cloud Score+ is produced from Sentinel-2
        # Level 1C data and can be applied to either L1C or L2A collections.
        csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')

        #  Use 'cs' or 'cs_cdf', depending on your use-case; see docs for guidance.
        QA_BAND = 'cs_cdf'

        # The threshold for masking; values between 0.50 and 0.65 generally work well.
        # Higher values will remove thin clouds, haze & cirrus shadows.
        CLEAR_THRESHOLD = 0.60

        # link Sentinel-2 image with corresponding Cloud Score+
        cloudMask = image.linkCollection(csPlus, [QA_BAND]).select(QA_BAND).gte(CLEAR_THRESHOLD)

        # Compute normalised burn ratio (NBR)
        nbr = image.normalizedDifference(['B12', 'B8']).rename('nbr')

        # scale and mask out clouds
        return image.divide(10000).addBands(nbr).updateMask(cloudMask)
   
# Prepare Sentinel or Landsat
def preProcessXCollection(image: ee.Image, nImages: int, returnInterval: int)-> ee.ImageCollection:
    """
    This function performs spatio-temporal filtering, cloud masking for Landsat and Sentinel-2.
    
    """
    # Get satellite name
    sensor = image.get('SPACECRAFT_ID').getInfo()
    if sensor is None:
        sensor = 'Sentinel-2'
    # Spatiio-temporal filtering
    logging.info("Preparing X image collection (filtering, scaling, cloud masking)")

    startDate = image.date().advance(-nImages*returnInterval*3, 'day')
    endDate = image.date()
    filteredImages = supportedSensors.get(sensor).filterBounds(image.geometry()).filterDate(startDate,endDate)\
        .filter(ee.Filter.eq('WRS_PATH',image.get('WRS_PATH')))\
        .filter(ee.Filter.eq('WRS_ROW',image.get('WRS_ROW')))
    refDate = image.date()
    result = filteredImages.map(lambda image: image.set('dayDist', refDate.difference(image.date(), 'day'))
                                ).sort('dayDist').limit(nImages).merge([image])\
                                .map(lambda image: cloudMask(sensor = sensor, image = image))     
    logging.info(f"Prepared {nImages+1} X images")
    return result

def confidenceMask(image: ee.Image)-> ee.Image:
    # Define a function to remove fires with confidence level < 50%
    conf = image.select('ConfidenceLevel')
    level = conf.gt(50)
    # yearBand = ee.Image(ee.Image(image).date().get('year')).toInt().rename('BurnYear')
    return image.set('BurnYear', ee.Image(image).date().get('year').toInt()).updateMask(level).select('BurnDate')#.addBands(yearBand)

def preProcessyCollection(collection: ee.ImageCollection, region: ee.Geometry, startDate: str, endDate: str):
    logging.info("Preparing y image collection (confidence masking, projection, select date Band)")
    # Filter and map the fire collection
    fire = collection \
        .filterBounds(region) \
        .filterDate(startDate, endDate) \
        .map(confidenceMask)

    # Set projection and scale
    projection = fire.select('BurnDate').first().projection()

    fire = fire.map(lambda image: image.setDefaultProjection(**{'crs':projection, 'scale': 250}))
    logging.info(f"Prepared {fire.size().getInfo()} y images")
    return fire

