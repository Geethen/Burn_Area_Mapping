import ee
import sys

try:
    ee.Initialize()
except:
    ee.Authenticate()

from src.logger import logging
from src.exception import customException

def cloudMask(sensor: str, image : ee.Image)-> ee.Image:
    """
    This function performs scaling to reflectance and cloud masking on Landsat 4, 5, 7, 8 and 9 and Sentinel-2.
    Landsat uses the QA bands and Sentinel-2 uses the CloudScore+ dataset.

    Args:
        sensor (str): Specifies a Lansat mission. Either 'L8' and 'L9'.
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
        # clouds = image.select('QA_PIXEL').bitwiseAnd(math.pow(2, 10)).eq(0)
        # cirrus = image.select('QA_PIXEL').bitwiseAnd(math.pow(2, 11)).eq(0)
        qaMask = image.select('QA_PIXEL').bitwiseAnd(parseInt('11111', 2)).eq(0)
        saturationMask = image.select('QA_RADSAT').eq(0)

        #   Apply the scaling factors to the appropriate bands.
        opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0)

        #  Replace the original bands with the scaled ones and apply the masks.
        return image.addBands(opticalBands, None, True)\
            .addBands(thermalBands, None, True)\
            .updateMask(qaMask)\
            .updateMask(saturationMask)
            # .updateMask(clouds)\
            # .updateMask(cirrus)\
            
    # Landsat 4,5,7 surface reflectance
    if sensor in ['LANDSAT_4', 'LANDSAT_5', 'LANDSAT_7']:
        # Bit 0 - Fill
        # Bit 1 - Dilated Cloud
        # Bit 2 - Unused
        # Bit 3 - Cloud
        # Bit 4 - Cloud Shadow
        # clouds = image.select('QA_PIXEL').bitwiseAnd(math.pow(2, 10)).eq(0)
        # cirrus = image.select('QA_PIXEL').bitwiseAnd(math.pow(2, 11)).eq(0)
        qaMask = image.select('QA_PIXEL').bitwiseAnd(parseInt('11111', 2)).eq(0)
        saturationMask = image.select('QA_RADSAT').eq(0)

        # Apply the scaling factors to the appropriate bands.
        opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        thermalBand = image.select('ST_B6').multiply(0.00341802).add(149.0)

        # Replace the original bands with the scaled ones and apply the masks.
        return image.addBands(opticalBands, None, True)\
            .addBands(thermalBand, None, True)\
            .updateMask(qaMask)\
            .updateMask(saturationMask)
            # .updateMask(clouds)\
            # .updateMask(cirrus)\
            
    
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
        # scale and mask out clouds
        return image.divide(10000).updateMask(cloudMask)


# Prepare Sentinel or Landsat
def preProcessXCollection(collection: ee.ImageCollection, region: ee.Geometry, startDate : str, endDate: str)-> ee.ImageCollection:
    """
    This function performs spatio-temporal filtering, cloud masking, and then creates a gap-filled composite
    for Landsat and Sentinel-2.
    
    """
    # Get satellite name
    sensor = collection.first().get('SPACECRAFT_ID').getInfo()
    if sensor is None:
        sensor = 'Sentinel-2'
    # Spatiio-temporal filtering
    logging.info("Preparing X image collection (filtering, scaling, cloud masking)")
    filteredCollection = collection.filterBounds(region)\
                        .filterDate(startDate, endDate)\
                        .map(lambda image: cloudMask(sensor = sensor, image = image))
    logging.info(f"Prepared {filteredCollection.size().getInfo()} X images")
    return filteredCollection

def confidenceMask(image: ee.Image)-> ee.Image:
    # Define a function to remove fires with confidence level < 50%
    conf = image.select('ConfidenceLevel')
    level = conf.gt(50)
    yearBand = ee.Image(ee.Image(image).date().get('year')).toInt().rename('BurnYear')
    return image.updateMask(level).select('BurnDate').addBands(yearBand)

def preProcessyCollection(collection: ee.ImageCollection, region: ee.Geometry, startDate: str, endDate: str):
    logging.info("Preparing y image collection (confidence masking, projection, add date Band)")
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

