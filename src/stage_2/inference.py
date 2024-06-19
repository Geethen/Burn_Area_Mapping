import sys
import os
import math
from pathlib import Path
import rasterio as rio
import torch
from torchvision import transforms
import segmentation_models_pytorch as smp
from geedim.download import BaseImage
from torchgeo.transforms import indices, AugmentationSequential

# Parallel compute
from tqdm.auto import tqdm
import concurrent.futures
import threading
import logging

from src.exception import customException
from src.utils import load_object, MyNormalize
from dataclasses import dataclass

import ee
try:
    service_account = 'bam-981@ee-geethensingh.iam.gserviceaccount.com'
    credentials = ee.ServiceAccountCredentials(service_account, 'secret.json')
    ee.Initialize(credentials)
except Exception as e:
    ee.Authenticate()
    ee.Initialize()
    customException = customException(e, sys)

def inference(infile, imgTransforms, model, outfile, patchSize, num_workers=4, device:str = None):
    """
    Run inference using model on infile block-by-block and write to a new file (outfile). 
    In the case, that the infile image width/height is not exactly divisible by 32, padding
    is added for inference and removed prior to the outfile being saved.
    
    Args:
        infile (string): Path to input image/covariates
        model (pth file): Loaded trained model/checkpoint
        outfile (string): Path to save predicted image
        patchSize (int): Must be a multiple of 32. Size independent of model input size.
        num_workers (int): Num of workers to parralelise across
        
    Returns:
        A tif saved to the outfile destination
        
    """

    with rio.open(infile) as src:
        
        logger = logging.getLogger(__name__)

        # Create a destination dataset based on source params. The
        # destination will be tiled, and we'll process the tiles
        # concurrently.
        profile = src.profile
        profile.update(blockxsize= patchSize, blockysize= patchSize, tiled=True, count=1)

        with rio.open(Path(outfile), "w", **profile) as dst:
            windows = [window for ij, window in dst.block_windows()]

            # use a lock to protect the DatasetReader/Writer
            read_lock = threading.Lock()
            write_lock = threading.Lock()

            def process(window):
                with read_lock:
                    src_array = src.read(window=window)#nbands, nrows, ncols(4, h, w)
                    w, h = src_array.shape[1], src_array.shape[2]
                    image = imgTransforms({"image": torch.from_numpy(src_array)})['image']
                    image = image.to(device, dtype=torch.float)#(1, h, w, 4)
                    hpad = math.ceil(h/32)*32-h
                    wpad = math.ceil(w/32)*32-w
                    transform = transforms.Pad((0, 0, hpad, wpad))
                    # add padding to image
                    image = transform(image)
                    model.eval()
                    output = model(image)[:, 1, :, :].squeeze()#(1,1,h,w)
                    # remove padding
                    result = output[0:w, 0:h].detach().cpu()
                    # plt.imshow(result.numpy())

                with write_lock:
                    dst.write(result, 1, window=window)

            # We map the process() function over the list of
            # windows.
            with tqdm(total=len(windows), desc = os.path.basename(outfile)) as pbar:
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = {executor.submit(process, window): window for window in windows}
                    
                    try:
                        for future in concurrent.futures.as_completed(futures):
                            future.result()
                            pbar.update(1)
                                    
                    except Exception as ex:
                        logger.info('Cancelling...')
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise ex        
@dataclass
class segModelConfig:
    downloadList_path = os.path.join('components/artifacts',"downloadList.pkl")
    model_path = os.path.join('components/artifacts',"segModel_22042024.pth")
    norm_vals_path = os.path.join('components/artifacts',"norm_vals.pkl")

class segment():
    def __init__(self):
        self.inference_config = segModelConfig()

    def main(self):
        # check which scenes need to be downloaded
        downloadList = load_object(self.inference_config.downloadList_path)

        if len(downloadList)>0:
            # load image
            for img in downloadList[:1]:
                # load image based on image id
                eeImg = ee.Image.load(f'LANDSAT/LC08/C02/T1_TOA/{img[2:]}').select(["B1","B2","B3","B4","B5","B6","B7","B9"])
                # download scenes
                downloadPath = os.path.join('artifacts/segScenes', f"{img[2:]}.tif")
                BaseImage(eeImg).download(downloadPath, crs='EPSG:4326', region= eeImg.geometry(), scale=30, overwrite=True, num_threads=20, dtype= 'float64')
                #load model
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model = smp.Unet(
                    encoder_name="resnet34",        
                    encoder_weights= None,     
                    in_channels=10,                  
                    classes=2,  
                ).to(device)
                checkpoint = torch.load(self.inference_config.model_path)
                model.load_state_dict(checkpoint)
                # load transforms
                mean, std = load_object(self.inference_config.norm_vals_path)
                normalize = MyNormalize(mean=mean, stdev=std)

                # Create transforms
                data_transform = AugmentationSequential(
                    indices.AppendNDWI(index_green=2, index_nir=4),
                    indices.AppendNDVI(index_nir=4, index_red=3),
                    normalize,
                    data_keys = ["image"]
                )
                # run inference
                #5min13s to download and 1min to run inference
                dd = os.path.join('artifacts/segScenes/predictions', f"pred_{img[2:]}.tif")
                inference(infile = downloadPath, imgTransforms= data_transform, model = model, outfile = dd, patchSize = 512, num_workers=10, device = device)
                # upload to gee imagecollection

if __name__ == '__main__':
    segment().main()