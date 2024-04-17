"""
Contains functionality for creating PyTorch DataLoaders for 
semantic segmentation data.
"""
from torchgeo.datasets import RasterDataset,  stack_samples
from torchgeo.samplers import RandomGeoSampler, Units
from torch.utils.data import DataLoader
from ensure import ensure_annotations

@ensure_annotatons
def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    crs: str = 'epsg:4326',
    res: float = 0.00025,
    img_size: int = 512,
    n_trainimages: int = 100,
    n_testimages: int = 50,
    batch_size: int = 8
):
  """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory. The train images and masks must be within this
      dir in folders named 'iamges  and 'masks'. 
    test_dir: Path to testing directory. The train images and masks must be within this
      dir in folders named 'iamges  and 'masks'. 
    crs: The CRS of the dataset. Defaults to EPSG:4326.
    res: The pixel size of the images in units of the CRS. For the default EPSG:4326,
      30m is provided as 0.00025 decimal degrees.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader).
    Example usage:
      train_dataloader, test_dataloader = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             batch_size=8)
  """
  train_imgs = RasterDataset(paths=(train_dir/'images').as_posix(), crs= crs, res= res)
  train_msks = RasterDataset(paths=(train_dir/'masks').as_posix(), crs= crs, res= res)

  valid_imgs = RasterDataset(paths=(test_dir/'images').as_posix(), crs= crs, res= res)
  valid_msks = RasterDataset(paths=(test_dir/'masks').as_posix(), crs= crs, res= res)

  # IMPORTANT
  train_msks.is_image = False
  valid_msks.is_image = False

  train_dset = train_imgs & train_msks
  valid_dset = valid_imgs & valid_msks

  train_sampler = RandomGeoSampler(train_imgs, size= img_size, length= n_trainimages, units=Units.PIXELS)
  valid_sampler = RandomGeoSampler(valid_imgs, size= img_size, length= n_testimages, units=Units.PIXELS)

  train_dataloader = DataLoader(train_dset, sampler=train_sampler, batch_size= batch_size, collate_fn=stack_samples)
  valid_dataloader = DataLoader(valid_dset, sampler=valid_sampler, batch_size= batch_size, collate_fn=stack_samples)
  
  return train_dataloader, valid_dataloader