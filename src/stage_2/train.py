"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
from src.stage2 import data_setup, engine, utils

from torchgeo.datasets import RasterDataset
from torchgeo.transforms import indices
from torchgeo.transforms import AugmentationSequential
import segmentation_models_pytorch as smp

# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001

# Setup directories
train_dir = "C:/Users/coach/myfiles/postdoc/Fire/data/indonesia/fire_data/train"
test_dir = "C:/Users/coach/myfiles/postdoc/Fire/data/indonesia/fire_data/test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# setup normalisation
train_imgs = RasterDataset(paths=(train_dir/'images').as_posix(), crs='epsg:4326', res= 0.00025, transforms=scale)
mean, std = utils.calc_statistics(train_imgs)
normalize = utils.MyNormalize(mean=mean, stdev=std)

# Create transforms
data_transform = AugmentationSequential(
    indices.AppendNDWI(index_green=2, index_nir=4),
    indices.AppendNDVI(index_nir=4, index_red=3),
    normalize,
    data_keys = ["image"]
)

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Create model with help from model_builder.py

model = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights= None,     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=10,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2,                      # model output channels (number of classes in your dataset)
).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name="05_going_modular_script_mode_tinyvgg_model.pth")