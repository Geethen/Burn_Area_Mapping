"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import argparse
import torch
from src.stage2 import data_setup, engine, utils
from torchgeo.datasets import RasterDataset
from torchgeo.transforms import indices, AugmentationSequential
import segmentation_models_pytorch as smp

def parse_args():
    parser = argparse.ArgumentParser(description="Trains a PyTorch image classification model using device-agnostic code.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--train_dir", type=str, default="C:/Users/coach/myfiles/postdoc/Fire/data/indonesia/fire_data/train", help="Training directory")
    parser.add_argument("--test_dir", type=str, default="C:/Users/coach/myfiles/postdoc/Fire/data/indonesia/fire_data/test", help="Testing directory")
    return parser.parse_args()

def main():
    args = parse_args()

    # Setup directories
    train_dir = args.train_dir
    test_dir = args.test_dir

    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # setup normalisation
    train_imgs = RasterDataset(paths=(train_dir/'images').as_posix(), crs='epsg:4326', res= 0.00025)
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
    train_dataloader, test_dataloader = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=args.batch_size
    )

    # Create model using segmentation_models_pytorch
    model = smp.Unet(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights= None,     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=10,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=2,                      # model output channels (number of classes in your dataset)
    ).to(device)

    # Set loss and optimizer
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate, weight_decay=args.weight_decay)

    # Start training with help from engine.py
    engine.train(model=model,
                 train_dataloader=train_dataloader,
                 test_dataloader=test_dataloader,
                 loss_fn=loss_fn,
                 optimizer=optimizer,
                 epochs=args.epochs,
                 device=device)

if __name__ == "__main__":
    main()