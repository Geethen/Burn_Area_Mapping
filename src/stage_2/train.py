"""
Trains a PyTorch image classification model using device-agnostic code.
"""
import os
import argparse
import torch
import data_setup, engine, utils
from src.utils import load_object, save_object
from sklearn.metrics import jaccard_score
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

    # Check if norm_vals pickle file exists in the artifacts folder
    norm_vals_file = os.path.join(r'src/components/artifacts', "norm_vals.pkl")

    if not os.path.exists(norm_vals_file):
        # Compute mean and std deviation
        train_imgs = RasterDataset(paths=(train_dir+'/images'), crs='epsg:4326', res=0.00025)
        mean, std = utils.calc_statistics(train_imgs)

        # Save mean and std deviation to pickle file
        save_object(norm_vals_file, (mean, std))
    else:
        # Load mean and std deviation from pickle file
        mean, std = load_object(norm_vals_file)

    # Setup normalization
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
        n_trainimages = 158,
        n_testimages = 69,
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
    def loss_fn(p, t):    
        return torch.nn.functional.cross_entropy(p, t.squeeze())

    def oa(pred, y):
        flat_y = y.squeeze()
        flat_pred = pred.argmax(dim=1)
        acc = torch.count_nonzero(flat_y == flat_pred) / torch.numel(flat_y)
        return acc

    def iou(pred, y):
        flat_y = y.cpu().numpy().squeeze()
        flat_pred = pred.argmax(dim=1).detach().cpu().numpy()
        return jaccard_score(flat_y.reshape(-1), flat_pred.reshape(-1), zero_division=1.)
    
    acc_fns = [oa, iou]
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate, weight_decay=args.weight_decay)

    # Start training with help from engine.py
    engine.train(model=model,
                 train_dataloader=train_dataloader,
                 test_dataloader=test_dataloader,
                 batch_tfms= data_transform,
                 loss_fn=loss_fn,
                 acc_fns= acc_fns,
                 optimizer=optimizer,
                 epochs=args.epochs,
                 callbacks= {'metric_index': 1, 'save_model_path': r"C:\Users\coach\myfiles\postdoc\Fire\code\Burn_Area_Mapping\src\components\artifacts\segModel_22042024.pth"},
                 device=device)

if __name__ == "__main__":
    main()