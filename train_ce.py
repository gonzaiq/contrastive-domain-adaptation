import os
import json
from comet_ml import Experiment
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from torchvision import models
from pytorch_lightning.loggers import CometLogger
from src.models.resnet import CustomResNet
from src.data.dataset import OfficeDataset, load_datalist
from src.trainers.ce_trainer import CrossEntropyTrainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameters for training")

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--image_size', type=int, default=64, help="Image size (after resizing operation)")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=20, help="Number of epochs for training")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for data loading")
    parser.add_argument('--model_name', type=str, default="resnet18", help="Name of the model (densenet or resnet supported)")
    parser.add_argument('--output_dir', type=str, default="results_ce", help="Directory to store the results")
    parser.add_argument('--data_base_dir', type=str, default="", help="Data and datalist directory")
    parser.add_argument("--weighted_sampler", action="store_true", help="Whether to use the weighted sampler or not")
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument('--datalist_pth', type=str, default="data/office/datalist_train_valid.json", help="Path to the datalist JSON file")
    parser.add_argument('--default_root_dir', type=str, default="", help="Default root directory for Pytorch Lightning")
    parser.add_argument('--comet_ml_project_name', type=str, default="contra-da-office", help="Project name, useful for Comet ML logging.")
    parser.add_argument('--comet_ml_exp_name', type=str, default="", help="Experiment name name, useful for Comet ML logging.")
    parser.add_argument('--comet_ml_workspace', type=str, default="gonzaq94", help="Comet ML workspace for logging.")
    
    return parser.parse_args()

args = parse_args()
os.environ["COMET_LOG_DIR"] = os.path.join(os.getenv("WORKDIR"), "comet_logs")

# Main function for running the training
def main():

    # initialize Comet experiment
    experiment = Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name=args.comet_ml_project_name,
        workspace=args.comet_ml_workspace
        
    )

    if len(args.comet_ml_exp_name) > 0:
        experiment.set_name(args.comet_ml_exp_name)

    comet_logger = CometLogger(
        api_key=os.getenv("COMET_API_KEY"), 
        project_name=args.comet_ml_project_name, 
        workspace=args.comet_ml_workspace,
        experiment_name=args.comet_ml_exp_name if len(args.comet_ml_exp_name) > 0 else None
    )

    progress_bar = TQDMProgressBar()

    # Initialize PyTorch Lightning model with ResNet
    with open(os.path.join(args.data_base_dir, "labels_dct.json"), 'r') as file:
        labels_dct = json.load(file)
    num_classes = len(labels_dct)

    with open(os.path.join(args.data_base_dir, "domains_dct.json"), 'r') as file:
        domains_dct = json.load(file)
    num_domains = len(domains_dct)

    model = CrossEntropyTrainer(
        model_name=args.model_name, 
        num_classes=num_classes, 
        learning_rate=args.learning_rate, 
        num_domains=num_domains
        )

    # Checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,  # Change this to a location with space
        filename="best_val_f1_macro",  # Custom filename format
        save_top_k=1,  # Save only the top models
        monitor="val_f1_macro",  # Monitor validation loss
        mode="max",  # Save checkpoints with lowest val_loss
    )

    # Datasets and dataloaders
    datalist = load_datalist(args.datalist_pth)

    # Define the transform for the training data
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),  # Resize
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.RandomVerticalFlip(),  # Random vertical flip
        transforms.RandomRotation(degrees=180),  # Rotate the image by a random degree in the range (-180, 180)
        transforms.ToTensor(),  # Convert image to tensor
        # Optional: Add normalization here if needed, e.g., transforms.Normalize(mean, std)
    ])

    # Define the transform for the validation data
    valid_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),  # Resize
        transforms.ToTensor(),  # Convert image to tensor
        # Optional: Add normalization here if needed
    ])

    train_dataset = OfficeDataset(datalist["training"], transform=train_transform)
    val_dataset = OfficeDataset(datalist["validation"], transform=valid_transform)
    test_dataset = OfficeDataset(datalist["test"], transform=valid_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Initialize Trainer and train the model
    trainer = pl.Trainer(
        default_root_dir=args.default_root_dir,
        max_epochs=args.num_epochs,
        val_check_interval=1.0,
        logger=comet_logger,
        callbacks=[progress_bar, checkpoint_callback],
        log_every_n_steps=1,
        num_sanity_val_steps=0
        )
    trainer.fit(model, train_dataloader, val_dataloader)

    # Evaluate best model on test set
    best_model_pth = os.path.join(args.checkpoint_dir, "best_val_f1_macro.ckpt")
    best_model = ImageClassifier.load_from_checkpoint(best_model_pth, model_name=args.model_name, num_classes=3)
    test_results = trainer.test(best_model, test_dataloader)

    # Log test results to Comet
    experiment.log_metrics({f"test_{k}": v for k, v in test_results[0].items()})

    # Log and register model in Comet
    registered_model_name = args.comet_ml_project_name + "-" + experiment.get_name()
    experiment.log_model(registered_model_name, best_model_pth, overwrite=True)
    experiment.register_model(registered_model_name)
    print(f"Registered checkpoint {os.path.join(args.checkpoint_dir, 'best_val_f1_macro.ckpt')}")

    # End comet experiment
    experiment.end()

if __name__ == "__main__":
    main()