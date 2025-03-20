import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchinfo import summary
import os
import ssl
import matplotlib.pyplot as plt
from resnet import CustomResNet

data_root = "."

## for the supervised contrastive loss
# https://github.com/HobbitLong/SupContrast

# bypass the use of certificates
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings
ssl._create_default_https_context = ssl._create_unverified_context
########

RETAIN_RATIO = 1.0

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
amazon_dir = os.path.join(data_root, "data/office/amazon/images")
dslr_dir = os.path.join(data_root, "data/office/dslr/images")
webcam_dir = os.path.join(data_root, "data/office/webcam/images")

# Hyperparameters
num_classes = len(os.listdir(amazon_dir))
batch_size = 128
learning_rate = 0.001
num_epochs = 100

def print_kernel_sizes(model):
    for name, layer in model.named_children():
        # Print the name of the layer
        print(f"Layer: {name}")

        # Check for Conv2d layers to print kernel size
        if isinstance(layer, nn.Conv2d):
            print(f"  Kernel Size: {layer.kernel_size}")
            print(f"  Stride: {layer.stride}")
            print(f"  Padding: {layer.padding}")

        # Check for other layers (e.g., MaxPool, AveragePool)
        elif isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d)):
            print(f"  Kernel Size: {layer.kernel_size}")
            print(f"  Stride: {layer.stride}")
            print(f"  Padding: {layer.padding}")

        # Recursively go through child modules
        if len(list(layer.children())) > 0:
            print_kernel_sizes(layer)

"""
# Data transformations
data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# Load individual datasets
amazon_dataset = datasets.ImageFolder(root=amazon_dir, transform=data_transforms["train"])
dslr_dataset = datasets.ImageFolder(root=dslr_dir, transform=data_transforms["train"])
webcam_dataset = datasets.ImageFolder(root=webcam_dir, transform=data_transforms["train"])

# Merge the datasets
merged_dataset = ConcatDataset([amazon_dataset, dslr_dataset, webcam_dataset])

# Calculate sizes
total_size = len(merged_dataset)
retain_size = int(total_size * RETAIN_RATIO)
discard_size = total_size - retain_size

# Split the dataset
merged_dataset, _ = random_split(merged_dataset, [retain_size, discard_size])

# Splitting dataset into train, validation, and test
train_size = int(0.8 * len(merged_dataset))
val_size = int(0.1 * len(merged_dataset))  # 10% for validation
test_size = len(merged_dataset) - train_size - val_size  # Remaining for test

# Random split for train and validation
train_dataset, temp_dataset = torch.utils.data.random_split(merged_dataset, [train_size, len(merged_dataset) - train_size])
val_dataset, test_dataset = torch.utils.data.random_split(temp_dataset, [val_size, test_size])

# Saving the splits (as example, you can adjust based on your needs)
torch.save(train_dataset, 'train_dataset_colab.pt')
torch.save(val_dataset, 'val_dataset_colab.pt')
torch.save(test_dataset, 'test_dataset_colab.pt')
"""

# Loading the saved datasets
train_dataset = torch.load(os.path.join(data_root, 'train_dataset.pt'))
val_dataset = torch.load(os.path.join(data_root, 'val_dataset.pt'))
test_dataset = torch.load(os.path.join(data_root, 'test_dataset.pt'))

# DataLoaders
dataloaders = {
    "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
    "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
}

# Initialize ResNet
model = CustomResNet(
    block=models.resnet.BasicBlock,
    layers=[2, 2, 2, 2],
    num_classes=num_classes,
    in_kernel_size=3,
    inplanes=16
    )
model = model.to(device)
model = model.to(device)
summary(model, input_size=(1, 3, 224, 224))  # Adjust input size to match your model's requirements

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Lists to save the loss and accuracy values
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training loop
def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 20)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "train":
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc.cpu().numpy())
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc.cpu().numpy())

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), os.path.join(data_root, "results/wo_da", "best_resnet_office_amazon.pth"))

    print(f"Best Validation Accuracy: {best_acc:.4f}")

    # Optionally, save the metrics to a file
    torch.save({
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }, 'training_metrics.pth')

    # Plot the training and validation metrics
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(data_root, "results/wo_da", "train_curves.png"))

# Train the model
train_model(model, dataloaders, criterion, optimizer, num_epochs)