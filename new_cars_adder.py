import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm  # For progress bars
import glob
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter  # For TensorBoard logging

# ---------------------------
# Configuration and Setup
# ---------------------------

# Paths
existing_model_path = 'path_to_existing_model.pt'  # Replace with your existing model path
car_model_path = 'path_to_car_models'  # Directory containing the expanded dataset
output_dir = 'path_to_output_dir'  # Replace with your desired output directory
checkpoints_dir = os.path.join(output_dir, 'checkpoints')
plots_dir = os.path.join(output_dir, 'plots')
tensorboard_logs_dir = os.path.join(output_dir, 'tensorboard_logs')

# Parameters
image_size = (299, 299)  # Adjusted to match architectures like Inception if needed
batch_size = 16
fine_tuning_learning_rate = 1e-5
patience = 10  # For early stopping
max_epochs = 20
accuracy_target = 0.99  # Target accuracy of 99%

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU detected. Running on CPU.")

# Create output directories
os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(tensorboard_logs_dir, exist_ok=True)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=tensorboard_logs_dir)

# ---------------------------
# Define the Ensemble Model
# ---------------------------

class EnsembleModel(nn.Module):
    def __init__(self, num_classes):
        super(EnsembleModel, self).__init__()
        # Model 1: ResNet50
        weights_resnet = models.ResNet50_Weights.DEFAULT
        self.model1 = models.resnet50(weights=weights_resnet)
        num_ftrs1 = self.model1.fc.in_features
        self.model1.fc = nn.Linear(num_ftrs1, num_classes)

        # Model 2: EfficientNet B0
        weights_efficientnet = models.EfficientNet_B0_Weights.DEFAULT
        self.model2 = models.efficientnet_b0(weights=weights_efficientnet)
        num_ftrs2 = self.model2.classifier[1].in_features
        self.model2.classifier[1] = nn.Linear(num_ftrs2, num_classes)

    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        x = (x1 + x2) / 2  # Average the outputs
        return x

# ---------------------------
# Utility Functions
# ---------------------------

def get_num_old_classes(existing_model_path, device):
    # Load the state_dict
    state_dict = torch.load(existing_model_path, map_location=device)
    # For ResNet50
    fc_weight_resnet = state_dict['model1.fc.weight']
    num_old_classes = fc_weight_resnet.shape[0]
    return num_old_classes

def load_and_modify_model(existing_model_path, num_old_classes, num_new_classes, device):
    # Load the existing model with old number of classes
    model = EnsembleModel(num_classes=num_old_classes)
    model = model.to(device)
    model.load_state_dict(torch.load(existing_model_path, map_location=device))
    
    # Modify the final layers to have num_new_classes output units
    # First, get the existing weights and biases
    # ResNet50
    old_fc_weights_resnet = model.model1.fc.weight.data.clone()
    old_fc_bias_resnet = model.model1.fc.bias.data.clone()
    num_ftrs1 = model.model1.fc.in_features
    # Replace the final layer
    model.model1.fc = nn.Linear(num_ftrs1, num_new_classes)
    # Initialize the new weights (the default initialization)
    nn.init.kaiming_normal_(model.model1.fc.weight, mode='fan_out', nonlinearity='relu')
    model.model1.fc.bias.data.zero_()
    # Copy over the existing weights
    model.model1.fc.weight.data[:num_old_classes] = old_fc_weights_resnet
    model.model1.fc.bias.data[:num_old_classes] = old_fc_bias_resnet

    # EfficientNet-B0
    old_fc_weights_efficientnet = model.model2.classifier[1].weight.data.clone()
    old_fc_bias_efficientnet = model.model2.classifier[1].bias.data.clone()
    num_ftrs2 = model.model2.classifier[1].in_features
    # Replace the final layer
    model.model2.classifier[1] = nn.Linear(num_ftrs2, num_new_classes)
    # Initialize the new weights
    nn.init.kaiming_normal_(model.model2.classifier[1].weight, mode='fan_out', nonlinearity='relu')
    model.model2.classifier[1].bias.data.zero_()
    # Copy over the existing weights
    model.model2.classifier[1].weight.data[:num_old_classes] = old_fc_weights_efficientnet
    model.model2.classifier[1].bias.data[:num_old_classes] = old_fc_bias_efficientnet

    return model

# ---------------------------
# Data Preparation
# ---------------------------

def prepare_data(car_model_path):
    # Load the dataset
    full_dataset = torchvision.datasets.ImageFolder(root=car_model_path)

    # Retrieve class names
    class_names = full_dataset.classes
    print(f"Number of classes: {len(class_names)}")
    num_classes = len(class_names)

    # Verify that the dataset is not empty
    if len(full_dataset) == 0:
        print("Dataset is empty. Please check your data directory.")
        exit()

    return full_dataset, num_classes

# Define TransformedDataset class
class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

# ---------------------------
# Data Augmentation and Normalization
# ---------------------------

def get_data_loaders(full_dataset):
    # Data Augmentation and Normalization
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Using ImageNet normalization
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Split dataset into training and validation
    num_samples = len(full_dataset)
    indices = list(range(num_samples))
    np.random.seed(123)
    np.random.shuffle(indices)
    split = int(np.floor(0.2 * num_samples))
    train_indices, val_indices = indices[split:], indices[:split]

    # Create subsets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # Apply transforms
    train_dataset = TransformedDataset(train_dataset, transform=train_transforms)
    val_dataset = TransformedDataset(val_dataset, transform=val_transforms)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

# ---------------------------
# Training Functions
# ---------------------------

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, scaler=None):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects.double() / total
    return epoch_loss, epoch_acc.item()

def validate(model, dataloader, criterion, device, scaler=None):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_corrects.double() / total
    return epoch_loss, epoch_acc.item()

# ---------------------------
# Plot Training History
# ---------------------------

def plot_history(history, title, filename, plots_dir):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')
    plt.suptitle(title)
    plt.savefig(os.path.join(plots_dir, filename))
    plt.show()

# ---------------------------
# Main Execution
# ---------------------------

if __name__ == '__main__':
    # Get number of old classes
    num_old_classes = get_num_old_classes(existing_model_path, device)
    print(f"Number of old classes: {num_old_classes}")

    # Prepare data
    full_dataset, num_new_classes = prepare_data(car_model_path)
    print(f"Number of new classes: {num_new_classes}")

    # Load and modify the model
    model = load_and_modify_model(existing_model_path, num_old_classes, num_new_classes, device)

    # Prepare data loaders
    train_loader, val_loader = get_data_loaders(full_dataset)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=fine_tuning_learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # Training Loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    epochs_no_improve = 0
    target_reached = False

    for epoch in range(max_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device, scaler)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        # Early stopping and checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'history': history,
            }, checkpoint_path)
            print(f"Validation loss decreased, saving model to {checkpoint_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping!")
                break

        # Check if target accuracy is reached
        if val_acc >= accuracy_target:
            print(f"Target accuracy of {accuracy_target*100}% reached.")
            break

    # Save the final model
    final_model_path = os.path.join(output_dir, "car_model_recognition_final_updated.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")

    # Plot training history
    plot_history(history, "Training History", "training_history.png", plots_dir)

    # Close the TensorBoard writer
    writer.close()
