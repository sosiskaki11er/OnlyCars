import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter  # For TensorBoard logging
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm  # For progress bars
import glob

# ---------------------------
# Configuration and Setup
# ---------------------------

# Parameters
image_size = (299, 299)  # Adjusted to match architectures like Inception if needed
batch_size = 16
initial_learning_rate = 1e-4
fine_tuning_learning_rate = 1e-5
patience = 10  # For early stopping
min_epochs = 4  # Minimum epochs before checking for accuracy
accuracy_target = 0.99  # Target accuracy of 99%

# ---------------------------
# Data Preparation
# --------------------------- 

def prepare_data(car_model_path):
    # Collect all image file paths
    image_file_paths = []
    supported_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

    for class_folder in os.listdir(car_model_path):
        class_folder_path = os.path.join(car_model_path, class_folder)
        if os.path.isdir(class_folder_path):
            for filename in os.listdir(class_folder_path):
                file_path = os.path.join(class_folder_path, filename)
                if os.path.isfile(file_path):
                    image_file_paths.append((file_path, class_folder_path))

    # Total number of images
    total_images = len(image_file_paths)
    print(f"Total images found: {total_images}")

    # Fix file extensions and remove corrupted images with a single progress bar
    print("Processing images...")
    valid_image_file_paths = []

    def is_image_corrupted(file_path):
        try:
            with Image.open(file_path) as img:
                img.load()  # Force loading of image data
            return False
        except (UnidentifiedImageError, IOError) as e:
            print(f"Corrupted image detected and removed: {file_path}")
            return True

    def fix_extension(file_path):
        base, ext = os.path.splitext(file_path)
        if ext.lower() not in supported_extensions and ext.upper() in [e.upper() for e in supported_extensions]:
            new_ext = ext.lower()
            new_file_path = base + new_ext
            try:
                os.rename(file_path, new_file_path)
                print(f"Renamed {file_path} to {new_file_path}")
                return new_file_path
            except Exception as e:
                print(f"Failed to rename {file_path}: {e}")
                return file_path
        return file_path

    with tqdm(total=total_images, desc="Processing images", unit="image") as pbar:
        for file_path, class_folder_path in image_file_paths:
            filename = os.path.basename(file_path)
            # Fix extension if needed
            file_path = fix_extension(file_path)
            filename = os.path.basename(file_path)
            # Check if file is an image
            if filename.lower().endswith(supported_extensions):
                # Use adjusted corruption check
                corrupted = is_image_corrupted(file_path)
                if not corrupted:
                    valid_image_file_paths.append(file_path)
                else:
                    try:
                        os.remove(file_path)
                        print(f"Removed corrupted image: {file_path}")
                    except Exception as e:
                        print(f"Failed to remove {file_path}: {e}")
            else:
                print(f"Skipping non-image file: {file_path}")
            pbar.update(1)

    # Remove empty class folders
    for class_folder in os.listdir(car_model_path):
        class_folder_path = os.path.join(car_model_path, class_folder)
        if os.path.isdir(class_folder_path):
            has_image = False
            for filename in os.listdir(class_folder_path):
                file_path = os.path.join(class_folder_path, filename)
                if os.path.isfile(file_path) and filename.lower().endswith(supported_extensions):
                    has_image = True
                    break
            if not has_image:
                try:
                    print(f"Removing empty class folder: {class_folder_path}")
                    os.rmdir(class_folder_path)
                except Exception as e:
                    print(f"Failed to remove {class_folder_path}: {e}")

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

# Move TransformedDataset class to global scope
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
# Model Definitions (Ensemble)
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

        # Do not freeze parameters here

    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        x = (x1 + x2) / 2  # Average the outputs
        return x

# ---------------------------
# Compile the Model
# ---------------------------

def get_model_and_optimizer(num_classes, device, fine_tuning=False):
    model = EnsembleModel(num_classes=num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    if not fine_tuning:
        # Initial Training Phase: Single Parameter Group
        optimizer = optim.AdamW(model.parameters(), lr=initial_learning_rate, weight_decay=1e-4)
    else:
        # Fine-Tuning Phase: Single Parameter Group with Lower Learning Rate
        optimizer = optim.AdamW(model.parameters(), lr=fine_tuning_learning_rate, weight_decay=1e-4)
    
    # Scheduler for learning rate decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    # Correct GradScaler Initialization
    scaler = torch.amp.GradScaler() if device.type == 'cuda' else None  # For mixed precision
    return model, criterion, optimizer, scheduler, scaler

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
    # Set working directory
    os.chdir("D:\\archive")

    # Paths
    car_model_path = os.path.join(os.getcwd(), "car_models-master")
    output_dir = os.path.join(os.getcwd(), "output")
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    plots_dir = os.path.join(output_dir, "plots")
    tensorboard_logs_dir = os.path.join(output_dir, "tensorboard_logs")

    # Create output directories
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tensorboard_logs_dir, exist_ok=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=tensorboard_logs_dir)

    # Device Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected. Running on CPU.")

    # Prepare data
    full_dataset, num_classes = prepare_data(car_model_path)
    train_loader, val_loader = get_data_loaders(full_dataset)

    # ---------------------------
    # Resume Training from Checkpoint if Available
    # ---------------------------

    start_epoch = 0
    fine_tuning = False  # Default training phase

    checkpoint_files = glob.glob(os.path.join(checkpoints_dir, 'checkpoint_epoch_*.pt'))
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime) if checkpoint_files else None

    if latest_checkpoint:
        print(f"Loading checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        
        # Retrieve the fine_tuning flag from the checkpoint
        fine_tuning = checkpoint.get('fine_tuning', False)
        
        # Initialize model and optimizer based on the fine_tuning flag
        model, criterion, optimizer, scheduler, scaler = get_model_and_optimizer(num_classes, device, fine_tuning=fine_tuning)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer and scheduler states
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # Load training parameters
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        history = checkpoint['history']
        print(f"Resuming training from epoch {start_epoch}, Fine-Tuning Phase: {fine_tuning}")
    else:
        # No checkpoint found, initialize everything
        model, criterion, optimizer, scheduler, scaler = get_model_and_optimizer(num_classes, device, fine_tuning=False)
        best_val_loss = float('inf')
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        print("Starting training from scratch.")

    # ---------------------------
    # Training Loop
    # ---------------------------

    epochs_no_improve = 0
    target_reached = False

    while not target_reached:
        for epoch in range(start_epoch, start_epoch + min_epochs):
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
                    'fine_tuning': fine_tuning
                }, checkpoint_path)
                print(f"Validation loss decreased, saving model to {checkpoint_path}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("Early stopping!")
                    target_reached = True
                    break

            # Check if target accuracy is reached
            if val_acc >= accuracy_target:
                print(f"Target accuracy of {accuracy_target*100}% reached.")
                target_reached = True
                break

        # Update start_epoch for the next loop
        start_epoch = epoch + 1

        # ---------------------------
        # Unfreeze All Layers for Fine-Tuning
        # ---------------------------
        if not fine_tuning:
            print("Unfreezing all layers for fine-tuning...")
            for param in model.parameters():
                param.requires_grad = True

            # Update the learning rate for fine-tuning
            for param_group in optimizer.param_groups:
                param_group['lr'] = fine_tuning_learning_rate

            # Update the fine_tuning flag
            fine_tuning = True

            # Save a checkpoint indicating the switch to fine-tuning
            checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_epoch_{epoch+1}_fine_tuning.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'history': history,
                'fine_tuning': fine_tuning
            }, checkpoint_path)
            print(f"Switching to fine-tuning phase. Saving checkpoint to {checkpoint_path}")

    # ---------------------------
    # Save the Final Model
    # ---------------------------

    final_model_path = os.path.join(output_dir, "car_model_recognition_final.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")

    # ---------------------------
    # Plot Training History
    # ---------------------------

    plot_history(history, "Training History", "training_history.png", plots_dir)

    # Close the TensorBoard writer
    writer.close()
