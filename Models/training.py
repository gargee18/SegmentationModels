import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from dataloader_2d_segmentation import SegmentationDataset
from CustomUnet import CustomUnet
# from unet_pytorch_segmentation import UNet
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter


# Tune Hyperparameters (learning rate, epoch, batch size, seed, optimizer)
learning_rate=0.001
num_epochs = 500
batch_size= 8
random_seed=42
optimizer ="SDG" 
do_augmentation=False

# Initialize SummaryWriter
unet_depth= "2"
naming=optimizer
exp_name=naming+"_bs_"+str(batch_size)+"__lr_"+str(learning_rate)+"__epoc_"+str(num_epochs)+"__optim_"+str(optimizer)+"__unet_depth_"+str(unet_depth)+"_augmentation"+str(do_augmentation)
log_dir='/home/phukon/Desktop/Model_Fitting/runs/training_custom_unet_'+exp_name

# Create directory
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

# Define paths for annotations and source image, and create the dataloader
json_file_path = '/home/phukon/Desktop/Model_Fitting/annotations/train_annotations.json'
image_dir = '/home/phukon/Desktop/Model_Fitting/images/train_set/'

dataset = SegmentationDataset(json_file=json_file_path, image_dir=image_dir, augment= do_augmentation)
# dataset_aug = SegmentationDataset(json_file=json_file_path, image_dir=image_dir, augment= True)
# new_dataset = ConcatDataset([dataset, dataset_aug])

# Split dataset into train and validation sets
train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=0.2, shuffle=True, random_state=random_seed)

# Subset of train and val datasets at specified indices
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Define device, model, transformation loss function and optimizer 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomUnet().to(device)
criterion = nn.CrossEntropyLoss()  # Multi-class segmentation task
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


#DEBUG : uncomment to test
#test_class_weights_tensor=torch.tensor([1,1,1,1,1,1,1,1], dtype=torch.float32).to(device)

# Define a function to calculate weights for imbalanced classes
def compute_class_frequencies(mask):
    mask = mask.squeeze()
    mask = mask.cpu()
    flattened_mask = mask.flatten() # 1D
    class_frequencies = np.bincount(flattened_mask) # Count occurences
    print(f"class_frequencies= {class_frequencies}")
    return class_frequencies

def compute_and_print_weights(class_frequencies):
    total_samples = np.sum(class_frequencies)
    print(f"Sum= {total_samples}")
    num_classes = len(class_frequencies)
    print(f"Number of classes= {num_classes}")
    # weights = (1 / np.sqrt(class_frequencies))
    weights = np.where(class_frequencies > 0, 1 / np.sqrt(class_frequencies), 0)
    # weights = (total_samples / (num_classes * class_frequencies)) 
    print(f"Weights= {weights}")
    #weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
    return weights


# Weights for training dataset
for images, masks in train_loader:
    masks = masks.squeeze()
    masks= masks.to(device)
    class_frequencies = compute_class_frequencies(masks)
    total_elements = np.sum(class_frequencies)
    train_class_weights = compute_and_print_weights(class_frequencies)

    # Print frequencies for each class
    for class_index, frequency in enumerate(class_frequencies):
        percentage = (frequency / total_elements) * 100 if total_elements > 0 else 0
        print(f"Class {class_index}: {frequency} occurrences ({percentage:.2f}%)")

    for class_index, weight in enumerate(train_class_weights):
        print(f"Class {class_index}: Weight {weight:.4f}")

# Weights for validation dataset
for images, masks in val_loader:
    masks = masks.squeeze()
    masks= masks.to(device)
    class_frequencies = compute_class_frequencies(masks)
    total_elements = np.sum(class_frequencies)
    val_class_weights = compute_and_print_weights(class_frequencies)
    
    # Print frequencies for each class
    for class_index, frequency in enumerate(class_frequencies):
        percentage = (frequency / total_elements) * 100 if total_elements > 0 else 0
        print(f"Class {class_index}: {frequency} occurrences ({percentage:.2f}%)")

    for class_index, weight in enumerate(val_class_weights):
        print(f"Class {class_index}: Weight {weight:.4f}")


# Train and validate
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    all_labels = []
    all_preds = []

    # Training loop

    for images, masks in train_loader:
        # images =torch.unsqueeze(images, dim=1)
        images = images.permute(0, 1, 2, 3)
        images = images.to(device)
        masks = masks.squeeze()
        masks = masks.to(device)
        # Zero the gradients
        optimizer.zero_grad()

        # Forward Pass
        y_pred = model(images)  

        # Compute loss
        masks_int=masks.long()
        
        train_class_weights_tensor = torch.tensor(train_class_weights, dtype=torch.float32).to(device)
        

        num_classes = y_pred.size(1)
        if len(train_class_weights_tensor) != num_classes:
            raise ValueError(f"Number of class weights ({len(train_class_weights_tensor)}) does not match number of classes ({num_classes})")

        loss = F.cross_entropy(y_pred, masks_int,weight=train_class_weights_tensor, ignore_index=7)
        # loss=criterion(y_pred,masks_int)
        loss.backward()
        optimizer.step()  # Update the weights
        running_loss += loss.item()
        # Calculate predictions and accumulate labels
        all_labels = masks.flatten().cpu().numpy()  # Ensure the labels are flattened and moved to CPU
        all_preds = torch.argmax(y_pred, dim=1).flatten().cpu().numpy()  # Get predicted class indices and flatten

    # Calculate training F1 score
    train_f1 = f1_score(all_labels, all_preds, labels=[1,2,3,4,5,6],average='weighted')
    writer.add_scalar('F1/train', train_f1, epoch)
    writer.add_scalar('Loss/train', running_loss, epoch)

    # Validation loop
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    all_labels = []
    all_preds = []
    with torch.no_grad():

        for images, masks in val_loader:
            # images = torch.unsqueeze(images, dim=1)
            images = images.permute(0, 1, 2, 3).to(device)
            masks = masks.squeeze()
            masks = masks.to(device)

            val_class_weights_tensor = torch.tensor(val_class_weights, dtype=torch.float32).to(device)


            y_pred = model(images)
            loss = F.cross_entropy(y_pred, masks.long(), weight=val_class_weights_tensor, ignore_index=7)
            # loss = criterion(y_pred,masks.long())
            val_loss += loss.item()
    
            # Calculate predictions and accumulate labels
            all_labels = masks.flatten().cpu().numpy()  # Ensure the labels are flattened and moved to CPU
            all_preds = torch.argmax(y_pred, dim=1).flatten().cpu().numpy()  # Get predicted class indices and flatten

    # Calculate validation F1 score
    val_f1 = f1_score(all_labels, all_preds, labels=[1,2,3,4,5,6],average='weighted')
    writer.add_scalar('F1/val', val_f1, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {running_loss / len(train_loader):.4f}, Train F1 Score: {train_f1:.4f}, Validation Loss: {val_loss / len(val_loader):.4f}, Validation F1 Score: {val_f1:.4f}")

writer.close()




# Limit to first four images
num_images_to_display = 4
def display_individual_segmentation_masks(val_loader, num_images_to_display, class_names=None):

    if class_names is None:
        class_names = [
        "Background", 
        "Healthy Functional", 
        "Healthy Nonfunctional",
        "Necrotic Infected", 
        "Necrotic Dry", 
        "Bark", 
        "White Rot", 
        "Unknown"
    ]
    for images, masks in val_loader:
        images = images.permute(0, 1, 2, 3).to(device)
        masks = masks.to(device)

        with torch.no_grad():
            y_pred = model(images)

        batch_size = images.shape[0]  # Get the current batch size
        num_classes = y_pred.shape[1]  # Get the number of classes

        # Create a figure and axes for the images
        fig, axes = plt.subplots(min(batch_size, num_images_to_display), num_classes + 2, figsize=(15, min(batch_size, num_images_to_display) * 5))  # +2 for original and actual mask
        
        for i in range(min(batch_size, num_images_to_display)):  # Only loop through the first four images
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
            actual_mask = masks[i].cpu().numpy().squeeze()  # Remove singleton dimensions
            predicted_mask = torch.argmax(y_pred, dim=1)[i].detach().cpu().numpy()  # Get the predicted class
            
            # Display original image
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"Original Image {i + 1}")
            axes[i, 0].axis('off')  # Hide axes

            # Display actual segmentation
            axes[i, 1].imshow(actual_mask, cmap='jet', alpha=0.5)  
            axes[i, 1].set_title(f"Actual Mask {i + 1}")
            axes[i, 1].axis('off')  # Hide axes

            # Display predicted segmentation for each class
            for class_index in range(num_classes):
                class_mask = (predicted_mask == class_index).astype(np.float32) 
                axes[i, class_index + 2].imshow(class_mask, cmap='jet', alpha=0.5) 
                axes[i, class_index + 2].set_title(f"{class_names[class_index]}")
                axes[i, class_index + 2].axis('off') 

        plt.tight_layout()
        plt.show()

        break  

    
def display_segmentation(val_loader):
    for images, masks in val_loader:
        images = images.permute(0, 1, 2, 3).to(device)
        masks = masks.to(device)

        with torch.no_grad():
            y_pred = model(images)

        batch_size = images.shape[0]  # Get the current batch size
        cols = 3  # Number of columns for the grid
        rows = batch_size  # Each row will display one image with its masks

        # Create a figure and axes for the images
        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
        axes = axes.flatten()  # Flatten the axes array for easy indexing

        for i in range(batch_size):
            # Get the current image, actual mask, and predicted mask
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
            actual_mask = masks[i].cpu().numpy().squeeze()  
            predicted_mask = torch.argmax(y_pred, dim=1)[i].detach().cpu().numpy() 

            # Display original image
            axes[i * cols].imshow(img)
            axes[i * cols].set_title(f"Original Image {i + 1}")
            axes[i * cols].axis('off')  # Hide axes

            # Display actual segmentation
            axes[i * cols + 1].imshow(actual_mask, cmap='jet', alpha=0.5)  
            axes[i * cols + 1].set_title(f"Actual Mask {i + 1}")
            axes[i * cols + 1].axis('off')  # Hide axes

            # Display predicted segmentation
            axes[i * cols + 2].imshow(predicted_mask, cmap='jet', alpha=0.5)  
            axes[i * cols + 2].set_title(f"Predicted Mask {i + 1}")
            axes[i * cols + 2].axis('off')  # Hide axes

        plt.tight_layout()
        plt.show()
    
        break  
display_individual_segmentation_masks(val_loader, 4, class_names=None)

display_segmentation(val_loader)


















# for images, masks in val_loader:
#     # images =torch.unsqueeze(images, dim=1)
#     images = images.permute(0, 1, 2, 3)
#     images = images.to(device)
#     masks = masks.to(device)

#     optimizer.zero_grad()
#     y_pred = model(images)  
    
#     titles = [
#     "Background", "Healthy Functional", "Healthy Nonfunctional",
#     "Necrotic Infected", "Necrotic Dry", "Bark", "White Rot", "Unknown"
#     ]

#     # Create a figure and axes
#     fig, axes = plt.subplots(2, 4, figsize=(12, 6))

#     for i in range(8):
#         ax = axes[i // 4, i % 4]  # Determine the position in the subplot grid
#         img = y_pred[0, i].detach().cpu().numpy()
#         ax.imshow(img)
#         ax.set_title(titles[i])
#         plt.colorbar(ax.imshow(img), ax=ax)

#     plt.show()

# Define class names