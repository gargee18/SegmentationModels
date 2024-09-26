import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from dataloader_2d_segmentation import SegmentationDataset
from CustomUnetWithSkip import CustomUnetWithSkip
# from unet_pytorch_segmentation import UNet
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter


# Tune Hyperparameters (learning rate, epoch, batch size, seed, optimizer)
learning_rate=0.005
num_epochs = 500
batch_size= 16
random_seed=42
optimizer ="SGD" 
do_augmentation=False
activation = "ReLU"

# Initialize SummaryWriter
unet_depth= "2"
naming=optimizer
exp_name=naming+"_bs_"+str(batch_size)+"__lr_"+str(learning_rate)+"__epoc_"+str(num_epochs)+"__optim_"+str(optimizer)+"__unet_depth_"+str(unet_depth)+"__augmentation_"+str(do_augmentation)+"__activation_"+str(activation)
log_dir='/home/gargee/Model_Fitting/runs/training_custom_unet_with_skip_connections'+exp_name

# Create directory
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

# Define paths for annotations and source image, and create the dataloader
json_file_path = '/home/gargee/Model_Fitting/annotations/train_annotations.json'
image_dir = '/home/gargee/Model_Fitting/images/train_set/'

dataset = SegmentationDataset(json_file=json_file_path, image_dir=image_dir, augment= do_augmentation)

# Split dataset into train and validation sets
train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=0.2, shuffle=True, random_state=random_seed)

# Subset of train and val datasets at specified indices
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Define device, model, transformation loss function and optimizer 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomUnetWithSkip(1,8).to(device)
criterion = nn.CrossEntropyLoss()  # Multi-class segmentation task
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


#DEBUG : uncomment to test
#test_class_weights_tensor=torch.tensor([1,1,1,1,1,1,1,1], dtype=torch.float32).to(device)
def compute_class_frequencies(mask):
    mask = mask.squeeze().cpu().flatten()
    class_frequencies = np.bincount(mask) 
    print(f"class_frequencies= {class_frequencies}")
    return class_frequencies


def compute_and_print_weights(class_frequencies):
    # Avoid division by zero
    valid_frequencies = np.where(class_frequencies > 0, class_frequencies, 1)  # Replace 0 with 1 to avoid division by zero
    weights = np.sqrt(1 / valid_frequencies)  # Compute weights using inverse of frequencies
    weights /= weights.sum()  # Normalize weights
    return weights


# Weights for training dataset
for images, masks in train_loader:
    masks= masks.squeeze().to(device)
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
    masks = masks.squeeze().to(device)
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
        # print(f"Predicted mask {y_pred.shape}")
        # Compute loss
        masks_int=masks.long()
        # print(f"Input mask {masks_int.shape}")
        train_class_weights_tensor = torch.tensor(train_class_weights, dtype=torch.float32).to(device)
        

        num_classes = y_pred.size(1)
        if len(train_class_weights_tensor) != num_classes:
            raise ValueError(f"Number of class weights ({len(train_class_weights_tensor)}) does not match number of classes ({num_classes})")

        #masks_int[masks_int==0]=7
        loss = F.cross_entropy(y_pred, masks_int, weight=train_class_weights_tensor,  ignore_index=7 )
        # loss=criterion(y_pred,masks_int)
        loss.backward()
        optimizer.step()  # Update the weights
        running_loss += loss.item()
        # Calculate predictions and accumulate labels
        all_labels = masks.flatten().cpu().numpy()  # Ensure the labels are flattened and moved to CPU
        # print(f"All labels{all_labels.shape}")
        all_preds = torch.argmax(y_pred[:,0:7,:,:], dim=1).flatten().cpu().numpy()  # Get predicted class indices and flatten


        # # Get probabilities for each class
        # softmax_probs = F.softmax(y_pred[:,0:7,:,:], dim=1)
        # # Get predicted class indices
        # all_preds = torch.argmax(softmax_probs, dim=1).flatten().cpu().numpy()



        # print(f"All preds argmax {all_preds.shape}")
        # all_preds = torch.softmax(y_pred[:,0:7,:,:], dim=1).detach().cpu()
        # print(f"Softmax {all_preds.shape}")
    # Calculate training F1 score
    train_f1 = f1_score(all_labels, all_preds, labels=[1,2,3,4,5,6],average='weighted')
    train_f1_hf = f1_score(all_labels, all_preds, labels=[1],average='weighted')
    train_f1_necd = f1_score(all_labels, all_preds, labels=[3],average='weighted')
    writer.add_scalar('F1/train', train_f1, epoch)
    writer.add_scalar('F1/train/hf', train_f1_hf, epoch)
    writer.add_scalar('F1/train/necd', train_f1_necd, epoch)
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
            # print(len(masks))
            # print(masks.size())
            masks = masks.squeeze()
            # print(masks.size())
            masks = masks.to(device)

            val_class_weights_tensor = torch.tensor(val_class_weights, dtype=torch.float32).to(device)


            y_pred = model(images)
            # print(y_pred.size())
            masks_int=masks.long()
            #masks_int[masks_int==0]=7
            loss = F.cross_entropy(y_pred, masks_int,weight=val_class_weights_tensor,  ignore_index=7)
            # loss = criterion(y_pred,masks.long())
            val_loss += loss.item()
    
            # Calculate predictions and accumulate labels
            all_labels = masks.flatten().cpu().numpy()  # Ensure the labels are flattened and moved to CPU
            all_preds = torch.argmax(y_pred[:,0:7,:,:], dim=1).flatten().cpu().numpy()  # Get predicted class indices and flatten
            # Get probabilities for each class
            # softmax_probs = F.softmax(y_pred[:,0:7,:,:], dim=1)
            # # # Get predicted class indices
            # all_preds = torch.argmax(softmax_probs, dim=1).flatten().cpu().numpy()
            # all_preds = torch.softmax(y_pred[:,0:7,:,:], dim=1).flatten().cpu()
    # Calculate validation F1 score
    val_f1 = f1_score(all_labels, all_preds, labels=[1,2,3,4,5,6],average='weighted')
    val_f1_hf=f1_score(all_labels, all_preds, labels=[1],average='weighted')
    val_f1_necd=f1_score(all_labels, all_preds, labels=[3],average='weighted')
    writer.add_scalar('F1/val', val_f1, epoch)
    writer.add_scalar('F1/val/hf', val_f1_hf, epoch)
    writer.add_scalar('F1/val/necd', val_f1_necd, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)

    print(f"[{epoch + 1}/{num_epochs}], TRLoss: {running_loss / len(train_loader):.4f}, ValLoss: {val_loss / len(val_loader):.4f}, TrainF1: {train_f1:.4f}, ValF1: {val_f1:.4f}, ValF1hf: {val_f1_hf:.4f}, ValF1necd: {val_f1_necd:.4f}")

writer.close()



def display_segmentation_with_overlay(val_loader, num_images_to_display=4, class_names=None, overlay_opacity=0.5):
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
    
    images_displayed = 0  # Keep track of how many images have been displayed

    for images, masks in val_loader:
        images = images.permute(0, 1, 2, 3).to(device)  # Permute to (B, C, H, W)
        masks = masks.to(device)

        with torch.no_grad():
            y_pred = model(images)

        batch_size = images.shape[0]  # Get the current batch size
        images_to_display = min(batch_size, num_images_to_display - images_displayed)  # Number of images to display in this batch

        # Create a figure and axes for the images, 4 columns for each: original, actual mask, predicted mask, overlay
        fig, axes = plt.subplots(images_to_display, 4, figsize=(20, images_to_display * 5))

        # Ensure that axes is always 2D, even if there's only one row
        axes = np.atleast_2d(axes)

        for i in range(images_to_display):
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)

            # Convert the RGB image to grayscale (average of R, G, B channels)
            grayscale_img = np.mean(img, axis=-1)

            actual_mask = masks[i].cpu().numpy().squeeze()  # Remove singleton dimensions
            predicted_mask = torch.argmax(y_pred, dim=1)[i].detach().cpu().numpy()  # Get the predicted mask

            # Display the original image (converted to grayscale)
            axes[i, 0].imshow(grayscale_img, cmap='gray')
            axes[i, 0].set_title(f"Original Grayscale Image {images_displayed + i + 1}")
            axes[i, 0].axis('off')

            # Display the actual segmentation mask
            axes[i, 1].imshow(actual_mask, cmap='jet', alpha=0.5)
            axes[i, 1].set_title(f"Expected Mask {images_displayed + i + 1}")
            axes[i, 1].axis('off')

            # Display the predicted segmentation mask
            axes[i, 2].imshow(predicted_mask, cmap='jet', alpha=0.5)
            axes[i, 2].set_title(f"Predicted Mask {images_displayed + i + 1}")
            axes[i, 2].axis('off')

            # Overlay the predicted mask on the grayscale image with low opacity
            axes[i, 3].imshow(grayscale_img, cmap='gray')  # Display grayscale image
            axes[i, 3].imshow(predicted_mask, cmap='jet', alpha=overlay_opacity)  # Overlay colorful mask
            axes[i, 3].set_title(f"Overlay {images_displayed + i + 1}")
            axes[i, 3].axis('off')

        plt.tight_layout()
        plt.show() #This line generate a bug when run in ssh on phenodrone : 


        images_displayed += images_to_display  # Update the count of displayed images

        if images_displayed >= num_images_to_display:
            break  # Stop if we have displayed the requested number of images
            #libGL error: MESA-LOADER: failed to open swrast: /usr/lib/dri/swrast_dri.so: 
            #Ne peut ouvrir le fichier d'objet partagé: Aucun fichier ou dossier de ce nom (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
            #libGL error: failed to load driver: swrast
    



display_segmentation_with_overlay(val_loader, 4, class_names=None)


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