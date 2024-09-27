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
from CustomUnet import CustomUnet
import Utils
# from unet_pytorch_segmentation import UNet
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter


# Tune Hyperparameters (learning rate, epoch, batch size, seed, optimizer)
learning_rate=0.001
num_epochs = 2000
batch_size= 8
random_seed=42
optimizer ="SGD" 
do_augmentation=True
activation = "ReLU"

# Initialize SummaryWriter
unet_depth= "4"
naming=optimizer
exp_name=naming+"_bs_"+str(batch_size)+"__lr_"+str(learning_rate)+"__epoc_"+str(num_epochs)+"__optim_"+str(optimizer)+"__unet_depth_"+str(unet_depth)+"__augmentation_"+str(do_augmentation)+"__activation_"+str(activation)
log_dir='/home/phukon/Desktop/Model_Fitting/runs/training_custom_unet_with_skip_connections'+exp_name

# Create directory
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

# Define paths for annotations and source image, and create the dataloader
json_file_path = '/home/phukon/Desktop/Model_Fitting/annotations/train_annotations.json'
image_dir = '/home/phukon/Desktop/Model_Fitting/images/train_set/'

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
# model = CustomUnet().to(device)

model = CustomUnetWithSkip(1,8).to(device)
criterion = nn.CrossEntropyLoss()  # Multi-class segmentation task
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


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
        masks_int=masks.long()
        num_classes = y_pred.size(1)
        train_class_weights_tensor = torch.tensor(Utils.get_weights(train_loader,device), dtype=torch.float32).to(device)
        if len(train_class_weights_tensor) != num_classes:
            raise ValueError(f"Number of class weights ({len(train_class_weights_tensor)}) does not match number of classes ({num_classes})")
        loss = F.cross_entropy(y_pred, masks_int, weight=train_class_weights_tensor,  ignore_index=7 )
        # loss=criterion(y_pred,masks_int)
        loss.backward()
        optimizer.step()  # Update the weights
        running_loss += loss.item()
        all_labels = masks.flatten().cpu().numpy()  # Ensure the labels are flattened and moved to CPU
        all_preds = torch.argmax(y_pred[:,0:7,:,:], dim=1).flatten().cpu().numpy()  # Get predicted class indices and flatten

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
            images = images.permute(0, 1, 2, 3).to(device)
            masks = masks.squeeze().to(device)
            y_pred = model(images)
            masks_int=masks.long()
            val_class_weights_tensor = torch.tensor(Utils.get_weights(val_loader,device), dtype=torch.float32).to(device)
            loss = F.cross_entropy(y_pred, masks_int,weight=val_class_weights_tensor,  ignore_index=7)
            val_loss += loss.item()
            # Calculate predictions and accumulate labels
            all_labels = masks.flatten().cpu().numpy()  # Ensure the labels are flattened and moved to CPU
            all_preds = torch.argmax(y_pred[:,0:7,:,:], dim=1).flatten().cpu().numpy()  # Get predicted class indices and flatten

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
Utils.display_segmentation_every_500_epochs(model, device, val_loader, 4)

# Utils.display_segmentation_with_overlay(model, device, val_loader, 4)

