import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from segmentation_dataset import SegmentationDataset 
from custom_unet_with_skip import CustomUnetWithSkip
from image_mask_dataset import ImageMaskDataset 
from early_stopping import EarlyStopping
import Utils  # Assuming Utils contains relevant utility functions


# Tune Hyperparameters (learning rate, epoch, batch size, seed, optimizer)
learning_rate=0.01
num_epochs = 10000
batch_size= 12
random_seed=42
optimizer ="SGD" 
do_augmentation=True
activation = "ReLU"

# Initialize SummaryWriter
unet_depth= "4"
naming=optimizer
exp_name=naming+"_bs_"+str(batch_size)+"__lr_"+str(learning_rate)+"__epoc_"+str(num_epochs)+"__optim_"+str(optimizer)+"__unet_depth_"+str(unet_depth)+"__augmentation_"+str(do_augmentation)+"__activation_"+str(activation)+"__EWMA_val_loss_"
log_dir='/home/phukon/Desktop/Model_Fitting/runs/training_custom_unet_with_skip_connections_'+exp_name

# Create directory
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

# Define paths for annotations and source image, and create the dataloader
# json_file_path = '/home/phukon/Desktop/Model_Fitting/annotations/train_annotations.json'
# image_dir = '/home/phukon/Desktop/Model_Fitting/images/train_set/'

mask_path = '/home/phukon/Desktop/Model_Fitting/weka_dataset/masks/'
image_dir = '/home/phukon/Desktop/Model_Fitting/weka_dataset/images/train_set/'
model_dir = '/home/phukon/Desktop/Model_Fitting/models/'
# dataset = SegmentationDataset(json_file=json_file_path, image_dir=image_dir, augment= do_augmentation)
dataset = ImageMaskDataset(image_dir=image_dir, mask_dir = mask_path, augment = do_augmentation)

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
# criterion = nn.CrossEntropyLoss()  # Multi-class segmentation task
criterion = F.cross_entropy()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Track the best epoch and minimum validation loss
best_val_loss = float('inf')  # Initialize with infinity
best_epoch = -1  # Track the epoch with minimum validation loss
best_model_path = os.path.join(model_dir, exp_name+"best_model.pth")  # Path to save the best model
window_size = 20  
moving_avg_val_loss = 0
early_stopping = EarlyStopping(patience=100, min_delta=0.01)

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
        loss = F.cross_entropy(y_pred, masks_int, weight=train_class_weights_tensor)#,  ignore_index=7 )
        # loss=criterion(y_pred,masks_int)
        loss.backward()
        optimizer.step()  # Update the weights
        running_loss += loss.item()
        all_labels = masks.flatten().cpu().numpy()  # Ensure the labels are flattened and moved to CPU
        all_preds = torch.argmax(y_pred, dim=1).flatten().cpu().numpy()  # Get predicted class indices and flatten
        # all_preds = torch.argmax(y_pred[:,0:7,:,:], dim=1).flatten().cpu().numpy()  # Get predicted class indices and flatten

    # Calculate training F1 score
    train_f1 = f1_score(all_labels, all_preds, labels=[0,1,2,3,4,5,6],average='weighted')
    train_f1_hf = f1_score(all_labels, all_preds, labels=[0,1],average='weighted')
    train_f1_necd = f1_score(all_labels, all_preds, labels=[2,3],average='weighted')
    writer.add_scalar('F1/train', train_f1, epoch)
    writer.add_scalar('F1/train/hf', train_f1_hf, epoch)
    writer.add_scalar('F1/train/necd', train_f1_necd, epoch)
    writer.add_scalar('Loss/train', running_loss/ len(train_loader), epoch)


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
            all_preds = torch.argmax(y_pred, dim=1).flatten().cpu().numpy()  # Get predicted class indices and flatten
            # all_preds = torch.argmax(y_pred[:,0:7,:,:], dim=1).flatten().cpu().numpy()  # Get predicted class indices and flatten
    
    val_loss = val_loss / len(val_loader)
    if epoch == 0:
        moving_avg_val_loss = val_loss  # Initialize the moving average with the first value
    else:
        moving_avg_val_loss = ((window_size - 1) / window_size) * moving_avg_val_loss + (1 / window_size) * val_loss

    
    # Calculate validation F1 score
    val_f1 = f1_score(all_labels, all_preds, labels=[0,1,2,3,4,5,6],average='weighted') #
    val_f1_hf=f1_score(all_labels, all_preds, labels=[0],average='weighted')
    val_f1_necd=f1_score(all_labels, all_preds, labels=[2,3],average='weighted')
    writer.add_scalar('F1/val', val_f1, epoch)
    writer.add_scalar('F1/val/hf', val_f1_hf, epoch)
    writer.add_scalar('F1/val/necd', val_f1_necd, epoch)
    writer.add_scalar('Loss/val', moving_avg_val_loss, epoch)

    print(f"[{epoch + 1}/{num_epochs}], TRLoss: {running_loss / len(train_loader):.4f}, ValLoss: {val_loss:.4f}, TrainF1: {train_f1:.4f}, ValF1: {val_f1:.4f}, ValF1hf: {val_f1_hf:.4f}, ValF1necd: {val_f1_necd:.4f}")

    #Early Stopping check
    early_stopping.check(moving_avg_val_loss) 
    if early_stopping.should_stop:
        print(f"Early stopping at epoch {epoch}. Best validation loss: {early_stopping.best_val_loss:.4f}")
        break

    if moving_avg_val_loss < best_val_loss:
            best_val_loss = moving_avg_val_loss  # Update the lowest validation moving average loss
            corresponding_train_loss = running_loss 
            best_epoch = epoch  # Update the best epoch
            torch.save(model.state_dict(), best_model_path)  # Save the model's state dict
            print(f"New best model saved at epoch {epoch + 1} with train loss {running_loss / len(train_loader):.4f}, val_loss {val_loss:.4f}, moving avg val_loss: {moving_avg_val_loss:.4f}")

    
    # if val_loss < best_val_loss:
    #     best_val_loss = val_loss  # Update the lowest validation loss
    #     corresponding_train_loss = running_loss 
    #     best_epoch = epoch  # Update the best epoch
    #     torch.save(model.state_dict(), best_model_path)  # Save the model's state dict
    #     print(f"New best model saved at epoch {epoch + 1} with train loss {running_loss/ len(train_loader):.4f} and val_loss {val_loss:.4f}")


print(f"Training complete. Best validation loss {best_val_loss:.4f} at epoch {best_epoch + 1}")
    

writer.close()


Utils.display_segmentation_with_overlay(model, device, val_loader, 4)

