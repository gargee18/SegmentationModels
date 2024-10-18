import torch
import time
import numpy as np
import torch.nn.functional as F
from config import get_config
from sklearn.metrics import f1_score
import Utils 

def train(model, train_loader, optimizer, device, epoch, writer):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    config = get_config()
    for images, masks in train_loader:
        
        images = images.permute(0, 1, 2, 3).to(device)
        masks = masks.squeeze().to(device)

        optimizer.zero_grad()
        y_pred = model(images)
        
        masks_int = masks.long()
        train_class_weights_tensor = torch.tensor(Utils.get_weights(masks, device, int(config['num_classes'])), dtype=torch.float32).to(device)
        
        loss = F.cross_entropy(y_pred, masks_int, weight=train_class_weights_tensor, ignore_index=8)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        all_labels = masks_int.flatten().cpu().numpy()
        all_preds = torch.argmax(y_pred, dim=1).flatten().cpu().numpy()
    
    # print(f"all_labels shape: {all_labels.shape}, all_preds shape: {all_preds.shape}")
    running_loss /= len(train_loader)
    train_f1 = f1_score(all_labels, all_preds, labels=[0, 1, 2, 3, 4, 5, 6], average='weighted')
    train_f1_hf = f1_score(all_labels, all_preds, labels=[0],average='weighted')
    train_f1_necd = f1_score(all_labels, all_preds, labels=[3],average='weighted')

    writer.add_scalar('F1/train', train_f1, epoch)
    writer.add_scalar('F1/train/hf', train_f1_hf, epoch)
    writer.add_scalar('F1/train/necd', train_f1_necd, epoch)
    writer.add_scalar('Loss/train', running_loss / len(train_loader), epoch)
    
    return running_loss, train_f1

def validate(model, val_loader, device, epoch, writer, window_size, moving_avg_val_loss):
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_preds = []
    mask_true=[]
    mask_pred=[]
    config = get_config()
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.permute(0, 1, 2, 3).to(device)
            masks = masks.squeeze().to(device)
            
            y_pred = model(images)
            masks_int = masks.long()
            
            val_class_weights_tensor = torch.tensor(Utils.get_weights(masks, device, int(config['num_classes'])), dtype=torch.float32).to(device)
            loss = F.cross_entropy(y_pred, masks_int, weight = val_class_weights_tensor, ignore_index=7)
            val_loss += loss.item()
            
            all_labels = masks_int.flatten().cpu().numpy() #array
            mask_true.extend(all_labels)   #list
            all_preds = torch.argmax(y_pred, dim=1).flatten().cpu().numpy() #array
            mask_pred.extend(all_preds)    #list
    
    val_loss /= len(val_loader)
    moving_avg_val_loss = ((window_size - 1) / window_size) * moving_avg_val_loss + (1 / window_size) * val_loss
    val_f1 = f1_score(all_labels, all_preds, labels=[0, 1, 2, 3, 4, 5, 6], average='weighted')
    val_f1_hf=f1_score(all_labels, all_preds, labels=[0],average='weighted')
    val_f1_hnf = f1_score(all_labels, all_preds, labels=[1],average='weighted')
    val_f1_neci = f1_score(all_labels, all_preds, labels=[2],average='weighted')
    val_f1_necd = f1_score(all_labels, all_preds, labels=[3],average='weighted')
    val_f1_wr = f1_score(all_labels, all_preds, labels=[4],average='weighted')
    val_f1_pit = f1_score(all_labels, all_preds, labels=[6],average='weighted')


    writer.add_scalar('F1/val', val_f1, epoch)
    writer.add_scalar('F1/val/hf', val_f1_hf, epoch)
    writer.add_scalar('F1/val/hnf', val_f1_hnf, epoch)
    writer.add_scalar('F1/val/neci', val_f1_neci, epoch)
    writer.add_scalar('F1/val/necd', val_f1_necd, epoch)
    writer.add_scalar('F1/val/wr', val_f1_wr, epoch)
    writer.add_scalar('F1/val/pit', val_f1_pit, epoch)
    writer.add_scalar('Loss/val', moving_avg_val_loss, epoch)

    return val_loss, val_f1, moving_avg_val_loss, val_f1_hf, val_f1_hnf, val_f1_neci, val_f1_necd, val_f1_wr, val_f1_pit,  mask_true, mask_pred, images
def train_model(model, train_loader, val_loader, optimizer, device, num_epochs, writer, early_stopping, best_model_path, window_size=5):
   
    best_val_loss = float('inf')
    best_epoch = 0
    moving_avg_val_loss = 0.0
    # print(num_epochs, type(num_epochs))
    for epoch in range(num_epochs):
        # Train for one epoch
        train_loss, train_f1 = train(model, train_loader, optimizer, device, epoch, writer)

        # Validate for one epoch
        val_loss, val_f1, moving_avg_val_loss, val_f1_hf, val_f1_hnf, val_f1_neci, val_f1_necd, val_f1_wr, val_f1_pit, mask_true, mask_pred, images = validate(model, val_loader, device, epoch, writer, window_size, moving_avg_val_loss)
        
        
        # Initialize moving average if it's the first epoch
        if epoch == 0:
            moving_avg_val_loss = val_loss  # Initialize with first validation loss
        
        print(f"[{epoch + 1}/{num_epochs}] TRL: {train_loss:.4f} VALL: {val_loss:.4f} MAVL: {moving_avg_val_loss:.4f} TRF1: {train_f1:.4f} VALF1: {val_f1:.4f} VF1HF: {val_f1_hf:.4f} VF1HF: {val_f1_hnf:.4f} VF1NECI: {val_f1_neci:.4f} VF1NECD: {val_f1_necd:.4f} VF1WR: {val_f1_wr:.4f} VF1PIT: {val_f1_pit:.4f}")
        
        # Early stopping check
        early_stopping.check(val_loss)
        if early_stopping.should_stop:
            print(f"Early stopping at epoch {epoch+1} TRL {train_loss:.4f} VALL {val_loss:.4f} MAVL: {moving_avg_val_loss:.4f} TRF1: {train_f1:.4f} VALF1: {val_f1:.4f}") #Best validation loss: {early_stopping.best_val_loss:.4f}")
            break

        # Save best model based on validation moving average loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss  # Update best validation moving average loss
            best_epoch = epoch  # Update best epoch
            corresponding_train_loss = train_loss
            corresponding_train_F1 = train_f1
            best_val_F1 = val_f1
            corresponding_MAVL = moving_avg_val_loss
            torch.save(model.state_dict(), best_model_path)  # Save the model's state dict
            print(f"New best model saved at epoch {epoch + 1} with TR L {train_loss:.4f} VAL L {val_loss:.4f} MAVL: {moving_avg_val_loss:.4f} TRF1: {train_f1:.4f} VALF1: {val_f1:.4f}")
    print('-' * 130 )
    print(f"Training complete. Best model at epoch {best_epoch + 1} with TR Loss {corresponding_train_loss:.4f} Val Loss: {best_val_loss:.4f} MAVL: {corresponding_MAVL:.4f} TRF1: {corresponding_train_F1:.4f} VALF1: {best_val_F1:.4f}")
    print('-' * 130 )
    return 


