import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
import Utils  # Assuming Utils contains custom utility functions

def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, writer):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    for images, masks in train_loader:
        images = images.permute(0, 1, 2, 3)
        images = images.to(device)
        masks = masks.squeeze()
        masks = masks.to(device)

        optimizer.zero_grad()
        y_pred = model(images)
        
        masks_int = masks.long()
        train_class_weights_tensor = torch.tensor(Utils.get_weights(train_loader, device), dtype=torch.float32).to(device)
        
        loss = F.cross_entropy(y_pred, masks_int, weight=train_class_weights_tensor)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        all_labels = masks.flatten().cpu().numpy()
        all_preds = torch.argmax(y_pred, dim=1).flatten().cpu().numpy()
    
    train_f1 = f1_score(all_labels, all_preds, labels=[0, 1, 2, 3, 4, 5, 6], average='weighted')
    train_f1_hf = f1_score(all_labels, all_preds, labels=[0,1],average='weighted')
    train_f1_necd = f1_score(all_labels, all_preds, labels=[2,3],average='weighted')
    writer.add_scalar('F1/train', train_f1, epoch)
    writer.add_scalar('F1/train/hf', train_f1_hf, epoch)
    writer.add_scalar('F1/train/necd', train_f1_necd, epoch)
    writer.add_scalar('Loss/train', running_loss / len(train_loader), epoch)
    
    return running_loss / len(train_loader), train_f1

def validate_one_epoch(model, val_loader, criterion, device, epoch, writer, window_size, moving_avg_val_loss):
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.permute(0, 1, 2, 3).to(device)
            masks = masks.squeeze().to(device)
            
            y_pred = model(images)
            masks_int = masks.long()
            
            val_class_weights_tensor = torch.tensor(Utils.get_weights(val_loader, device), dtype=torch.float32).to(device)
            loss = F.cross_entropy(y_pred, masks_int, weight=val_class_weights_tensor, ignore_index=7)
            val_loss += loss.item()
            
            all_labels = masks.flatten().cpu().numpy()
            all_preds = torch.argmax(y_pred, dim=1).flatten().cpu().numpy()
    
    val_loss /= len(val_loader)
    moving_avg_val_loss = ((window_size - 1) / window_size) * moving_avg_val_loss + (1 / window_size) * val_loss
    val_f1 = f1_score(all_labels, all_preds, labels=[0, 1, 2, 3, 4, 5, 6], average='weighted')
    val_f1_hf=f1_score(all_labels, all_preds, labels=[0],average='weighted')
    val_f1_necd=f1_score(all_labels, all_preds, labels=[2,3],average='weighted')
    writer.add_scalar('F1/val', val_f1, epoch)
    writer.add_scalar('F1/val/hf', val_f1_hf, epoch)
    writer.add_scalar('F1/val/necd', val_f1_necd, epoch)
    writer.add_scalar('Loss/val', moving_avg_val_loss, epoch)
    
    return val_loss, val_f1, moving_avg_val_loss, val_f1_hf, val_f1_necd
