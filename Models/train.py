import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
import Utils  # Assuming Utils contains custom utility functions

def train_one_epoch(model, train_loader, optimizer, device, epoch, writer):
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

def validate_one_epoch(model, val_loader, device, epoch, writer, window_size, moving_avg_val_loss):
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

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs, writer, early_stopping, best_model_path, window_size=5):
    best_val_loss = float('inf')
    best_epoch = 0
    moving_avg_val_loss = 0.0

    for epoch in range(num_epochs):
        # Train for one epoch
        train_loss, train_f1 = train_one_epoch(model, train_loader, optimizer, device, epoch, writer)
        
        # Validate for one epoch
        val_loss, val_f1, moving_avg_val_loss, val_f1_hf, val_f1_necd = validate_one_epoch(model, val_loader, device, epoch, writer, window_size, moving_avg_val_loss)
        
        # Initialize moving average if it's the first epoch
        if epoch == 0:
            moving_avg_val_loss = val_loss  # Initialize with first validation loss
        
        print(f"[{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Moving Avg Val Loss: {moving_avg_val_loss:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, Val F1 HF: {val_f1_hf:.4f}, Val F1 NECD: {val_f1_necd:.4f}")
        
        # Early stopping check
        early_stopping.check(moving_avg_val_loss)
        if early_stopping.should_stop:
            print(f"Early stopping at epoch {epoch}. Best validation loss: {early_stopping.best_val_loss:.4f}")
            break

        # Save best model based on validation moving average loss
        if moving_avg_val_loss < best_val_loss:
            best_val_loss = moving_avg_val_loss  # Update best validation moving average loss
            best_epoch = epoch  # Update best epoch
            torch.save(model.state_dict(), best_model_path)  # Save the model's state dict
            print(f"New best model saved at epoch {epoch + 1} with train loss {train_loss:.4f}, val loss {val_loss:.4f}, moving avg val loss: {moving_avg_val_loss:.4f}")

    print(f"Training complete. Best model found at epoch {best_epoch + 1} with val loss: {best_val_loss:.4f}")
