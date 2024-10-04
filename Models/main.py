from config import get_config
from data_loader import get_dataloaders
from model import setup_model_and_optimizer
from train import train_one_epoch, validate_one_epoch
from Utils import save_checkpoint
from torch.utils.tensorboard import SummaryWriter
import torch
import os

def main():
    config = get_config()
    train_loader, val_loader = get_dataloaders(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, optimizer, criterion = setup_model_and_optimizer(config, device)
    
    exp_name = config['optimizer_name'] + f"_bs_{config['batch_size']}__lr_{config['learning_rate']}__epoc_{config['num_epochs']}"
    log_dir = os.path.join(config['log_dir'], exp_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    best_val_loss = float('inf')
    best_model_path = os.path.join(config['model_dir'], exp_name + "_best_model.pth")
    moving_avg_val_loss = 0
    window_size = 20
    
    for epoch in range(config['num_epochs']):
        train_loss, train_f1 = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, writer)
        val_loss, val_f1, moving_avg_val_loss,val_f1_hf, val_f1_necd = validate_one_epoch(model, val_loader, criterion, device, epoch, writer, window_size, moving_avg_val_loss)
        
        best_val_loss = save_checkpoint(model, best_model_path, epoch, train_loss, val_loss, moving_avg_val_loss, best_val_loss)
        
        print(f"Epoch [{epoch + 1}/{config['num_epochs']}], TrainLoss: {train_loss:.4f}, ValLoss: {val_loss:.4f}, TrainF1: {train_f1:.4f}, ValF1: {val_f1:.4f} , ValF1hf: {val_f1_hf:.4f}, ValF1necd: {val_f1_necd:.4f}")
        # print(f"[{epoch + 1}/{num_epochs}], TRLoss: {running_loss / len(train_loader):.4f}, ValLoss: {val_loss:.4f}, TrainF1: {train_f1:.4f}, ValF1: {val_f1:.4f}, ValF1hf: {val_f1_hf:.4f}, ValF1necd: {val_f1_necd:.4f}")


    #Early Stopping check
    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
    writer.close()

if __name__ == "__main__":
    main()