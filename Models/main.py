from config import get_config
from data_loader import get_dataloaders
from model import setup_model_and_optimizer
from train import train_model
import Utils
from early_stopping import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
import torch
import os
import time

def main():
    #Get params
    config = get_config()

    #Get loaders 
    train_loader, val_loader = get_dataloaders(config) 

    #Set to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Define model and optim 
    model, optimizer = setup_model_and_optimizer(config, device) 

    #Save to dir
    log_dir = config['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir) 
    best_model_path = os.path.join(config['model_dir'], config['exp_name']+ "_best_model.pth")

    #Set window size
    window_size = 20

    #Initialize early stopping
    early_stopping = EarlyStopping(patience=100, min_delta=0)

    #Train
    start_time = time.time()
    train_model(model, train_loader, val_loader, optimizer, device, config['num_epochs'], writer, early_stopping, best_model_path, window_size)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds.")

    #Write to tensorboardx
    writer.close()
    
    #Display 4 results from validation dataset
    Utils.display_segmentation_with_contours(model, device, val_loader, 3, config['class_names'])

if __name__ == "__main__":
    main()