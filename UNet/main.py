from config import get_config, parse_args
from data_loader import get_dataloaders
from model import setup_model_and_optimizer
from train import train_model
import Utils
from early_stopping import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
import torch
import os
import time
import cProfile
import pstats


def main():
    #Get params
     
    profiling=True 
    args = parse_args()

    start_time = time.time()
    config = get_config()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to get config {elapsed_time:.2f} seconds.")
    #Get loaders 
    start_time = time.time()
    train_loader, val_loader = get_dataloaders(config) 
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to load data {elapsed_time:.2f} seconds.")
    #Set to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Define model and optim 
    start_time = time.time()
    model, optimizer = setup_model_and_optimizer(config, device) 
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to define model {elapsed_time:.2f} seconds.")
    #Save to dir
    log_dir = config['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir) 
    best_model_path = os.path.join(config['model_dir'], config['exp_name']+ "__best_model.pth")

    #Set window size
    window_size = 20

    #Initialize early stopping
    early_stopping = EarlyStopping(patience=1000, min_delta=0)

    #Train

    train_model(model, train_loader, val_loader, optimizer, device, int(config['num_epochs']), writer, early_stopping, best_model_path, window_size)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds.")
    
    #Write to tensorboardx
    writer.close()

if __name__ == "__main__":
    with cProfile.Profile() as profile:
        main()

    
    res = pstats.Stats(profile)
    res.strip_dirs().sort_stats('cumulative')
    res.print_stats()
    res.dump_stats('/home/phukon/Desktop/Model_Fitting/stats/results.prof')