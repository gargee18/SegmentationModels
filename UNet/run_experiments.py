from config import get_config, generate_exp_name
from data_loader import get_dataloaders
from model import setup_model_and_optimizer
from train import train_model
import Utils

from early_stopping import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
import torch
import os
import time
import multiprocessing

def run_training(config, device_id):
     
    #Get loaders
    train_loader, val_loader = get_dataloaders(config)

    #Set to device
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    #Define model and optimizer
    model, optimizer = setup_model_and_optimizer(config, device)

    #Save to dir
    log_dir = config['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir) 
    best_model_path = os.path.join(config['model_dir'], config['exp_name']+ "_best_model.pth")

    #Set window size
    window_size = 20

    #Initialize early stopping
    early_stopping = EarlyStopping(patience =200, min_delta=0)

    #Train
    start_time= time.time()
    train_model(model, config, train_loader, val_loader, optimizer, device, config['num_epochs'], writer, early_stopping, best_model_path, window_size)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds for config: {config['exp_name']}.")

    # Write to tensorboard
    writer.close()
    
    # Display results from validation dataset
    Utils.display_segmentation_with_contours(model, device, val_loader, 3, config['class_names'])
    Utils.display_segmentation_with_errormap(model, device, val_loader, 3, config['class_names'])
    Utils.display_segmentation_with_nice_overlay(model, device, val_loader, 3, config['class_names'])

    
if __name__ == "__main__":
    #Define two configs to send simultaneously 
    config1 = get_config()
    config2 = get_config()

    #Modify params for the second run
    config2['learning_rate']=0.01
    config2['num_epochs']=50

    generate_exp_name(config2)
    # Determine the number of available GPUs
    num_gpus = torch.cuda.device_count()

    #Create separate jobs/processes for both configurations
    job1 = multiprocessing.Process(target=run_training, args=(config1,0))
    job2 = multiprocessing.Process(target=run_training, args=(config2, 1 if num_gpus > 1 else 0))

    #Start both jobs
    job1.start()
    job2.start()

    #Wait for them to finish
    job1.join()
    job2.join()

    print('Both jobs done, stop crying!')

