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
from confusion_matrix import ConfusionMatrix

def main():
    #Get params
     
    profiling=True 
    args = parse_args()

    start_time = time.time()
    config = get_config(args)
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
    best_model_path = os.path.join(config['model_dir'], config['exp_name']+ "_best_model.pth")

    #Set window size
    window_size = 20

    #Initialize early stopping
    early_stopping = EarlyStopping(patience=1000, min_delta=0)

    #Train

    mask_true, mask_pred, images =train_model(model, train_loader, val_loader, optimizer, device, int(config['num_epochs']), writer, early_stopping, best_model_path, window_size)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds.")
    start_time = time.time()
    
  
    ConfusionMatrix(mask_true, mask_pred, config)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to draw confusion matrix {elapsed_time:.2f} seconds.")
    #Write to tensorboardx
    writer.close()
    # Utils.display_segmentation_with_nice_overlay(images, mask_true, mask_pred, 3, config['class_names'])
    #Display 4 results from validation dataset
    # Utils.display_segmentation_with_contours(images, mask_true, mask_pred, 3, config['class_names'])
    
    
    Utils.display_segmentation_with_errormap(images, mask_true, mask_pred, 3, config['class_names'])
    # Utils.display_segmentation_with_nice_overlay(model, device, val_loader, 3, config['class_names'])

if __name__ == "__main__":
    with cProfile.Profile() as profile:
        main()

    
    res = pstats.Stats(profile)
    res.strip_dirs().sort_stats('cumulative')
    res.print_stats()
    res.dump_stats('/home/phukon/Desktop/Model_Fitting/stats/results.prof')

# if __name__ == "__main__":
#     profile_output = io.StringIO()

#     # Start profiling
#     profiler = cProfile.Profile()
#     profiler.enable()
    
#     main()
    
#     profiler.disable()

#     # Analyze the results using pstats
#     stats = pstats.Stats(profiler, stream=profile_output)
#     stats.strip_dirs().sort_stats('cumulative')  # Sort by cumulative time

#     # Print the first 100 lines of profiling results
#     stats.print_stats(500)  # Display only the top 100 lines
#     print(profile_output.getvalue()) 