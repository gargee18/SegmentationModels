import os
import argparse
def get_config(args=None):
    config=  {
        "learning_rate": 0.05,
        "num_epochs": 500,
        "batch_size": 16,
        "random_seed": 42,
        "optimizer_name": "SGD",
        "do_augmentation": True,
        "activation": "ReLU",
        "unet_depth": "3",
        "num_classes" : 8,
        "log_base_dir": '/home/phukon/Desktop/Model_Fitting/runs/',
        "image_dir": '/home/phukon/Desktop/Model_Fitting/weka_dataset/images/train_set/',
        "mask_path": '/home/phukon/Desktop/Model_Fitting/weka_dataset/masks/',
        "model_dir": '/home/phukon/Desktop/Model_Fitting/models/',
        "class_names": [
            "Healthy Functional",    # 0
            "Healthy Nonfunctional", # 1
            "Necrotic Infected",     # 2
            "Necrotic Dry",          # 3
            "White Rot",             # 4
            "Bark",                  # 5
            "Pith",                  # 6
            "Background",            # 7
        ]
    }
    if args:
        # Override specific arguments from the command line
        if args.learning_rate:
            config['learning_rate'] = args.learning_rate
        if args.num_epochs:
            config['num_epochs'] = args.num_epochs
        if args.batch_size:
            config['batch_size'] = args.batch_size
        if args.optimizer_name:
            config['optimizer_name'] = args.optimizer_name
        if args.do_augmentation is not None:
            config['do_augmentation'] = args.do_augmentation
        if args.activation:
            config['activation'] = args.activation
        if args.unet_depth:
            config['unet_depth'] = args.unet_depth

    generate_exp_name(config)  # Automatically generate the experiment name on config creation
    return config

# Function to generate the experiment name
def generate_exp_name(config):
    exp_name = (
        config['optimizer_name']
        + "_bs_" + str(config['batch_size'])
        + "__lr_" + str(config['learning_rate'])
        + "__epoc_" + str(config['num_epochs'])
        + "__optim_" + str(config['optimizer_name'])
        + "__unet_depth_" + str(config['unet_depth'])
        + "__augmentation_" + str(config['do_augmentation'])
        + "__activation_" + str(config['activation'])
        + "__EWMA_val_loss"
    )

    # Define the full log directory path based on the experiment name
    config['exp_name'] = exp_name
    config['log_dir'] = os.path.join(config['log_base_dir'], "training_custom_unet_" + exp_name)

def parse_args():
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Run segmentation model training with configurable parameters")
    
    # Add arguments for configuration overrides
    parser.add_argument('--learning_rate', type=float, help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--optimizer_name', type=str, help='Name of the optimizer to use')
    parser.add_argument('--do_augmentation', type=bool, help='Flag to enable/disable data augmentation')
    parser.add_argument('--activation', type=str, help='Activation function to use')
    parser.add_argument('--unet_depth', type=str, help='Depth of the U-Net model')

    return parser.parse_args()