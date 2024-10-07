import os

def get_config():
    config=  {
        "learning_rate": 0.1,
        "num_epochs": 100,
        "batch_size": 16,
        "random_seed": 42,
        "optimizer_name": "SGD",
        "do_augmentation": True,
        "activation": "ReLU",
        "unet_depth": "4",
        "log_base_dir": '/home/phukon/Desktop/Model_Fitting/runs/',
        "image_dir": '/home/phukon/Desktop/Model_Fitting/weka_dataset/images/train_set/',
        "mask_path": '/home/phukon/Desktop/Model_Fitting/weka_dataset/masks/',
        "model_dir": '/home/phukon/Desktop/Model_Fitting/models/'
    }

# Generate experiment name based on configuration
    exp_name = (
        config['optimizer_name']
        + "_bs_" + str(config['batch_size'])
        + "__lr_" + str(config['learning_rate'])
        + "__epoc_" + str(config['num_epochs'])
        + "__optim_" + str(config['optimizer_name'])
        + "__unet_depth_" + str(config['unet_depth'])
        + "__augmentation_" + str(config['do_augmentation'])
        + "__activation_" + str(config['activation'])
        + "__EWMA_val_loss_test"
    )

    # Define the full log directory path based on the experiment name
    config['exp_name'] = exp_name
    config['log_dir'] = os.path.join(config['log_base_dir'], "training_custom_unet_with_skip_connections_" + exp_name)

    return config