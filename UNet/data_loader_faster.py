from torch.utils.data import DataLoader 
from image_mask_dataset import ImageMaskDataset
import time

def get_dataloaders(config):
    train_dataset = ImageMaskDataset(
        image_dir = "/home/phukon/Desktop/Model_Fitting/weka_dataset/images/test/train/",
        mask_dir = "/home/phukon/Desktop/Model_Fitting/weka_dataset/images/test/train_mask/",
        augment=config['do_augmentation']
    )
    val_dataset = ImageMaskDataset(
        image_dir= "/home/phukon/Desktop/Model_Fitting/weka_dataset/images/test/validate/",
        mask_dir = "/home/phukon/Desktop/Model_Fitting/weka_dataset/images/test/validate_mask/",
        augment=config['do_augmentation']
    )
    # start_time = time.time()
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=16, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=16, prefetch_factor=2)
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"Time taken to load data {elapsed_time:.2f} seconds.")
    return train_loader, val_loader