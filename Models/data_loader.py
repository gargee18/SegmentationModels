from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from image_mask_dataset import ImageMaskDataset

def get_dataloaders(config):
    dataset = ImageMaskDataset(
        image_dir=config['image_dir'],
        mask_dir=config['mask_path'],
        augment=config['do_augmentation']
    )
    
    train_indices, val_indices = train_test_split(list(range(len(dataset))),test_size=0.2,shuffle=True,random_state=config['random_seed'])
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    return train_loader, val_loader