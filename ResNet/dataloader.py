
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from dataset import MyDataset

def get_dataloader(config):
    #Define transformations 
    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])

    #Load dataset
    dataset = MyDataset(config['image_dir'], config['mask_path'], img_transform, mask_transform)

    #Split dataset (train:val = 80:20)
    train_length = int(len(dataset)*0.8)
    val_length = len(dataset) - train_length
    train_dataset, val_dataset = random_split(dataset, [train_length, val_length])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    return train_loader, val_loader









