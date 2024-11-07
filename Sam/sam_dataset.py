import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class SamDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = os.listdir(image_dir)
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # Load image and mask
        img_name = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_name = os.path.join(self.mask_dir, self.image_filenames[idx])

        image = Image.open(img_name).convert("RGB")  
        mask = Image.open(mask_name).convert("L") 

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask
