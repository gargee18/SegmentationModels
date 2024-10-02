
import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

class ImageMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = os.listdir(image_dir)
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # Get image and mask paths
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.image_filenames[idx]) 
        
        # Load images and masks
        
        image = Image.open(img_path)
        mask = Image.open(mask_path)
    
        image = torch.tensor(np.array(image).astype(np.float32)).unsqueeze(0)
        mask = torch.tensor(np.array(mask).astype(np.float32)).unsqueeze(0)
        
        # Apply transformations if provided
        if self.transform and False:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask


