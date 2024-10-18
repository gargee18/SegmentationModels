import os
from torch.utils.data import Dataset # provides basic structure for custom datasets
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import time
from torchvision.transforms import InterpolationMode
class ImageMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, augment=False):
        self.seed = torch.random.seed() 
        self.augment=augment
        if self.augment:
            self.transform_xray = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation = InterpolationMode.NEAREST, fill=40), # degree of distortion
                transforms.RandomRotation([0,360], fill=40),
                # transforms.ToTensor()
            ])
            self.transform_mask = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation = InterpolationMode.NEAREST, fill=7), # degree of distortion
                transforms.RandomRotation([0,360], fill=7),
                # transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])

        self.image_dir = image_dir
        self.mask_dir = mask_dir
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
        if  self.augment :
            self.seed += 1 # Store the random seed
            torch.manual_seed(self.seed)     # Ensure the same transformations
            image = self.transform_xray(image)#Image.fromarray(image)
            torch.manual_seed(self.seed)     
            mask = self.transform_mask(mask)

        
        return image, mask
