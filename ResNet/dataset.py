import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):

    def __init__(self, image_dir, mask_dir, img_transform, mask_transform):
        self.seed = torch.random.seed() 
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.image_filenames = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self,idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.image_filenames[idx]) 

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        image = np.array(image).astype(np.float32)
        mask = np.array(mask).astype(np.float32)

        image = (image - image.min()) / (image.max() - image.min())
        
        image = torch.tensor(image).permute(2, 0, 1)
        mask = torch.tensor(mask).unsqueeze(0)


        if self.img_transform:
            self.seed += 1
            torch.manual_seed(self.seed)
            image = self.img_transform(image)

        if self.mask_transform:
            self.seed += 1
            torch.manual_seed(self.seed)
            mask = self.mask_transform(mask)

        return image, mask
