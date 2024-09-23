# This module act as a dataloader, an can be used for training any 2d semgentaiton model.
#It starts from an image directory and a json file, convert it into a mask with N=6 classes, and load it to the model


import os
import json
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_region_by_coords(image_filename):
    coordinates = {
        'CEP_313B_2024_sl_598.tif': (178, 132),
        'CEP_318_2022_sl_719.tif': (152, 119),
        'CEP_322_2023_sl_781.tif': (119, 142),
        'CEP_323_2024_sl_468.tif': (113, 141),
        'CEP_330_2022_sl_787.tif': (193, 184),
        'CEP_378A_2024_sl_1204.tif': (182, 110),
        'CEP_378B_2023_sl_1234.tif': (119, 122),
        'CEP_380A_2022_sl_1154.tif': (142, 124),
        'CEP_764B_2022_sl_924.tif': (146, 137),
        'CEP_988B_2024_sl_640.tif': (175, 115),
        'CEP_1181_2024_sl_409.tif': (110, 108),
        'CEP_1189_2023_sl_882.tif': (147, 149),
        'CEP_1193_2022_sl_632.tif': (160, 96),
        'CEP_1195_2023_sl_1080.tif': (135, 180),
        'CEP_1266A_2023_sl_671.tif': (131, 137),
        'CEP_2184A_2023_sl_537.tif': (94, 145)

    }

    # Return the coordinates if the filename is found, otherwise return a default value
    return coordinates.get(image_filename, (0, 0))



class SegmentationDataset(Dataset):

    # def __init__(self, json_file, image_dir, transform_mask=None, transform_image=None):
    #     # Load the JSON file
    #     if(transform_mask==None):
    #         transform_mask=transforms.Compose([
    #         transforms.ToTensor(),
    #         ])
    #         transform_image = transforms.Compose([
    #         transforms.ToTensor(),
    #         ])

    #     with open(json_file, 'r') as f:    # r for read
    #         self.data = json.load(f)
    #         for key in self.data:
    #             # Get the filename and change its extension
    #             old_filename = self.data[key]['filename']
    #             if old_filename.endswith('.jpg'):
    #                 new_filename = old_filename.replace('.jpg', '.tif')
    #                 self.data[key]['filename'] = new_filename
    #             self.image_dir = image_dir
    #             self.transform_image = transform_image
    #             self.transform_mask = transform_mask


    def __init__(self, json_file, image_dir, augment=False):
    # If augment is False, apply only ToTensor, otherwise apply augmentations
            if augment:
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation([0,360]),
                    transforms.ToTensor()
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToTensor()
                ])

            with open(json_file, 'r') as f:    # r for read
                self.data = json.load(f)
                for key in self.data:
                    # Get the filename and change its extension
                    old_filename = self.data[key]['filename']
                    if old_filename.endswith('.jpg'):
                        new_filename = old_filename.replace('.jpg', '.tif')
                        self.data[key]['filename'] = new_filename
                        self.image_dir = image_dir

            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the image information
        image_info = list(self.data.values())[idx]
        image_filename = image_info['filename']
        image_path = os.path.join(self.image_dir, image_filename)
        regions = image_info['regions']

        # Load image
        image = Image.open(image_path)

        # Create an empty mask with class values
        mask = Image.new('I', image.size, 7)  # 'I' mode for 32-bit integer pixels

        for region in regions:
            shape = region['shape_attributes']
            if 'Tissue Class' not in region['region_attributes']:
                raise ValueError(f"'Tissue Class' is missing in region: {region}")
            tissue_class = region['region_attributes']['Tissue Class']
            class_index, num_classes = self.class_to_index(tissue_class)

            if shape['name'] == 'polygon':
                points = list(zip(shape['all_points_x'], shape['all_points_y']))
                self.draw_polygon(mask, points, class_index)

            elif shape['name'] == 'rect':
                x, y, width, height = shape['x'], shape['y'], shape['width'], shape['height']
                points = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
                self.draw_polygon(mask, points, class_index)

        # Convert mask to numpy array for consistency with image transformations
        mask = np.array(mask, dtype=np.int32)
        x0,y0=extract_region_by_coords(image_filename)

        # Crop to 256x256
        mask=mask[x0:x0+256,y0:y0+256]
        image_np = np.array(image)
        image=image_np[x0:x0+256,y0:y0+256]   

        if image.dtype.byteorder not in ('=', '|'):  # '=' means native byte order, '|' means not applicable (for non-byte type)
            image = image.byteswap().newbyteorder()

        if  self.transform :
            seed = torch.random.seed()  # Store the random seed
            torch.manual_seed(seed)     # Ensure the same transformations
            image = self.transform(Image.fromarray(image))
            torch.manual_seed(seed)     # Ensure the same transformations
            mask = self.transform(Image.fromarray(mask)) 
    
        # return torch.tensor(image, dtype=torch.float32).to(device), torch.tensor(mask, dtype=torch.float32).to(device)
        return image.clone().detach().float().requires_grad_(True).to(device), mask.clone().detach().float().to(device)
    


    def draw_polygon(self, mask, points, class_index):
        # Draw a polygon on the mask using class index
        draw = ImageDraw.Draw(mask)
        draw.polygon(points, outline=class_index, fill=class_index)

    def class_to_index(self, tissue_class):
        # Map tissue classes to integer values
        class_map = {
            'Background': 0,
            'Healthy Functional': 1,
            'Healthy Nonfunctional': 2,
            'Necrotic Infected': 3,
            'Necrotic Dry': 4,
            'Bark': 5,
            'White Rot': 6,
            'Unknown': 7,
        }
        # Handle unrecognized tissue classes
        if tissue_class not in class_map:
            print(f"Warning: Unrecognized Tissue Class '{tissue_class}'")
        num_classes = len(class_map)

        return class_map.get(tissue_class, 0),num_classes  # Default to Unknown if class not found
    
    

    











