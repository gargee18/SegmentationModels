# Trial of inference of a diffuser model from hugging face
import os
from diffusers import UNet2DModel
import json
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image , ImageDraw
import matplotlib.pyplot as plt
# Load model "https://huggingface.co/CompVis/ldm-celebahq-256/blob/main/unet/config.json"
#model = UNet2DModel.from_pretrained("CompVis/ldm-celebahq-256",subfolder="unet")
# Save the model locally
#model.save_pretrained("/home/phukon/code_python/SegmentationModels/Models/local_unet_model")

# Using locally saved pretrained model
model = UNet2DModel.from_pretrained("/home/phukon/code_python/SegmentationModels/Models/local_unet_model")

class SegmentationDataset(Dataset):
    def __init__(self, json_file, image_dir, transform=None):
        # Load the JSON file
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.image_dir = image_dir  # Directory where images are stored
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the image information
        image_info = list(self.data.values())[idx]
        image_filename = image_info['filename']
        image_path = os.path.join(self.image_dir, image_filename)  # Combine directory and filename
        regions = image_info['regions']
        
        # Load image
        image = Image.open(image_path).convert('RGB')

        # Create an empty mask
        mask = Image.new('L', image.size, 0)  # 'L' mode for 8-bit pixels, black and white
        
        draw = ImageDraw.Draw(mask)

        # Draw each region on the mask
        for region in regions:
            shape = region['shape_attributes']
            tissue_class = region['region_attributes'].get('Tissue Class', 'Background')

            if shape['name'] == 'polygon':
                points = list(zip(shape['all_points_x'], shape['all_points_y']))
                draw.polygon(points, outline=1, fill=self.class_to_index(tissue_class))
            elif shape['name'] == 'rect':
                x, y, width, height = shape['x'], shape['y'], shape['width'], shape['height']
                points = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
                draw.polygon(points, outline=1, fill=self.class_to_index(tissue_class))

        # Convert mask to numpy array for consistency with image transformations
        mask = np.array(mask)

        if self.transform:
            image = self.transform(image)
            # Custom transformation to ensure mask is handled correctly
            mask = self.transform(Image.fromarray(mask))

        return image, torch.tensor(mask, dtype=torch.long)  # Ensure mask is in tensor format

    def class_to_index(self, tissue_class):
        # Map tissue classes to integer values
        class_map = {
            'Background': 0,
            'Healthy Functional': 1,
            'Healthy NonFunctional': 2,
            'Necrotic Infected': 3,
            'Necrotic Dry': 4,
            'Bark': 5
        }
        return class_map.get(tissue_class, 0)  # Default to Background if class not found

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Path to your JSON file
json_file_path = '/home/phukon/Desktop/Annotation_VIA/Train/via_project_2Sep2024_16h21m_json.json'
# Directory where images are stored
image_dir = '/home/phukon/Desktop/Annotation_VIA/Train/'

# Create dataset and dataloader
dataset = SegmentationDataset(json_file=json_file_path, image_dir=image_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)



for images, masks in dataloader:
    print(images.shape)  # Example: torch.Size([8, 3, 256, 256])
    print(masks.shape)   # Example: torch.Size([8, 256, 256])
    break  # Just to see the output of one batch


