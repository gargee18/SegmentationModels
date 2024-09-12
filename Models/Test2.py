
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

        # Create an empty mask with class values
        mask = Image.new('I', image.size, 0)  # 'I' mode for 32-bit integer pixels

        for region in regions:
            shape = region['shape_attributes']
            tissue_class = region['region_attributes'].get('Tissue Class', 'Background')
            class_index = self.class_to_index(tissue_class)

            if shape['name'] == 'polygon':
                points = list(zip(shape['all_points_x'], shape['all_points_y']))
                self.draw_polygon(mask, points, class_index)
            elif shape['name'] == 'rect':
                x, y, width, height = shape['x'], shape['y'], shape['width'], shape['height']
                points = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
                self.draw_polygon(mask, points, class_index)

        # Convert mask to numpy array for consistency with image transformations
        mask = np.array(mask, dtype=np.int32)

        if self.transform:
            image = self.transform(image)
            # Custom transformation to ensure mask is handled correctly
            mask = self.transform(Image.fromarray(mask))

        return image, torch.tensor(mask, dtype=torch.long)

    def draw_polygon(self, mask, points, class_index):
        # Draw a polygon on the mask using class index
        draw = ImageDraw.Draw(mask)
        draw.polygon(points, outline=class_index, fill=class_index)

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

class_names = {
    0: 'Background',
    1: 'Healthy Functional',
    2: 'Healthy NonFunctional',
    3: 'Necrotic Infected',
    4: 'Necrotic Dry',
    5: 'Bark'
}

colors = {
    0: 'black',
    1: 'green',
    2: 'blue',
    3: 'red',
    4: 'purple',
    5: 'orange'
}

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Path to your JSON file
json_file_path = '/home/phukon/Desktop/Annotation_VIA/Train/jsons/via_project_2Sep2024_16h21m_kickstart_json.json'
# Directory where images are stored
image_dir = '/home/phukon/Desktop/Annotation_VIA/Train/imgs_kickstart/'

# Create dataset and dataloader
dataset = SegmentationDataset(json_file=json_file_path, image_dir=image_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


for images, masks in dataloader:
    print(images.shape)  # Example: torch.Size([8, 3, 256, 256])
    print(masks.shape)   # Example: torch.Size([8, 256, 256])


    images_np = images.cpu().numpy()
    masks_np = masks.cpu().numpy()

    images_np = np.transpose(images_np, (0, 2, 3, 1))  # Change shape from [batch_size, channels, height, width] to [batch_size, height, width, channels]
    masks_np = masks_np.squeeze(1)
    
def plot_images_and_masks(images_np, masks_np):
    plt.figure(figsize=(15, 10))

    for i in range(images_np.shape[0]):
        plt.subplot(2, len(images_np), i + 1)
        plt.title(f'Image {i}')
        plt.imshow(images_np[i])
        plt.axis('off')

        plt.subplot(2, len(images_np), len(images_np) + i + 1)
        plt.title(f'Mask {i}')
        plt.imshow(masks_np[i], cmap='tab20')
        plt.axis('off')

        plt.plot([0], [0], color=colors[i], label=f'{i}: {class_names[i]}', marker='o', linestyle='')
              
       
        plt.axis('off')
        plt.show()

for images, masks in dataloader:
    images_np = images.cpu().numpy()
    masks_np = masks.cpu().numpy()
    images_np = np.transpose(images_np, (0, 2, 3, 1))
    masks_np = masks_np.squeeze(1)

    plot_images_and_masks(images_np, masks_np)
    break