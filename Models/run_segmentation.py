import numpy as np
import torch
from torch.utils.data import DataLoader
from diffusers import UNet2DModel
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt

from segmentation_UNet2D import SegmentationDataset  

# Load model from local storage
model = UNet2DModel.from_pretrained("/home/phukon/code_python/SegmentationModels/Models/local_unet_model")

# Define transformations
transform_image = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

transform_mask = transforms.Compose([
    transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
    transforms.ToTensor(),
])

# Path to your JSON file
json_file_path = '/home/phukon/Desktop/Annotation_VIA/Train/jsons/via_project_2Sep2024_16h21m_kickstart_json.json'
# Directory where images are stored
image_dir = '/home/phukon/Desktop/Annotation_VIA/Train/imgs_kickstart/'

# Create dataset and dataloader
dataset = SegmentationDataset(json_file=json_file_path, image_dir=image_dir, transform_mask=transform_mask, transform_image=transform_image)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Iterate through the dataset and display images and masks
for images, masks in dataloader:
    images_np = images.cpu().numpy()
    masks_np = masks.cpu().numpy()

    # Change shape from [batch_size, channels, height, width] to [batch_size, height, width, channels]
    images_np = np.transpose(images_np, (0, 2, 3, 1))  
    masks_np = masks_np.squeeze(1)

    # Display images and masks
    for i in range(images_np.shape[0]):
        plt.figure(figsize=(12, 6))
        print(f"Image Size: {images_np.shape}")  # Example: [2, 256, 256, 3]
        print(f"Mask Size: {masks_np.shape}")    # Example: [2, 256, 256]

        # Display the image
        plt.subplot(1, 2, 1)
        plt.title(f'Image {i}')
        print(f"Mean: {np.mean(images_np[i,:,:,0]):.2f}, Std: {np.std(images_np[i]):.2f}")
        plt.imshow(images_np[i])
        plt.axis('off')

        # Display the mask
        plt.subplot(1, 2, 2)
        plt.title(f'Mask {i}')
        plt.imshow(masks_np[i], cmap='gray')
        print(f"Max: {np.max(masks_np[i])}, Min: {np.min(masks_np[i])}")
        plt.axis('off')
        plt.show()