from dataloader_2d_segmentation import SegmentationDataset  # Import the class from the class definition file
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

#BATCH SIZE FOR TESTING


#Define annotations and source image, and create the dataloader
json_file_path = '/home/phukon/Desktop/Model_Fitting/annotations/train_annotations.json'
image_dir = '/home/phukon/Desktop/Model_Fitting/images/train_set/'
dataset = SegmentationDataset(json_file=json_file_path, image_dir=image_dir, augment = False)
dataset_aug = SegmentationDataset(json_file=json_file_path, image_dir=image_dir, augment = True)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
dataloader_aug = DataLoader(dataset_aug,batch_size=4, shuffle=False)

for images, masks in dataloader:
    images_np = images.cpu().numpy()
    masks_np = masks.cpu().numpy()
    
    # Transpose the image array for correct plotting (from [batch_size, channels, height, width] to [batch_size, height, width, channels])
    images_np = np.transpose(images_np, (0, 2, 3, 1))
    
    # Squeeze the masks array if needed
    masks_np = masks_np.squeeze()

    # Number of images and masks to display
    num_display = min(4, images_np.shape[0])
    
    # Create a single figure with subplots for images and masks
    fig, axes = plt.subplots(2, num_display, figsize=(12, 6))
    
    # Plot the first 4 images in the first row
    for i in range(num_display):
        axes[0, i].imshow(images_np[i])
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Image {i}')
    
    # Plot the first 4 masks in the second row
    for i in range(num_display):
        axes[1, i].imshow(masks_np[i], cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Mask {i}')
    
    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.show()
    break  # Remove or adjust this line if you want to iterate over the dataloader

for images, masks  in dataloader_aug:
    images_np = images.cpu().numpy()
    masks_np = masks.cpu().numpy()
    print(f'Size of augmented images: {images_np.shape}')
    print(f'Size of augmented masks: {masks_np.shape}')
    # Transpose the image array for correct plotting (from [batch_size, channels, height, width] to [batch_size, height, width, channels])
    images_np = np.transpose(images_np, (0, 2, 3, 1))
    
    # Squeeze the masks array if needed
    masks_np = masks_np.squeeze()

    # Number of images and masks to display
    num_display = min(4, images_np.shape[0])
    
    # Create a single figure with subplots for images and masks
    fig, axes = plt.subplots(2, num_display, figsize=(12, 6))
    
    # Plot the first 4 images in the first row
    for i in range(num_display):
        axes[0, i].imshow(images_np[i])
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Image {i}')
    
    # Plot the first 4 masks in the second row
    for i in range(num_display):
        axes[1, i].imshow(masks_np[i], cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Mask {i}')
    
    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.show()
    break  # Remove or adjust this line if you want to iterate over the dataloader