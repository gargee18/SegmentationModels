
from dataloader import get_dataloader
from UNet.config import get_config
import numpy as np
import matplotlib.pyplot as plt

train_loader, val_loader= get_dataloader(get_config())

for images, masks in train_loader:

    images_np = images.cpu().numpy().squeeze()
    masks_np = masks.cpu().numpy().squeeze()

for images, masks in val_loader:

    images_np = images.cpu().numpy().squeeze()
    masks_np = masks.cpu().numpy().squeeze()
    # Check the shape of images and masks
    print("Images shape:", images_np.shape)
    print("Masks shape:", masks_np.shape)

    images_np = np.transpose(images_np, (0,2,3,1))
    masks_np = np.transpose(masks_np, (0,1,2))
    # Number of images and masks to display
    num_display = min(4, images_np.shape[0])
    
    # Create a single figure with subplots for images and masks
    fig, axes = plt.subplots(2, num_display, figsize=(12, 6))
    
    # Normalize the images to [0, 1] if they are not in that range
    for i in range(num_display):

        image = images_np[i]
        
        axes[0, i].imshow(image)  
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Image {i}')

        # Ensure masks are displayed correctly
        mask = masks_np[i]
        
        axes[1, i].imshow(mask, cmap='viridis')  # Use viridis or gray for binary masks
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Mask {i}')
        
    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.show()
    break  