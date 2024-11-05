import os
from ultralytics import SAM

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# Load the original image
image_path = "/home/phukon/Desktop/Model_Fitting/multiyear_dataset/train_set/"
image_files = [f for f in os.listdir(image_path) if f.endswith('.tif')]
mask_save_directory = "/home/phukon/Desktop/Model_Fitting/multiyear_dataset/SAM_predicted_masks/"
# Load model
model = SAM("sam_b.pt")



for image_file in image_files:
    #Infer
    image = os.path.join(image_path, image_file)
    image = Image.open(image)
    results = model.predict(image)

    # print("Type of `results`:", type(results))
    # print("Contents of `results`:", results)

    # Access the masks object
    masks = results[0].masks

    mask_array = masks.data.cpu().numpy() 

    # Check the shape of the mask array
    print("Mask array shape:", mask_array.shape)
    
    # Plot each mask overlayed on the original image
    for i in range(mask_array.shape[0]):

        mask = mask_array[i]

        # Convert the mask to a PIL image (0-255 range for grayscale)
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))

        # Create a filename for the mask image
        mask_filename = f"{os.path.splitext(image_file)[0]}_mask_{i + 1}.png"
        mask_save_path = os.path.join(mask_save_directory, mask_filename)

        # Save the mask image
        mask_image.save(mask_save_path)

        print(f"Saved mask {i + 1} for {image_file} as {mask_filename}")

    num_masks = mask_array.shape[0]
    cols = 4
    rows = (num_masks + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(10, 5 * rows))
    axes = axes.flatten()

    # Plot each mask for visualization (optional)
    for i in range(num_masks):
        mask = mask_array[i]

        axes[i].imshow(image, cmap="gray")
        axes[i].imshow(mask, cmap="jet", alpha=0.5)
        axes[i].set_title(f"{image_file} - Mask {i + 1}")
        axes[i].axis('off')

    for j in range(num_masks, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()