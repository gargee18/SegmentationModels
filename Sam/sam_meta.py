# This script uses the Segment Anything Model (SAM) to automatically generate segmentation masks 
# for a given image. It loads the SAM model, processes the image, and saves the generated masks 
# as binary images (0 or 255) to a specified directory. 

# Steps:
# 1. Load the SAM model with a pre-trained checkpoint.
# 2. Load an input image.
# 3. Automatically generate segmentation masks using SAM.
# 4. Save the generated masks as binary images (one per mask).


from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

model_type = "default"
checkpoint = "/home/phukon/Desktop/Model_Fitting/SAM_test/sam_vit_h_4b8939.pth"

# Load model
sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device)

# Create automatic mask generator object
mask_generator = SamAutomaticMaskGenerator(sam)

# Load input image
image_path = "/home/phukon/Desktop/Model_Fitting/SAM_test/CEP_318_2023_XR_sl_719.png"
image = cv2.imread(image_path)

# Generate mask automatically
masks = mask_generator.generate(image)

# Create a directory to save masks if it doesn't exist
output_dir = "/home/phukon/Desktop/Model_Fitting/SAM_test/masks/"
os.makedirs(output_dir, exist_ok=True)

# Save each mask
for i, mask in enumerate(masks):
    # Create a mask image (same size as original)
    mask_image = np.zeros(image.shape, dtype=np.uint8)

    # Fill the mask image with the mask's segmentation
    mask_image[mask['segmentation']] = (255, 255, 255)  # White mask

    # Save the mask image
    # mask_filename = os.path.join(output_dir, f"mask_{i+1}.png")
    # cv2.imwrite(mask_filename, mask_image)
    # print(f"Saved mask {i+1} as {mask_filename}")

# Create an overlay image
overlay_image = image.copy()

# Function to generate a random color
def random_color():
    return [random.randint(0, 255) for _ in range(3)]

# Overlay each mask with a different color
for mask in masks:
    color = random_color()  # Get a random color
    overlay_image[mask['segmentation']] = color  # Apply color to the mask region

# Save the overlay image with different colors for each mask
overlay_filename = os.path.join(output_dir, "overlay_image.png")
cv2.imwrite(overlay_filename, overlay_image)
print(f"Saved overlay image as {overlay_filename}")

# Display the overlay image (optional)
cv2.imshow("Overlay Image", overlay_image)














































# # Visualize the results
# plt.figure(figsize=(10, 10))
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
# plt.axis('off')

# # Overlay masks on the image
# for mask in masks:
#     # Convert mask to uint8 and create a color overlay
#     mask_overlay = np.zeros_like(image, dtype=np.uint8)
#     mask_overlay[mask['segmentation']] = (0, 255, 0)  # Green color for the mask
    
#     # Blend the original image with the mask overlay
#     alpha = 0.5  # Transparency factor
#     blended = cv2.addWeighted(image, 1 - alpha, mask_overlay, alpha, 0)

#     # Display the blended image
#     plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))  # Convert to RGB for display
#     plt.axis('off')
#     plt.title("Segmented Image with Masks")
    
# plt.show()