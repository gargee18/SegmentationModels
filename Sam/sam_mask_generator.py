import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from PIL import Image


# image = cv2.imread('/home/phukon/Desktop/Model_Fitting/SAM_test/CEP_318_2023_XR_sl_719.tif')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# sam_checkpoint = "/home/phukon/Desktop/Model_Fitting/SAM_test/sam_vit_h_4b8939.pth"
# model_type = "vit_h"

# device = "cuda"

# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device=device)

# mask_generator = SamAutomaticMaskGenerator(sam)

# masks = mask_generator.generate(image)

# output_mask_dir = "/home/phukon/Desktop/Model_Fitting/SAM_test/masks/"

# def save_color_mask_overlay(img):
#     # Convert the mask overlay to a PIL image
#     mask_overlay_img = Image.fromarray((img * 255).astype(np.uint8))  # Convert to 8-bit RGB for saving
#     output_path = "/home/phukon/Desktop/Model_Fitting/SAM_test/generated_masks/CEP_318_2023_XR_sl_719_overlay_image.tif"
#     mask_overlay_img.save(output_path)
#     print(f"Color mask overlay saved to {output_path}")
# def show_anns(anns):
#     if len(anns) == 0:
#         return
#     sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
#     ax = plt.gca()
#     ax.set_autoscale_on(False)

#     img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
#     img[:,:,3] = 0
#     for ann in sorted_anns:
#         m = ann['segmentation']
#         color_mask = np.concatenate([np.random.random(3), [1]])
#         img[m] = color_mask
#     ax.imshow(img)
#     save_color_mask_overlay(img)


# plt.figure(figsize=(10,10))
# plt.imshow(image)
# show_anns(masks)
# plt.axis('off')
# plt.show() 





# Function to load and process an image
def load_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Function to save the color mask overlay
def save_color_mask_overlay(img, output_path):
    # Convert the mask overlay to a PIL image
    mask_overlay_img = Image.fromarray((img * 255).astype(np.uint8))  # Convert to 8-bit RGB for saving
    mask_overlay_img.save(output_path)
    print(f"Color mask overlay saved to {output_path}")

# Function to show annotations and save the overlay
def show_anns(anns, output_overlay_path):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0  # Transparent background

    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [1]])  # Random color with full opacity
        img[m] = color_mask  # Apply the color to the mask area

    ax.imshow(img)
    save_color_mask_overlay(img, output_overlay_path)

# Function to generate and save masks for all images in the input directory
def process_images_in_directory(input_dir, output_dir):
    # Get all image file paths in the directory
    image_paths = [os.path.join(input_dir, fname) for fname in os.listdir(input_dir) if fname.lower().endswith(('.tif', '.jpg', '.png', '.jpeg'))]
    
    for image_path in image_paths:
        # Get the base name of the image file (without the directory path)
        image_name = os.path.basename(image_path)
        
        # Define the output file path (same name but with '_overlay_image' suffix)
        output_overlay_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_overlay_image.tif")
        
        # Load and preprocess the image
        image = load_image(image_path)

        # Generate masks using the SAM model
        masks = mask_generator.generate(image)

        # Show annotations and save the color mask overlay
        show_anns(masks, output_overlay_path)

# Define input and output directories
input_dir = "/home/phukon/Desktop/Model_Fitting/SAM_test/dataset_to_generate_masks/"  # Directory containing your images
output_dir = "/home/phukon/Desktop/Model_Fitting/SAM_test/generated_masks/"  # Directory to save overlay images

# Initialize the SAM model
sam_checkpoint = "/home/phukon/Desktop/Model_Fitting/SAM_test/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# Create the mask generator
mask_generator = SamAutomaticMaskGenerator(sam)

# Process and save masks for all images in the input directory
process_images_in_directory(input_dir, output_dir)