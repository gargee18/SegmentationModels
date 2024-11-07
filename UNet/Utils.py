import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import cv2
from skimage import measure
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
from PIL import Image

from config import get_config
#DEBUG : uncomment to test
#test_class_weights_tensor=torch.tensor([1,1,1,1,1,1,1,1], dtype=torch.float32).to(device)

def save_checkpoint(model, best_model_path, epoch, train_loss, val_loss, moving_avg_val_loss, best_val_loss):
    if moving_avg_val_loss < best_val_loss:
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved new best model at epoch {epoch + 1}")
        return moving_avg_val_loss
    return best_val_loss

def compute_class_frequencies(mask):
    mask = mask.squeeze().cpu().flatten()
    class_frequencies = np.bincount(mask) 
    # print(f"class_frequencies= {class_frequencies}")
    return class_frequencies

def compute_and_print_weights(class_frequencies):
    # Avoid division by zero
    valid_frequencies = np.where(class_frequencies > 0, class_frequencies, 1)  # Replace 0 with 1 to avoid division by zero
    weights = np.sqrt(1 / valid_frequencies)  # Compute weights using inverse of frequencies
    weights /= weights.sum()  # Normalize weights
    return weights

def get_weights(masks, device, num_classes):
    total_class_frequencies = np.zeros(num_classes)
    # for images, masks in loader:
    masks= masks.squeeze().to(device)
    class_frequencies = compute_class_frequencies(masks)
    total_elements = np.sum(class_frequencies)
    total_class_frequencies += class_frequencies
    class_weights = compute_and_print_weights(class_frequencies)
    

    # # Print frequencies for each class
    # for class_index, frequency in enumerate(class_frequencies):
    #     percentage = (frequency / total_elements) * 100 if total_elements > 0 else 0
    #     print(f"Class {class_index}: {frequency} occurrences ({percentage:.2f}%)")

    # for class_index, weight in enumerate(class_weights):
    #     print(f"Class {class_index}: Weight {weight:.4f}")

    return class_weights

def display_segmentation_with_errormap( images, true_labels, pred_labels, nb_images_to_display, class_names):

    cmap = plt.get_cmap('viridis')
    bounds = np.arange(len(class_names) + 1)  # Boundaries between classes (0, 1, 2, ..., 8)
    norm = BoundaryNorm(bounds, cmap.N)  # Ensures fixed color per class index
    columns = 5
    nb_images_displayed = 0  # Keep track of how many images have been displayed
    config = get_config()
    image_height, image_width = 256, 256
    batch_size = images.shape[0]  # Get the current batch size
    images_to_display = min(batch_size, nb_images_to_display - nb_images_displayed)  # Number of images to display in this batch

    # Create a figure and axes for the images, 4 columns for each: original, actual mask, predicted mask, overlay
    fig, axes = plt.subplots(images_to_display, columns, figsize=(20, images_to_display * 5))

    # Ensure that axes is always 2D, even if there's only one row
    axes = np.atleast_2d(axes)

    for i in range(images_to_display):
        img = images[i].detach().cpu().numpy().transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)

        # Convert the RGB image to grayscale (average of R, G, B channels)
        grayscale_img = np.mean(img, axis=-1)
        grayscale_img_8bit = np.interp(grayscale_img, (grayscale_img.min(), grayscale_img.max()), (0, 255)).astype(np.uint8)

        # Reshape true and predicted masks from lists back to 2D images
        actual_mask = np.array(true_labels[i * (image_height * image_width): (i + 1) * (image_height * image_width)]).reshape(image_height, image_width)
        predicted_mask = np.array(pred_labels[i * (image_height * image_width): (i + 1) * (image_height * image_width)]).reshape(image_height, image_width)
       
        # Display the original image (converted to grayscale)
        axes[i, 0].imshow(grayscale_img_8bit, cmap='gray')
        axes[i, 0].set_title(f"Original Grayscale Image {nb_images_displayed + i + 1}")
        axes[i, 0].axis('off')

        # Display the error on original image 
        error = (actual_mask != predicted_mask).astype(int)
        modified_grayscale_img = grayscale_img_8bit.copy()
        modified_grayscale_img[error > 0] = 255  # Set non-zero locations to 255
        axes[i, 1].imshow(modified_grayscale_img, alpha=1, cmap='gray')  # Display grayscale image
        axes[i, 1].imshow(error, alpha=0.5,cmap='Reds', vmin=0, vmax=0.1)  # Overlay colorful mask
        axes[i, 1].set_title(f"Overlay {nb_images_displayed + i + 1}")
        axes[i, 1].axis('off')

        # Display the actual segmentation mask
        axes[i, 2].imshow(actual_mask, cmap=cmap, norm=norm)
        axes[i, 2].set_title(f"Expected Mask {nb_images_displayed + i + 1}")
        axes[i, 2].axis('off')

        # Display the predicted segmentation mask
        axes[i, 3].imshow(predicted_mask, cmap=cmap, norm=norm)
        axes[i, 3].set_title(f"Predicted Mask {nb_images_displayed + i + 1}")
        axes[i, 3].axis('off')

        # Difference of the actual and pedicted mask
        axes[i, 4].imshow(error, cmap='gray', vmin=0, vmax=0.1)
        axes[i, 4].set_title(f'Error Map {i + 1}')
        axes[i, 4].axis('off')
        plt.tight_layout()

        nb_images_displayed += images_to_display
    plt.savefig(f'/home/phukon/Desktop/Model_Fitting/predicted_masks/{config['exp_name']}_with_ERRORMAP_{nb_images_displayed + i + 1}.png', bbox_inches='tight')
    plt.show() #This line generate a bug when run in ssh on phenodrone : 
    

        # nb_images_displayed += images_to_display  # Update the count of displayed images

        # if images_displayed >= num_images_to_display:
        #     break  # Stop if we have displayed the requested number of images
            #libGL error: MESA-LOADER: failed to open swrast: /usr/lib/dri/swrast_dri.so: 
            #Ne peut ouvrir le fichier d'objet partagé: Aucun fichier ou dossier de ce nom (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
            #libGL error: failed to load driver: swrast

def display_segmentation_with_contours(images, true_labels, pred_labels, nb_images_to_display, class_names):

    cmap = plt.get_cmap('tab10')
    bounds = np.arange(len(class_names) + 1)  # Boundaries between classes (0, 1, 2, ..., 8)
    norm = BoundaryNorm(bounds, cmap.N)  # Ensures fixed color per class index
    columns = 5
    nb_images_displayed = 0  # Keep track of how many images have been displayed
    config = get_config()
    # Define RGB colors for each label (8 classes)
    RGBforLabel = {
        0: (255, 0, 0),    # Class 0 - Red
        1: (0, 255, 0),    # Class 1 - Green
        2: (0, 0, 255),    # Class 2 - Blue
        3: (255, 255, 0),  # Class 3 - Yellow
        4: (0, 255, 255),  # Class 4 - Cyan
        5: (255, 0, 255),  # Class 5 - Magenta
        6: (128, 0, 128),  # Class 6 - Purple
        7: (255, 165, 0)   # Class 7 - Orange
    }

    image_height, image_width = 256, 256
    batch_size = images.shape[0]  # Get the current batch size
    images_to_display = min(batch_size, nb_images_to_display - nb_images_displayed)  # Number of images to display in this batch

    # Create a figure and axes for the images, 4 columns for each: original, actual mask, predicted mask, overlay
    fig, axes = plt.subplots(images_to_display, columns, figsize=(20, images_to_display * 5))

    # Ensure that axes is always 2D, even if there's only one row
    axes = np.atleast_2d(axes)

    for i in range(images_to_display):
        img = images[i].detach().cpu().numpy().transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)

        # Convert the RGB image to grayscale (average of R, G, B channels)
        grayscale_img = np.mean(img, axis=-1)
        grayscale_img_8bit = np.interp(grayscale_img, (grayscale_img.min(), grayscale_img.max()), (0, 255)).astype(np.uint8)

         # Reshape true and predicted masks from lists back to 2D images
        actual_mask = np.array(true_labels[i * (image_height * image_width): (i + 1) * (image_height * image_width)]).reshape(image_height, image_width)
        predicted_mask = np.array(pred_labels[i * (image_height * image_width): (i + 1) * (image_height * image_width)]).reshape(image_height, image_width)

        # Display the original image (converted to grayscale)
        axes[i, 0].imshow(grayscale_img_8bit, cmap='gray')
        axes[i, 0].set_title(f"Original Grayscale Image {nb_images_displayed + i + 1}")
        axes[i, 0].axis('off')

        # Display the actual segmentation mask
        axes[i, 1].imshow(actual_mask, cmap=cmap, norm=norm)
        axes[i, 1].set_title(f"Expected Mask {nb_images_displayed + i + 1}")
        axes[i, 1].axis('off')

        # Display the predicted segmentation mask
        axes[i, 2].imshow(predicted_mask, cmap=cmap, norm=norm)
        axes[i, 2].set_title(f"Predicted Mask {nb_images_displayed + i + 1}")
        axes[i, 2].axis('off')

        # Prepare the main image for contour drawing
        main = cv2.cvtColor(grayscale_img_8bit, cv2.COLOR_GRAY2BGR)

    
        # Iterate over each class to find and draw contours
        for class_label in np.unique(predicted_mask):
            if class_label == 7:  # Assuming class '0' might represent the background, you can adjust accordingly
                continue
            
            # Find contours for the current class
            contours = cv2.findContours((predicted_mask == class_label).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]

            # Draw contours for each detected contour of the current class
            for c in contours:
                colour = RGBforLabel.get(int(class_label), (255, 255, 255))  # Default to white if label is not found
                cv2.drawContours(main, [c], -1, colour, 1)


        # Display the annotated main image with contours
        axes[i, 3].imshow(main)
        axes[i, 3].set_title(f"Overlay with Contours {nb_images_displayed + i + 1}")

        # Difference of the actual and pedicted mask
        error = (actual_mask != predicted_mask).astype(int)
        axes[i, 4].imshow(error, cmap='gray', vmin=0, vmax=0.1)
        axes[i, 4].set_title(f'Error Map {i + 1}')
        axes[i, 4].axis('off')

        plt.tight_layout()
        nb_images_displayed += images_to_display
        
    plt.savefig(f'/home/phukon/Desktop/Model_Fitting/predicted_masks/{config['exp_name']}_with_CONTOURS_{nb_images_displayed + i + 1}.png', bbox_inches='tight')
    plt.show() #This line generate a bug when run in ssh on phenodrone : 
    

        # Update the count of displayed images

def display_segmentation_with_nice_overlay(images, true_labels, pred_labels, nb_images_to_display, class_names):
    cmap = plt.get_cmap('tab10')
    bounds = np.arange(len(class_names) + 1)  # Boundaries between classes (0, 1, 2, ..., 8)
    norm = BoundaryNorm(bounds, cmap.N)  # Ensures fixed color per class index
    columns = 4
    nb_images_displayed = 0  # Keep track of how many images have been displayed
    config = get_config()
    batch_size = images.shape[0]  # Get the current batch size
    images_to_display = min(batch_size, nb_images_to_display - nb_images_displayed)  # Number of images to display in this batch
    image_height, image_width = 256, 256
    # Create a figure and axes for the images, 4 columns for each: original, actual mask, predicted mask, overlay
    fig, axes = plt.subplots(images_to_display, columns, figsize=(20, images_to_display * 5))

    # Ensure that axes is always 2D, even if there's only one row
    axes = np.atleast_2d(axes)

    for i in range(images_to_display):
        img = images[i].detach().cpu().numpy().transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)

        # Convert the RGB image to grayscale (average of R, G, B channels)
        grayscale_img = np.mean(img, axis=-1)
        grayscale_img_8bit = np.interp(grayscale_img, (grayscale_img.min(), grayscale_img.max()), (0, 255)).astype(np.uint8)

        # Reshape true and predicted masks from lists back to 2D images
        actual_mask = np.array(true_labels[i * (image_height * image_width): (i + 1) * (image_height * image_width)]).reshape(image_height, image_width)
        predicted_mask = np.array(pred_labels[i * (image_height * image_width): (i + 1) * (image_height * image_width)]).reshape(image_height, image_width)
        
        # Display the original image (converted to grayscale)
        axes[i, 0].imshow(grayscale_img_8bit, cmap='gray')
        axes[i, 0].set_title(f"Original Grayscale Image {nb_images_displayed + i + 1}")
        axes[i, 0].axis('off')

        # Display the actual segmentation mask
        axes[i, 1].imshow(actual_mask, cmap=cmap, norm=norm)
        axes[i, 1].set_title(f"Expected Mask {nb_images_displayed + i + 1}")
        axes[i, 1].axis('off')

        # Display the predicted segmentation mask
        axes[i, 2].imshow(predicted_mask, cmap=cmap, norm=norm)
        axes[i, 2].set_title(f"Predicted Mask {nb_images_displayed + i + 1}")
        axes[i, 2].axis('off')

        # **Apply color map and blend for the overlay**
        colored_mask = cm.get_cmap('viridis')(predicted_mask / np.max(predicted_mask))  # Apply color map
        colored_mask = (colored_mask[:, :, :3] * 255).astype(np.uint8)  # Remove alpha channel and convert to 8-bit

        # Convert grayscale image to BGR format for OpenCV
        grayscale_img_bgr = cv2.cvtColor(grayscale_img_8bit, cv2.COLOR_GRAY2BGR)

        # Resize the colored mask to match the grayscale image size
        colored_mask_resized = cv2.resize(colored_mask, (grayscale_img_bgr.shape[1], grayscale_img_bgr.shape[0]))

        # **Blend the grayscale image with the color mask** (this is the key overlay step)
        alpha = 0.6  # Adjust transparency for the overlay
        blended_overlay = cv2.addWeighted(grayscale_img_bgr, 1, colored_mask_resized, alpha, 0)

        # Display the blended overlay (without contours)
        axes[i, 3].imshow(blended_overlay)
        axes[i, 3].set_title(f"Overlay {nb_images_displayed + i + 1}")
        axes[i, 3].axis('off')

        plt.tight_layout()
        nb_images_displayed += images_to_display
        
    plt.savefig(f'/home/phukon/Desktop/Model_Fitting/predicted_masks/{config['exp_name']}_with_OVERLAY_{nb_images_displayed + i + 1}.png', bbox_inches='tight')
    plt.show()

def display_individual_segmentation_masks(images, true_labels, pred_labels, num_images_to_display, class_names=None):

    if class_names is None:
        class_names = [
        "Background", 
        "Healthy Functional", 
        "Healthy Nonfunctional",
        "Necrotic Infected", 
        "Necrotic Dry", 
        "Bark", 
        "White Rot", 
        "Unknown"
    ]
        
    image_height, image_width = 256, 256
    batch_size = images.shape[0]  # Get the current batch size
    num_classes = 8  # Get the number of classes

    # Create a figure and axes for the images
    fig, axes = plt.subplots(min(batch_size, num_images_to_display), num_classes + 2, figsize=(15, min(batch_size, num_images_to_display) * 5))  # +2 for original and actual mask
    
    for i in range(min(batch_size, num_images_to_display)):  # Only loop through the first four images
        img = images[i].detach().cpu().numpy().transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
      
        actual_mask = np.array(true_labels[i * (image_height * image_width): (i + 1) * (image_height * image_width)]).reshape(image_height, image_width)
        predicted_mask = np.array(pred_labels[i * (image_height * image_width): (i + 1) * (image_height * image_width)]).reshape(image_height, image_width)
        
        # Display original image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Original Image {i + 1}")
        axes[i, 0].axis('off')  # Hide axes

        # Display actual segmentation
        axes[i, 1].imshow(actual_mask, cmap='jet', alpha=0.5)  
        axes[i, 1].set_title(f"Actual Mask {i + 1}")
        axes[i, 1].axis('off')  # Hide axes

        # Display predicted segmentation for each class
        for class_index in range(num_classes):
            class_mask = (predicted_mask == class_index).astype(np.float32) 
            axes[i, class_index + 2].imshow(class_mask, cmap='jet', alpha=0.5) 
            axes[i, class_index + 2].set_title(f"{class_names[class_index]}")
            axes[i, class_index + 2].axis('off') 

    plt.tight_layout()
    plt.show()

def display_segmentation(images, true_labels, pred_labels):

        batch_size = images.shape[0]  # Get the current batch size
        cols = 3  # Number of columns for the grid
        rows = batch_size  # Each row will display one image with its masks

        # Create a figure and axes for the images
        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
        axes = axes.flatten()  # Flatten the axes array for easy indexing
        image_height, image_width = 256, 256
        for i in range(batch_size):
            # Get the current image, actual mask, and predicted mask
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
            actual_mask = np.array(true_labels[i * (image_height * image_width): (i + 1) * (image_height * image_width)]).reshape(image_height, image_width)
            predicted_mask = np.array(pred_labels[i * (image_height * image_width): (i + 1) * (image_height * image_width)]).reshape(image_height, image_width)

            # Display original image
            axes[i * cols].imshow(img)
            axes[i * cols].set_title(f"Original Image {i + 1}")
            axes[i * cols].axis('off')  # Hide axes

            # Display actual segmentation
            axes[i * cols + 1].imshow(actual_mask, cmap='jet', alpha=0.5)  
            axes[i * cols + 1].set_title(f"Actual Mask {i + 1}")
            axes[i * cols + 1].axis('off')  # Hide axes

            # Display predicted segmentation
            axes[i * cols + 2].imshow(predicted_mask, cmap='jet', alpha=0.5)  
            axes[i * cols + 2].set_title(f"Predicted Mask {i + 1}")
            axes[i * cols + 2].axis('off')  # Hide axes

        plt.tight_layout()
        plt.show()

def basic_output_display(pred_labels, optimizer):

        titles = [
        "Background", "Healthy Functional", "Healthy Nonfunctional",
        "Necrotic Infected", "Necrotic Dry", "Bark", "White Rot", "Unknown"
        ]

        # Create a figure and axes
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))

        for i in range(8):
            ax = axes[i // 4, i % 4]  # Determine the position in the subplot grid
            img = pred_labels[0, i].detach().cpu().numpy()
            ax.imshow(img)
            ax.set_title(titles[i])
            plt.colorbar(ax.imshow(img), ax=ax)

        plt.show()

def save_predictions(predictions, output_dir, image_names):
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist
    for i, pred in enumerate(predictions):
        np.save(os.path.join(output_dir, f"{image_names[i]}.npy"), pred)  # Save each prediction as .npy

def load_predictions(output_dir, image_names):
    predictions = []
    for name in image_names:
        pred = np.load(os.path.join(output_dir, f"{name}.npy"))  # Load each prediction
        predictions.append(pred)
    return predictions


def compute_class_statistics(images, mask_pred, class_1=2, class_2=3):
    class_1_pixels = []
    class_2_pixels = []
    
    # Iterate through each predicted mask
    for i in range(len(images)):
        image = images[i].cpu().numpy().transpose(1, 2, 0)  # Convert tensor to numpy and channel-last
        mask = mask_pred[i].cpu().numpy()  # Convert mask to numpy array

        # Extract pixel values where the prediction is class_1 (Degradation Level 1)
        class_1_pixels.extend(image[mask == class_1].flatten())  # Flatten to 1D array
        
        # Extract pixel values where the prediction is class_2 (Degradation Level 2)
        class_2_pixels.extend(image[mask == class_2].flatten())  # Flatten to 1D array
    
    # Convert pixel lists to numpy arrays for easier processing
    class_1_pixels = np.array(class_1_pixels)
    class_2_pixels = np.array(class_2_pixels)
    
    # Calculate statistics for Degradation Level 1 (class_1)
    class_1_mean = np.mean(class_1_pixels) if len(class_1_pixels) > 0 else 0
    class_1_variance = np.var(class_1_pixels) if len(class_1_pixels) > 0 else 0
    
    # Calculate statistics for Degradation Level 2 (class_2)
    class_2_mean = np.mean(class_2_pixels) if len(class_2_pixels) > 0 else 0
    class_2_variance = np.var(class_2_pixels) if len(class_2_pixels) > 0 else 0
    
    # Compile stats into a dictionary
    stats_dict = {
        'class_1': {
            'mean': class_1_mean,
            'variance': class_1_variance
        },
        'class_2': {
            'mean': class_2_mean,
            'variance': class_2_variance
        }
    }
    
    return stats_dict
# Example usage
