import os 
import torch
import Utils 
import numpy as np
import torch.nn.functional as F
from config import get_config
from custom_unet_with_skip import CustomUnetWithSkip
from image_mask_dataset import ImageMaskDataset
from torch.utils.data import DataLoader
from data_loader import get_dataloaders

# Load configuration 
config = get_config() 
best_model_path = os.path.join(config['model_dir'], config['exp_name'] + "_best_model.pth")

# Set device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = CustomUnetWithSkip(config['in_ch'], config['num_classes']).to(device)

# Load saved model
model.load_state_dict(torch.load(best_model_path, weights_only=True))

# Set to evaluation mode
model.eval()

def apply_model(model, data_loader, device, config):
    val_loss = 0.0
    true_masks = []
    predicted_masks = []
    image_indices = []  # To store image indices for debugging

    with torch.no_grad():
        for batch_index, (images, masks) in enumerate(data_loader):
            images = images.permute(0, 1, 2, 3).to(device)
            masks = masks.squeeze().to(device)
            masks_int = masks.long()
            
            # Unique values for debugging
            print("Unique values in ground truth masks:", torch.unique(masks_int))
            
            y_pred = model(images)
            print("Unique values in predicted masks:", torch.unique(torch.argmax(y_pred, dim=1)))
            
            # Loss calculation
            val_class_weights_tensor = torch.tensor(Utils.get_weights(masks, device, int(config['num_classes'])), dtype=torch.float32).to(device)
            loss = F.cross_entropy(y_pred, masks_int, weight=val_class_weights_tensor, ignore_index=7)
            val_loss += loss.item()
            
            # Collecting all true and predicted masks along with their indices
            for i in range(images.shape[0]):  # Loop over batch size
                true_masks.append(masks_int[i].flatten().cpu().numpy())
                predicted_masks.append(torch.argmax(y_pred[i], dim=0).flatten().cpu().numpy())
                image_indices.append(batch_index * data_loader.batch_size + i)  # Calculate the overall index

    return np.array(true_masks), np.array(predicted_masks), images, image_indices

if __name__ == "__main__":
    # Prepare DataLoader for inference
    _, data_loader = get_dataloaders(config)

    # Run the model on the data_loader
    true_masks, predicted_masks, sample_images, image_indices = apply_model(model, data_loader, device, config)
    print(sample_images.shape)
    image_height, image_width = sample_images[0].shape[2:4]
    true_masks_reshaped = true_masks.reshape(-1, image_height, image_width)
      
    predicted_masks_reshaped = predicted_masks.reshape(-1, image_height, image_width)
    # Visualize the results
    for idx in range(len(true_masks)):
        print(f"Image Index: {image_indices[idx]}, True Mask: {true_masks[idx]}, Predicted Mask: {predicted_masks[idx]}")
    
    # You can also visualize using your existing Utils function
    Utils.display_segmentation_with_errormap(sample_images, true_masks, predicted_masks, 3, config['class_names'])
