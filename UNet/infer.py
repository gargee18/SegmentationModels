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
import Utils

#Load configuration 
config = get_config() 
best_model_path = os.path.join(config['model_dir'], config['exp_name']+ "_best_model.pth")

#Set device 
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load model
model=CustomUnetWithSkip(config['in_ch'], config['num_classes']).to(device)

#Load saved model
model.load_state_dict(torch.load(best_model_path,weights_only=True))

#Set to evaluation mode
model.eval()

def apply_model(model,data_loader,device,config, image_indices=None):
    # if(images_number is None):
    #     images_number=[0,1,2,3]
    val_loss = 0.0
    mask_true=[]
    mask_pred=[]
    image_names = [] 
    
    with torch.no_grad():
        for images, masks in data_loader:
            # print("Unique values in ground truth masks:", torch.unique(masks))
            # break
            images = images.permute(0, 1, 2, 3).to(device)
            masks = masks.squeeze().to(device)
            masks_int = masks.long()
            print("Unique values in ground truth masks:", torch.unique(masks_int))
            y_pred = model(images)
            print("Unique values in predicted masks:", torch.unique(torch.argmax(y_pred, dim=1)))
            
            
            val_class_weights_tensor = torch.tensor(Utils.get_weights(masks, device, int(config['num_classes'])), dtype=torch.float32).to(device)
            loss = F.cross_entropy(y_pred, masks_int, weight = val_class_weights_tensor, ignore_index=7)
            val_loss += loss.item()
            
            all_labels = masks_int.flatten().cpu().numpy() #array
            mask_true.extend(all_labels)   #list
            all_preds = torch.argmax(y_pred, dim=1).flatten().cpu().numpy() #array
            mask_pred.extend(all_preds)    #list   
            
    # Convert to numpy arrays for easier indexing
    mask_true = np.array(mask_true)
    mask_pred = np.array(mask_pred)

    return mask_true, mask_pred, images


if __name__ == "__main__":
    # Prepare DataLoader for inference
    _, data_loader = get_dataloaders(config)

    # Load the entire dataset to access filenames
    dataset = ImageMaskDataset(
        image_dir=config['image_dir'],
        mask_dir=config['mask_path'],
        augment=config['do_augmentation']
    )
    # Get the filenames from the dataset
    image_filenames = dataset.image_filenames

    # Run the model on the data_loader
    true_masks, predicted_masks, sample_images = apply_model(model, data_loader, device, config, None)

    # Specify the indices of the images to be displayed
    # images_to_display = [1]  
    print("True Masks:", true_masks)
    print("Predicted Masks:", predicted_masks)

    # Print the names of the images being displayed
    # for idx in images_to_display:
    #     print("Displaying image:", image_filenames[idx])
 
    Utils.display_segmentation_with_errormap(sample_images, true_masks, predicted_masks, 4, config['class_names'])

