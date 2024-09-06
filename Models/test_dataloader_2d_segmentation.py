from dataloader_2d_segmentation import SegmentationDataset  # Import the class from the class definition file
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

#BATCH SIZE FOR TESTING
batch_size=2


#Define annotations and source image, and create the dataloader
json_file_path = '/home/phukon/Desktop/Annotation_VIA/Train/jsons/via_project_2Sep2024_16h21m_kickstart_json.json'
image_dir = '/home/phukon/Desktop/Annotation_VIA/Train/imgs_kickstart/'
dataset = SegmentationDataset(json_file=json_file_path, image_dir=image_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


for images, masks in dataloader:
    images_np = images.cpu().numpy()
    masks_np = masks.cpu().numpy()

    images_np = np.transpose(images_np, (0, 2, 3, 1))  # Change shape from [batch_size, channels, height, width] to [batch_size, height, width, channels]
    masks_np = masks_np.squeeze(1)
    # Display images and masks
    for i in range(images_np.shape[0]):
        plt.figure(figsize=(12, 6))
        print("Image Size: "+str(images_np.shape))  # Example: torch.Size([8, 3, 256, 256])
        print("Mask Size: "+str(masks_np.shape))   # Example: torch.Size([8, 256, 256])

        plt.subplot(1, 2, 1)
        plt.title(f'Image {i}')
        print("Mean: "+str(np.mean(images_np[i,:,:,0]))+" , "+"Std: "+str(np.std(images_np[i])))
        plt.imshow(images_np[i])
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title(f'Mask {i}')
        plt.imshow(masks_np[i,:,:], cmap='gray')  
        print("Max: "+str(np.max(masks_np[i]))+" , "+"Min: "+str(np.min(masks_np[i])))
        plt.axis('off')
        plt.show()


