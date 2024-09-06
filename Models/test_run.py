import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from dataloader_2d_segmentation import SegmentationDataset
from CustomUnet import CustomUnet
import torch.nn.functional as F






#Define annotations and source image, and create the dataloader
json_file_path = '/home/phukon/Desktop/Annotation_VIA/Train/jsons/via_project_2Sep2024_16h21m_kickstart_json.json'
image_dir = '/home/phukon/Desktop/Annotation_VIA/Train/imgs_kickstart/'
dataset = SegmentationDataset(json_file=json_file_path, image_dir=image_dir)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)



# Define device, model, transformations loss function and optimizer, and epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomUnet().to(device)
criterion = nn.CrossEntropyLoss()  # Multi-class segmentation task
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 500








# Training loop
for epoch in range(num_epochs):
    #print(f"Epoch {epoch + 1}/{num_epochs}")
    model.train()  # Set model to training mode
    running_loss = 0.0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.squeeze()
        #print(np.mean(masks.cpu().numpy()))
        #print(np.shape(masks))
#        masks = torch.argmax(masks, dim=1)
        masks = masks.to(device)

        # Print the mask tensor
        #print("Mask Tensor (part):")
        # Zero the gradients
        optimizer.zero_grad()
        outputs = model(images)  # Get the output from the model
#        print(np.shape(outputs))
#        plt.imshow(outputs[0,1].detach().cpu().numpy())
#        plt.show()
        # Compute loss
        masks_int=masks.long()
#        one_hot_masks=F.one_hot(masks_int,num_classes=6)
    
 #       print(np.shape(one_hot_masks))
        loss = criterion(outputs, masks_int)
        loss.backward()
        optimizer.step()  # Update the weights
        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")



for images, masks in dataloader:
    images = images.to(device)
    masks = masks.squeeze()
    #print(np.mean(masks.cpu().numpy()))
    #print(np.shape(masks))
#        masks = torch.argmax(masks, dim=1)
    masks = masks.to(device)

    # Print the mask tensor
    #print("Mask Tensor (part):")
    # Zero the gradients
    optimizer.zero_grad()
    outputs = model(images)  # Get the output from the model
    plt.imshow(outputs[0,5].detach().cpu().numpy())
    plt.show()





# # Training loop
# for epoch in range(num_epochs):
#     print(epoch)
#     model.train()  # Set model to training mode
#     running_loss = 0.0

#     for images, masks in dataloader:
#         images = images.to(device)
#         masks = masks.squeeze()
#         masks=torch.argmax(masks,dim=1)
#         masks=masks.to(device) 

#         # Zero the gradients
#         optimizer.zero_grad()
#         outputs = model(images)  # Get the output from the model
#         #print("Toto outputs "+str(np.shape(outputs)))
#         #print("Toto masks "+str(np.shape(masks)))
#         loss = criterion(outputs, masks)
#         loss.backward()
#         optimizer.step()        # Update the weights
#         running_loss += loss.item()

#     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")
# # torch.save(model.state_dict(), "unet_segmentation_model.pth")
