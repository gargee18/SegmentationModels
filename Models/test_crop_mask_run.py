import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from diffusers import UNet2DModel
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
from new_seg import SegmentationDataset  # Import the class from the class definition file
from ExampleCustomUnet import UNet  # Import the class from the class definition file


def show_img(a):
    plt.imshow(a)
    plt.show()
        

# Define device, model, transformations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
transform_image = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
transform_mask = transforms.Compose([
    transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
    transforms.ToTensor(),
])



#Define annotations and source image, and create the dataloader
json_file_path = '/home/phukon/Desktop/Annotation_VIA/Train/jsons/via_project_2Sep2024_16h21m_kickstart_json.json'
image_dir = '/home/phukon/Desktop/Annotation_VIA/Train/imgs_kickstart/'
dataset = SegmentationDataset(json_file=json_file_path, image_dir=image_dir, transform_mask=transform_mask, transform_image=transform_image)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)



# Define loss function and optimizer, and epochs
criterion = nn.CrossEntropyLoss()  # Multi-class segmentation task
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10



# Training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    model.train()  # Set model to training mode
    running_loss = 0.0

    for images, masks in dataloader:
        print(np.mean(masks.cpu().numpy()))
        show_img(masks.cpu().numpy()[0,0,0,:,:])
        images = images.to(device)
        masks = masks.squeeze()
        masks = torch.argmax(masks, dim=1)
        masks = masks.to(device)

        # Print the shape and a small part of the mask tensor from the middle
        print("Mask Tensor Shape:", masks.shape)
        
        # Determine the center indices
        batch_size, height, width = masks.shape
        mid_h, mid_w = height // 2, width // 2

        # Define the size of the section you want to print
        section_size = 10
        half_section = section_size // 2

        # Calculate the start and end indices
        start_h = max(mid_h - half_section, 0)
        end_h = min(mid_h + half_section, height)
        start_w = max(mid_w - half_section, 0)
        end_w = min(mid_w + half_section, width)
        print(f"Mask Tensor (middle part) from indices [{start_h}:{end_h}, {start_w}:{end_w}]:")
        print(f"mean={np.mean(masks.cpu().numpy()[:, :,:])}")
        # Handle different possible dimensions
        if masks.dim() == 3:  # Shape could be (batch_size, height, width)
            print(masks.cpu().numpy()[0, start_h:end_h, start_w:end_w])  # Print middle 10x10 section of the first mask in the batch
        elif masks.dim() == 2:  # Shape could be (height, width)
            print(masks.cpu().numpy()[start_h:end_h, start_w:end_w])  # Print middle 10x10 section of the mask
        else:
            print("Unexpected tensor dimensions:", masks.dim())

        # Zero the gradients
        optimizer.zero_grad()
        outputs = model(images)  # Get the output from the model

        # Compute loss
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()  # Update the weights
        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")
