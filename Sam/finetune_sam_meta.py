from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import os
import torch
import random
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from ResNet.dataset import MyDataset
from torch.utils.data import DataLoader

sam = sam_model_registry["default"](checkpoint="/home/phukon/Desktop/Model_Fitting/SAM_test/sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)


train_dataset = MyDataset()  # Implement your dataset loading here
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Create a directory to save masks if it doesn't exist
output_dir = "/home/phukon/Desktop/Model_Fitting/SAM_test/masks"
os.makedirs(output_dir, exist_ok=True)

# Define loss function and optimizer
optimizer = torch.optim.Adam(sam.parameters(), lr=1e-5)
loss_fn = F.cross_entropy()  # Choose an appropriate loss function

# Fine-tuning loop
num_epochs = 100
for epoch in range(num_epochs):
    sam.train()
    for images, masks in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = sam(images)
        
        # Calculate loss
        loss = loss_fn(outputs, masks)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the fine-tuned model
torch.save(sam.state_dict(), "/home/phukon/Desktop/Model_Fitting/SAM_test/fine_tuned_sam.pth")