# Trial of inference of a diffuser model from hugging face
import os
import json
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, Dataset
from PIL import Image , ImageDraw
import matplotlib.pyplot as plt
import torch.nn as nn
from dataloader_2d_segmentation import SegmentationDataset


# Using locally saved pretrained model
class CustomUnet(nn.Module):
    def __init__(self):
        super(CustomUnet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.final_layer = nn.Conv2d(3, 6, kernel_size=1)  # 6 classes

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        return self.final_layer(x2)


################ This is the test if required #############################
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = CustomUnet().to(device)
#print(model)


