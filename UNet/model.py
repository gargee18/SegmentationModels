import torch
import torch.nn as nn
import torch.optim as optim
from custom_unet_with_skip import CustomUnetWithSkip
import torch.nn.functional as F

def setup_model_and_optimizer(config, device):
    model = CustomUnetWithSkip(1, 8).to(device)
   
    optimizer = None
    if config['optimizer_name'] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])
    elif config['optimizer_name'] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    # optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])
    return model, optimizer