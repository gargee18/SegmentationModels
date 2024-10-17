import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from image_mask_dataset import ImageMaskDataset
from custom_unet_with_skip import CustomUnetWithSkip
import torch.nn.functional as F
from config import get_config
from sklearn.metrics import f1_score
import Utils
from torch.utils.tensorboard import SummaryWriter
# Initialize the SummaryWriter

# mask_path = '/home/phukon/Desktop/Model_Fitting/weka_dataset/masks/'
# image_dir = '/home/phukon/Desktop/Model_Fitting/weka_dataset/images/train_set/'


def test(model, test_loader, device, epoch, writer, window_size, moving_avg_test_loss):
    model.eval()
    test_loss = 0.0
    all_labels = []
    all_preds = []
    mask_true=[]
    mask_pred=[]
    config = get_config()
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.permute(0, 1, 2, 3).to(device)
            masks = masks.squeeze().to(device)
            
            y_pred = model(images)
            masks_int = masks.long()
            
            test_class_weights_tensor = torch.tensor(Utils.get_weights(masks, device, int(config['num_classes'])), dtype=torch.float32).to(device)
            loss = F.cross_entropy(y_pred, masks_int, weight = test_class_weights_tensor, ignore_index=7)
            test_loss += loss.item()
            
            all_labels = masks_int.flatten().cpu().numpy() #array
            mask_true.extend(all_labels)   #list
            all_preds = torch.argmax(y_pred, dim=1).flatten().cpu().numpy() #array
            mask_pred.extend(all_preds)    #list
    
    test_loss /= len(test_loader)
    moving_avg_test_loss = ((window_size - 1) / window_size) * moving_avg_test_loss + (1 / window_size) * test_loss
    test_f1 = f1_score(all_labels, all_preds, labels=[0, 1, 2, 3, 4, 5, 6], average='weighted')
    test_f1_hf=f1_score(all_labels, all_preds, labels=[0],average='weighted')
    test_f1_hnf = f1_score(all_labels, all_preds, labels=[1],average='weighted')
    test_f1_neci = f1_score(all_labels, all_preds, labels=[2],average='weighted')
    test_f1_necd = f1_score(all_labels, all_preds, labels=[3],average='weighted')
    test_f1_wr = f1_score(all_labels, all_preds, labels=[4],average='weighted')
    test_f1_pit = f1_score(all_labels, all_preds, labels=[6],average='weighted')


    writer.add_scalar('F1/val', test_f1, epoch)
    writer.add_scalar('F1/val/hf', test_f1_hf, epoch)
    writer.add_scalar('F1/val/hnf', test_f1_hnf, epoch)
    writer.add_scalar('F1/val/neci', test_f1_neci, epoch)
    writer.add_scalar('F1/val/necd', test_f1_necd, epoch)
    writer.add_scalar('F1/val/wr', test_f1_wr, epoch)
    writer.add_scalar('F1/val/pit', test_f1_pit, epoch)
    writer.add_scalar('Loss/val', moving_avg_test_loss, epoch)