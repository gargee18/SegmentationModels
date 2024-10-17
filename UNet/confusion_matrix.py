from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Confusion Matrix
def ConfusionMatrix(mask_true,mask_pred, config):
    classes = config['class_names']
    conf_mtx = confusion_matrix(mask_true,mask_pred)
    dataframe_confmat = pd.DataFrame(conf_mtx/np.sum(conf_mtx,axis=1)[:, None],index = classes,columns= classes)
    plt.figure(figsize = (12,7))
    sn.heatmap(dataframe_confmat, annot=True)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('Predicted Label')
    plt.xlabel('True Label')
    plt.savefig(f'/home/phukon/Desktop/Model_Fitting/confusion_mats/{config['exp_name']}_Normalized_Confusion_Matrix_Heatmap.png', bbox_inches='tight')
    plt.show()

def ConfusionMatrixNoBG(mask_true,mask_pred, config):
    classes = config['class_names']
    conf_mtx = confusion_matrix(mask_true,mask_pred)
    last_class_index = len(classes) - 1  
    filtered_conf_mtx = np.delete(conf_mtx, last_class_index, axis=0)  # Remove last row
    filtered_conf_mtx = np.delete(filtered_conf_mtx, last_class_index, axis=1)  # Remove last column
    filtered_classes = classes[:-1]
    dataframe_confmat = pd.DataFrame(filtered_conf_mtx/np.sum(filtered_conf_mtx,axis=1)[:, None],index = filtered_classes,columns= filtered_classes)
    plt.figure(figsize = (12,7))
    sn.heatmap(dataframe_confmat, annot=True)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('Predicted Label')
    plt.xlabel('True Label')
    plt.savefig(f'/home/phukon/Desktop/Model_Fitting/confusion_mats/{config['exp_name']}_Normalized_Confusion_Matrix_Heatmap.png', bbox_inches='tight')
    plt.show()