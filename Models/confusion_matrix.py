from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from config import get_config
import numpy as np

#Confusion Matrix
def ConfusionMatrix(mask_true,mask_pred):
    config = get_config()
    classes = config['class_names']
    conf_mtx = confusion_matrix(mask_true,mask_pred)
    dataframe_confmat = pd.DataFrame(conf_mtx/np.sum(conf_mtx,axis=1)[:, None],index = classes,columns= classes)
    plt.figure(figsize = (12,7))
    sn.heatmap(dataframe_confmat, annot=True)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
