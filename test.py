import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from torch.utils.data import DataLoader

from data_proc import path, classes
from transforms import train_transforms
from datasets import PointCloudData
from model import ClsPointNet

model = ClsPointNet()
checkpoint = torch.load('./save/' + 'save_15.pth')
model.load_state_dict(checkpoint['state_dict'])
model.eval()

valid_ds = PointCloudData(path, valid=True, folder='test', transform=train_transforms())
valid_loader = DataLoader(dataset=valid_ds, batch_size=64, num_workers=0)
    
total_preds = []
total_labels = []
with torch.no_grad():
    for data in valid_loader:
        inputs, labels = data['pointcloud'].float(), data['category']
        outputs, _, _ = model(inputs.transpose(1,2))
        _, pred_class = torch.max(outputs.detach(), 1)
        total_preds += list(pred_class.numpy())
        total_labels += list(labels.numpy())

confu_m = confusion_matrix(total_labels, total_preds)

# function from https://deeplizard.com/learn/video/0LhiS6yu2qQ
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plt.figure(figsize=(8,8))
plot_confusion_matrix(confu_m, list(classes.keys()), normalize=True)
plt.savefig('./save/confusion_matrix.jpg')