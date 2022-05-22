'''
Date: 2022-05-22 06:25:51
Author: Liu Yahui
LastEditors: Liu Yahui
LastEditTime: 2022-05-22 07:02:59
'''
import os
import torch
import torch.nn as nn
import numpy as np
import sklearn.metrics as metrics

from torch.utils.data import DataLoader

from datasets import ModelNet40
from model import ClsPointNet2SSG, ClsPointNet2MSG
from args import parse_args

from path import Path
path=Path("data/")

arg = parse_args()

# device
os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ClsPointNet2MSG(num_class=arg.classes, normal_channel=arg.use_normal)
model = model.to(device)
if arg.is_dist:
    model = nn.DataParallel(model)

checkpoint = torch.load(arg.ckptroot + 'model.pth',
                                map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

test_ds = ModelNet40(path, partition='test', num_points=arg.num_points)
test_loader = DataLoader(dataset=test_ds, batch_size=arg.valid_bs, shuffle=False, num_workers=arg.num_workers)

test_loss = 0.0
test_pred = []
test_true = []

for data, label in test_loader:
    data, label = data.cuda(), label.cuda().squeeze()
    logits = model(data)
    preds = logits.max(dim=1)[1]
    test_true.append(label.cpu().numpy())
    test_pred.append(preds.detach().cpu().numpy())
test_true = np.concatenate(test_true)
test_pred = np.concatenate(test_pred)
test_acc = metrics.accuracy_score(test_true, test_pred)
print('Test Acc: %.6f'%(test_acc))
