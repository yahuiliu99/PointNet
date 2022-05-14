'''
Date: 2022-05-14 06:57:37
Author: Liu Yahui
LastEditors: Liu Yahui
LastEditTime: 2022-05-14 12:51:45
'''
import torch
import numpy as np
import os

import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import ModelNet40
from model import ClsPointNet2SSG, ClsPointNet2MSG, cal_loss
from train_utils import Trainer
from args import parse_args

from path import Path
path=Path("data/")

def _init_(seed):
    # fix random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.set_printoptions(10)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    # init random seed
    seed = 7727 # np.random.randint(1, 10000)
    _init_(seed)
    print("==> Random seed: {}".format(seed))
    # parse command line arguments
    arg = parse_args()

    # device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("==> Using accelerator: {}".format(device))

    # load dataset
    train_ds = ModelNet40(path, partition='train', num_points=arg.num_points)
    valid_ds = ModelNet40(path, partition='test', num_points=arg.num_points)

    train_loader = DataLoader(dataset=train_ds, batch_size=arg.train_bs, shuffle=True, num_workers=arg.num_workers)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=arg.valid_bs, shuffle=False, num_workers=arg.num_workers)
    print("==> Preparing dataset ...")

    # define model
    print("==> Initialize model ...")
    model = ClsPointNet2SSG(num_class=arg.classes, normal_channel=arg.use_normal)
    model = model.to(device)
    if arg.is_dist:
        model = nn.DataParallel(model)
    # for param in model.parameters():
    #     nn.init.normal_(param, mean=0, std=0.01)

    # define optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(),
                            lr=arg.lr)
    criterion = cal_loss
    best_val_acc = 0

    # # learning rate scheduler
    # scheduler = MultiStepLR(optimizer, milestones=[30, 50], gamma=0.1)

    # resume
    if arg.resume:
        print("==> Loading checkpoint ...")
        # use pre-trained model
        # the optimizer's state will be loaded to the device as same as the model. 
        # You must load the model to GPU at first, and then load the optimizer's state. 
        # So that both the model and the optimizer's state are loaded in GPU.
        model.to(device)
        if arg.is_dist:
            model = nn.DataParallel(model)

        checkpoint = torch.load(arg.ckptroot + 'model.pth',
                                map_location=lambda storage, loc: storage)

        arg.start_epoch = checkpoint['epoch'] 
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_val_acc = checkpoint['best_val_acc']

        print("==> Loading pre-trained successfully ...")

    # training
    print("==> Start training ...")
    trainer = Trainer(arg.ckptroot,
                      device,
                      model,
                      arg.epochs,arg.start_epoch,
                      criterion,optimizer,
                      train_loader,
                      valid_loader,
                      best_val_acc)
    trainer.train()
    # end of training
    print("==> End of training...")

