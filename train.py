import torch
import numpy as np

import torch.nn as nn
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler as DS  
from torch.nn.parallel import DistributedDataParallel as DDP

from data_proc import path
from transforms import train_transforms
from datasets import PointCloudData
from model import ClsPointNet, PointNetLoss
from train_utils import Trainer
from args import parse_args


if __name__ == '__main__':
    # parse command line arguments
    arg = parse_args()

    # device
    if arg.is_dist:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(arg.local_rank)
        device = torch.device(f'cuda:{arg.local_rank}')
        print("==> Using accelerator: {}".format(device))
        
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("==> Using accelerator: {}".format(device))

    # load dataset
    train_ds = PointCloudData(path, transform=train_transforms())
    valid_ds = PointCloudData(path, valid=True, folder='test', transform=train_transforms())

    if arg.is_dist:
        train_sampler = DS(train_ds)
    else:
        train_sampler = None

    train_loader = DataLoader(dataset=train_ds, batch_size=arg.train_bs, shuffle=(train_sampler is None), num_workers=arg.num_workers, pin_memory=True, sampler=train_sampler)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=arg.valid_bs, num_workers=arg.num_workers)
    print("==> Preparing dataset ...")

    # define model
    print("==> Initialize model ...")
    model = ClsPointNet(classes=arg.classes)
    model = model.to(device)
    if arg.is_dist:
        model = DDP(model, device_ids=[arg.local_rank])
    # for param in model.parameters():
    #     nn.init.normal_(param, mean=0, std=0.01)

    # define optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(),
                            lr=arg.lr)
    criterion = PointNetLoss()

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
            model = DDP(model, device_ids=[arg.local_rank])

        checkpoint = torch.load(arg.ckptroot + 'save_10.pth',
                                map_location=lambda storage, loc: storage)

        arg.start_epoch = checkpoint['epoch'] 
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        print("==> Loading checkpoint {} successfully ...".format(arg.start_epoch))

    # training
    print("==> Start training ...")
    trainer = Trainer(arg.ckptroot,
                      device,
                      model,
                      arg.epochs,arg.start_epoch,
                      criterion,optimizer,
                      train_loader,train_sampler,
                      valid_loader)
    trainer.train()

