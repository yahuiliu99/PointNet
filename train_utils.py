'''
Date: 2022-05-14 07:10:31
Author: Liu Yahui
LastEditors: Liu Yahui
LastEditTime: 2022-05-14 09:35:27
'''
import os
import torch
import numpy as np
from tqdm import tqdm
import sklearn.metrics as metrics


class Trainer(object):
    def __init__(self, ckptroot, device, model, epochs, start_epoch, criterion, optimizer, train_loader, val_loader, best_val_acc = 0):
        super().__init__()
        self.ckptroot = ckptroot
        self.device = device
        self.model = model
        self.epochs = epochs
        self.start_epoch = start_epoch
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.best_val_acc = best_val_acc
     
    def train(self): 

        for epoch in range(self.start_epoch, self.epochs):
            print('Epoch (%d/%s):' % (epoch + 1, self.epochs))
            # Training
            self.model.train()
            train_loss = 0.0
            count = 0.0
            train_pred = []
            train_true = []

            # for step, data in enumerate(self.train_loader):
            #     inputs, labels = data['pointcloud'].to(self.device).float(), data['category'].to(self.device)
            #     outputs, m_3, m_64 = self.model(inputs.transpose(1,2))
            #     loss = self.criterion(outputs, labels, m_3, m_64)
            
            for data, label in tqdm(self.train_loader):
                    
                data, label = data.cuda(), label.cuda().squeeze()
                batch_size = data.size()[0]
                self.optimizer.zero_grad()
                logits = self.model(data)
                loss = self.criterion(logits, label)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

                preds = logits.max(dim=1)[1]
                count += batch_size
                train_loss += loss.item() * batch_size
                train_true.append(label.cpu().numpy())
                train_pred.append(preds.detach().cpu().numpy())

            train_true = np.concatenate(train_true)
            train_pred = np.concatenate(train_pred)
            train_loss = train_loss*1.0/count
            train_acc = metrics.accuracy_score(train_true, train_pred)
            print('Train loss: %.6f, train acc: %.6f' % (train_loss, train_acc))

            # Validation
            
            self.model.eval()
            val_loss = 0.0
            count = 0.0
            val_pred = []
            val_true = []

            with torch.no_grad():
                for data, label in self.val_loader:
                    data, label = data.cuda(), label.cuda().squeeze()
                    batch_size = data.size()[0]
                    logits = self.model(data)
                    loss = self.criterion(logits, label)
                    preds = logits.max(dim=1)[1]
                    count += batch_size
                    val_loss += loss.item() * batch_size
                    val_true.append(label.cpu().numpy())
                    val_pred.append(preds.detach().cpu().numpy())

                val_true = np.concatenate(val_true)
                val_pred = np.concatenate(val_pred)
                val_loss = val_loss*1.0/count
                val_acc = metrics.accuracy_score(val_true, val_pred)
                print('Val loss: %.6f, Val acc: %.6f' % (val_loss, val_acc))


            # Save Model
            if val_acc >= self.best_val_acc:
                self.best_val_acc = val_acc
                state = {
                    'epoch': epoch+1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'val_acc': self.best_val_acc,
                }
                torch.save(state, os.path.join(self.ckptroot, 'model.pth'))
                print("==> Save checkpoint ...")

            print('best: %.3f' % self.best_val_acc)
                
