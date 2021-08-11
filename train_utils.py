import os
import torch
import numpy as np

import torch.distributed as dist

class Trainer(object):
    def __init__(self, ckptroot, device, model, epochs, start_epoch, criterion, optimizer, train_loader, train_sampler, val_loader=None):
        super().__init__()
        self.ckptroot = ckptroot
        self.device = device
        self.model = model
        self.epochs = epochs
        self.start_epoch = start_epoch
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.train_sampler = train_sampler
        self.val_loader = val_loader
     
    def train(self): 

        for epoch in range(self.start_epoch, self.epochs):
            if self.train_sampler:
                # Shuffle
                self.train_sampler.set_epoch(epoch)

            # Training
            self.model.train()
            train_loss = []

            for step, data in enumerate(self.train_loader):
                inputs, labels = data['pointcloud'].to(self.device).float(), data['category'].to(self.device)
                # input.shape : (B, N, 3) ==> (B, 3, N)
                outputs, m_3, m_64 = self.model(inputs.transpose(1,2))
                loss = self.criterion(outputs, labels, m_3, m_64)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss.append(loss.detach().item())
                if step % 10 ==9:
                    print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f \n' 
                    %(epoch+1, step+1, len(self.train_loader), np.mean(train_loss)))
                    train_loss.clear()

            # Validation
            self.model.eval()
            correct = total = 0 

            if self.val_loader:
                with torch.no_grad():
                    for data in self.val_loader:
                        inputs, labels = data['pointcloud'].to(self.device).float(), data['category'].to(self.device)
                        # input.shape : (B, N, 3) ==> (B, 3, N)
                        outputs, _, _ = self.model(inputs.transpose(1,2))
                        _, pred = torch.max(outputs.detach(), 1)

                        correct += (pred == labels).sum().item()
                        total += labels.size(0)
                val_acc = 100.* correct / total
                print('Valid accuracy: %d %%' %val_acc)


            # Save Model
            if (epoch+1) % 5 == 0:
                if self.train_sampler:
                    if dist.get_rank() == 0:
                        state = {
                            'epoch': epoch+1,
                            'state_dict': self.model.module.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                        }
                        torch.save(state, os.path.join(self.ckptroot, 'save_{}.pth'.format(epoch+1)))
                        print("==> Save checkpoint ...")

                else:
                    state = {
                        'epoch': epoch+1,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    }
                    torch.save(state, os.path.join(self.ckptroot, 'save_{}.pth'.format(epoch+1)))
                    print("==> Save checkpoint ...")
                
