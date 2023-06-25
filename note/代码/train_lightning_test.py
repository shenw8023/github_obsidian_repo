import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import time
from torchmetrics.functional import accuracy


#number of samples
n_positive,n_negative = 2000,2000

#positive samples
r_p = 5.0 + torch.normal(0.0,1.0,size = [n_positive,1]) 
theta_p = 2*np.pi*torch.rand([n_positive,1])
Xp = torch.cat([r_p*torch.cos(theta_p),r_p*torch.sin(theta_p)],axis = 1)
Yp = torch.ones_like(r_p)

#negative samples
r_n = 8.0 + torch.normal(0.0,1.0,size = [n_negative,1]) 
theta_n = 2*np.pi*torch.rand([n_negative,1])
Xn = torch.cat([r_n*torch.cos(theta_n),r_n*torch.sin(theta_n)],axis = 1)
Yn = torch.zeros_like(r_n)

#concat positive and negative samples
X = torch.cat([Xp,Xn],axis = 0)
Y = torch.cat([Yp,Yn],axis = 0)

ds = TensorDataset(X,Y)
ds_train,ds_val = torch.utils.data.random_split(ds,[int(len(ds)*0.7),len(ds)-int(len(ds)*0.7)])
dl_train = DataLoader(ds_train,batch_size = 50,shuffle=True,num_workers=2)
dl_val = DataLoader(ds_val,batch_size = 50,num_workers=2)


class Net(nn.Module):  
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2,4)
        self.fc2 = nn.Linear(4,8) 
        self.fc3 = nn.Linear(8,1)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.fc3(x) #don't need nn.Sigmoid()
        return y


class LitModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.save_hyperparameters()
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        # acc = accuracy(y_hat, y)  
        self.log('t_loss', loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        #TODO 进度条上显示的loss值是跟上一个取均值的结果，和training_step返回的实际loss值不同。The loss value shown in the progress bar is smoothed (averaged) over the last values, so it differs from the actual loss returned in train/validation step.
        time.sleep(0.3)
        return {'loss':loss, 'y_hat':y_hat}

    def training_epoch_end(self, train_step_output): #TODO接收的是所有training_step返回值组成的列表
        "处理所有training_step的返回内容"
        pass
        # print('sss')
        # print(type(train_step_output))
        # print(len(train_step_output))
        # one = train_step_output[0]
        # print(one['loss'].item())
        # print(one['y_hat'].shape)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.03)


if __name__ == "__main__":
    trainer = pl.Trainer(log_every_n_steps=3,precision=16, limit_train_batches=1.0, devices=[0], accelerator='gpu', max_epochs=10, callbacks=[checkpoint_callback])
    net = LitModel(Net())
    trainer.fit(net, train_dataloaders=dl_train)


