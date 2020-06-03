from __future__ import print_function,division
import torch
from torch import optim,nn,device
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import numpy as np
import torchvision
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset
from torchvision import datasets,models,transforms
import matplotlib.pyplot as plt
import time,os,copy
import cv2
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def load_csv_to_numpy():
    train_data = pd.read_csv("./Titanic/train.csv",header=None)
    test_data = pd.read_csv("./Titanic/test.csv",header=None)
    train_data = train_data.fillna(0)
    test_data = test_data.fillna(0)
    train_data.head()
    train_data = train_data.values.tolist()
    train_data = np.asarray(train_data)
    
    test_data = test_data.values.tolist()
    test_data = np.asarray(test_data)
    print(train_data)
    train_data = np.delete(train_data,0,0)
    test_data = np.delete(test_data,0,0)
    
    train_labels = train_data[:,1]
    train_labels = np.squeeze(train_labels)
    train_labels = np.asarray(train_labels,dtype='int64')

    train_data = np.delete(train_data,[0,1,3,8],1)
    test_data = np.delete(test_data,[0,2,7],1)

    
    train_features = []
    test_features = []
    for j in range(8):
        fea = train_data[:,j]
        test_fea = test_data[:,j]
        if j == 7:
            for i in range(len(fea)):
                if fea[i] == 'S':
                    fea[i] = 0.3
                elif fea[i] == 'Q':
                    fea[i] = 0.6
                else:
                    fea[i] = 1
            for i in range(len(test_fea)):
                if test_fea[i] == 'S':
                    test_fea[i] = 0.3
                elif test_fea[i] == 'Q':
                    test_fea[i] = 0.6
                else:
                    test_fea[i] = 1
            fea = np.asarray(fea,dtype='float')
            test_fea = np.asarray(test_fea,dtype='float')
        elif j == 1:
            fea = np.where(fea=='male',1,0)
            test_fea = np.where(test_fea=='male',1,0)
        elif j == 2:
            fea = np.asarray(fea,dtype='float')
            fea = fea / np.max(fea)

            test_fea = np.asarray(test_fea,dtype='float')
            test_fea = test_fea / np.max(test_fea)


        elif j == 6:
            fea = np.where(fea=='0',0,1)
            test_fea = np.where(test_fea=='0',0,1)
        else:
            fea = np.asarray(fea,dtype='float')
            #fea = fea / np.max(fea)

            test_fea = np.asarray(test_fea,dtype='float')
            #test_fea = test_fea / np.max(test_fea)

        train_features.append(fea)
        test_features.append(test_fea)
    train_features = np.asarray(train_features,dtype='float32')
    train_features = np.transpose(train_features)
    
    test_features = np.asarray(test_features,dtype='float32')
    test_features = np.transpose(test_features)

    return train_features,train_labels,test_features


class TitanicNet(nn.Module):
    def __init__(self):
        super(TitanicNet,self).__init__()
        self.block_1 = nn.Sequential(nn.Linear(8,64),nn.BatchNorm1d(64),nn.Sigmoid(),nn.Dropout(0.2))
        self.block_2 = nn.Sequential(nn.Linear(64,128),nn.BatchNorm1d(128),nn.Sigmoid(),nn.Dropout(0.2))
        self.block_3 = nn.Sequential(nn.Linear(128,256),nn.BatchNorm1d(256),nn.Sigmoid(),nn.Dropout(0.3))
        self.block_4 = nn.Sequential(nn.Linear(256,1024),nn.BatchNorm1d(1024),nn.ReLU(),nn.Dropout(0.3))
        self.block_5 = nn.Sequential(nn.Linear(1024,32),nn.BatchNorm1d(32),nn.ReLU(),nn.Dropout(0.2))
        self.block_6 = nn.Sequential(nn.Linear(32,2),nn.Softmax())
    def forward(self,x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_6(x)
        return x

def main():
    test_data = pd.read_csv("./Titanic/test.csv")
    batch_size = 128
    lr = 1e-4
    epochs = 10000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()
    model = TitanicNet()
    model.to(device)
    if os.path.exists("./Titanic/titanic.pth"):
        model = torch.load("./Titanic/titanic.pth")
    model.to(device)
    loss_criterion = nn.CrossEntropyLoss()
    opitimizer = optim.Adam(model.parameters(),lr=lr)
    train_features,train_labels,test_features = load_csv_to_numpy()
    train_features = torch.from_numpy(train_features)
    train_labels = torch.from_numpy(train_labels)
    test_features = torch.from_numpy(test_features)
    train_set = TensorDataset(train_features,train_labels)
    train_loaders = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=2)
    dataloaders_size = len(train_loaders)


    for epoch in range(epochs):
        for idx,(inputs,labels) in enumerate(train_loaders):
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model(inputs)
            loss = loss_criterion(output,labels)
            indice = torch.argmax(output,1)
            #print(indice)
            correct_sum = torch.sum(indice == labels)
            writer.add_scalar("train_loss",loss.item(),idx+epoch*dataloaders_size)
            opitimizer.zero_grad()
            loss.backward()
            opitimizer.step()
            print("epoch: {0}/{1}, batch:{2}/{3} loss: {4}, accuracy:{5}".format(epoch,epochs,idx,dataloaders_size,loss.item(),correct_sum.item()/batch_size))
        if epoch %100 == 0:
            torch.save(model,"./Titanic/titanic.pth")
        if epoch % 10 == 0:
            test_features = test_features.to(device)
            test_out = model(test_features)
            indice_2 = torch.argmax(test_out,1)
            print(indice_2)

    test_features = test_features.to(device)
    output = model(test_features)
    predictions = torch.argmax(output,1)
    results = pd.Series(predictions.cpu().numpy(),name="Survived")
    submission = pd.concat([pd.Series(range(892,1310),name = "PassengerId"),results],axis = 1)
    
    submission.to_csv("my_submission.csv",index=False)

if __name__ == '__main__':
    main()