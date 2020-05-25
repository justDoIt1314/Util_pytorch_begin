import torch
import torchvision
from torch import nn,optim
from torch.autograd import Variable
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from pathlib import Path
import requests
import cv2
batch_size=128
train_dataset = datasets.MNIST("data",train=True,transform=transforms.ToTensor(),download=True)
test_dataset = datasets.MNIST("data",train=False,transform=transforms.ToTensor(),download=True)
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

def loss_func(xb,yb):
    return nn.MSELoss(xb,yb)
class myModle(nn.Module):
    def __init__(self):
        super(myModle,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1,64,3,1,1),nn.ReLU(),nn.MaxPool2d(2,2))
        self.conv2 = nn.Sequential(nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.MaxPool2d(2,2))
        self.dense_1 = nn.Sequential(nn.Linear(7*7*128,128),nn.ReLU())
        self.dense_2 = nn.Sequential(nn.Linear(128,10))

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0],-1)
        x = self.dense_1(x)
        x = self.dense_2(x)
        return x
    
lr = 0.1
epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = myModle().to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

def train():
    for epoch in range(epochs):
        for i,data in enumerate(train_loader):
            inputs,labels = data
            inputs,labels = Variable(inputs).cuda(),Variable(labels).cuda()
            optimizer.zero_grad()
            outs = model(inputs)
            loss = criterion(outs,labels)
            
            loss.backward()
            optimizer.step()
            if i%100 == 0:
                print(loss.item())

def test():
    for epoch in range(epochs):
        correct = 0
        total = 0
        for i,data in enumerate(test_loader):
            inputs,labels = data
            inputs,labels = Variable(inputs).cuda(),Variable(labels).cuda()
            out_test = model(inputs)
            predict = torch.argmax(out_test,1)
            total += len(labels)
            correct += (predict==labels).sum()
        print("correct: ",correct)
        print("test acc:{0}".format(correct.item()/total))

train()
model.eval()
test()