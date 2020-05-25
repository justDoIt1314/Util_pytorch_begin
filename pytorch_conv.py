import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,64,3)
        self.conv2 = nn.Conv2d(64,128,3)
        self.fc1 = nn.Linear(128*32*32,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,10)
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x = x.view(x.size()[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x;

net = Net()

print(net)
params = list(net.parameters())
print(len(params))
print(params[0].size())
intput =  torch.randn(1,3,134,134)
out = net(intput)
target = torch.randn(1,10)
criterion = nn.MSELoss()
loss = criterion(out,target)
print(loss)

from torch import optim
optimizer = optim.SGD(net.parameters(),lr=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    output = net(intput)
    loss = criterion(output,target)
    print(loss)
    loss.backward()
    optimizer.step()

print(loss)
