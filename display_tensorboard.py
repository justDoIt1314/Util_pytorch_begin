import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms,datasets
import torchvision
import torchvision.transforms as transforms
 
from torch.utils.tensorboard import SummaryWriter
 
def get_num_correct(preds,labels):
    return preds.argmax(dim=1).eq(labels).sum().item()
 
class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1,64,3,1,1),nn.ReLU(),nn.BatchNorm2d(64),nn.MaxPool2d(2,2))
        self.conv2 = nn.Sequential(nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.MaxPool2d(2,2),nn.Dropout2d(0.3))
        self.dense_1 = nn.Sequential(nn.Linear(7*7*128,128),nn.ReLU())
        self.dense_2 = nn.Sequential(nn.Linear(128,10))

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0],-1)
        x = self.dense_1(x)
        x = self.dense_2(x)
        return x
 
if __name__ == '__main__':

    train_set = datasets.MNIST("data",train=True,transform=transforms.ToTensor(),download=True)
    train_loader=torch.utils.data.DataLoader(train_set,batch_size=100,shuffle=True)
 
    #tensor board
    tb=SummaryWriter()
    network=Network()
#取出训练用图
    images,labels=next(iter(train_loader))
    grid=torchvision.utils.make_grid(images)
#想用tensorboard看什么，你就tb.add什么。image、graph、scalar等
    tb.add_image('images', grid)
    x = torch.rand(100, 1, 28, 28)
    tb.add_graph(model=network,input_to_model=images)
    tb.close()
    exit(0)