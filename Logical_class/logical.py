from __future__ import print_function,division
import torch
from tensorboardX import SummaryWriter
from torch import optim,nn,device
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import numpy as np
import torchvision
from torchvision import datasets,models,transforms
import matplotlib.pyplot as plt
from torchvision.models import resnet50
import glob

def loadData(path):
    feature = []
    lable = []
    fr = open(path)
    lines = fr.readlines()
    for line in lines:
        lineArr = line.strip().split()
        feature.append([lineArr[0], lineArr[1]])
        lable.append([lineArr[-1]])
    return np.array(feature, dtype='float32'), np.array(lable, dtype='float32')



def loadImage(path):
    model = resnet50(pretrained=False)
    model.cuda()
    model.load_state_dict(torch.load("X:\\Downloads\\resnet50-19c8e357.pth"))

    samples = glob.glob(path,'/*') 
    model.eval()
    imgs = [] 
    feature_1000 = None
    flag = True

    with torch.no_grad():
        for idx, imagePath in enumerate(samples):
            img = cv2.imread(imagePath)
            img = np.transpose(img,(2,0,1))
            imgs.append(img)
            if (idx+1) % 32 == 0:
                batch = np.asarray(imgs)
                batch = torch.tensor(batch,dtype=torch.float32)
                batch = Variable(batch).cuda()
                if flag:
                    feature_1000 = model.forward(batch)
                    flag = False
                else:
                    xx = model.forward(batch)
                    feature_1000 = torch.cat((feature_1000,xx),0)
                imgs = []
                batch = None
    
        if imgs != []:
            batch = np.asarray(imgs)
            batch = torch.tensor(batch,dtype=torch.float32)
            batch = Variable(batch).cuda()
            feature_1000 = torch.cat((feature_1000,model(batch)),0)
    
    x = feature_1000.cpu().detach().numpy()
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components = 100)
    principalComponents = pca.fit_transform(x)
    labels = np.random.rand(len(principalComponents),1)
    return principalComponents,labels
class BP(nn.Module):
    def __init__(self,layers):
        super(BP,self).__init__()
        self.linear_1 = nn.Linear(layers[0],layers[1],bias=True)
        self.activ_1 = nn.Sigmoid()
        self.linear_2 = nn.Linear(layers[1],layers[2],bias=True)
    def forward(self,x):
        x = self.linear_1(x)
        x = self.activ_1(x)
        x = self.linear_2(x)
        return x

def main():
    writer = SummaryWriter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features,labels = loadData('./Logical_class/set.txt')
    features,labels = torch.from_numpy(features),torch.from_numpy(labels)
    train_set = torch.utils.data.TensorDataset(features,labels)
    train_loaders = DataLoader(train_set,batch_size=8,shuffle=True)
    learning_rate = 1e-4
    model = BP([2,128,1])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    loss_criterion = torch.nn.MSELoss()
    data_size = len(train_loaders)
    for epoch in range(100):
        for idx,data in enumerate(train_loaders):
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            out = model(inputs)
            loss = loss_criterion(out,targets)
            writer.add_scalar("train_loss",loss.item(),idx + epoch * data_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() #模型权重根据它的导数更新
            print("epoch: {0}, loss: {1}".format(epoch,loss.item()))

    torch.save(model,"./Logical_class/logical.pth")

if __name__ == '__main__':
    main()