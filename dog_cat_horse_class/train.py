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

# entroy=nn.CrossEntropyLoss()
# inputs=torch.Tensor([[-0.7715, -0.6205,-0.2562]])
# target = torch.tensor([1])
# output = entroy(inputs, target)
# print(output)

# s1 = torch.tensor([[2,3,4],[3,4,6]])
# s2 = torch.tensor([[2,3,4],[1,32,4]])
# indx1 = torch.argmax(s1,1)
# indx2 = torch.argmax(s2,1)
# sum = torch.sum(indx1 == indx2)
# sum.item()
# class ResNet_50(nn.Module):
#     def __init__(self,num_classes):
#         super(ResNet_50,self).__init__()
#         self.features = nn.Sequential(
#                 nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
#                 nn.ReLU(),
#                 nn.BatchNorm2d(64),
#                 nn.MaxPool2d(kernel_size=3, stride=2),
#                 nn.Dropout(0.5),
#                 nn.Conv2d(64, 192, kernel_size=5, padding=2),
#                 nn.ReLU(),
#                 nn.BatchNorm2d(192),
#                 nn.MaxPool2d(kernel_size=3, stride=2),
#                 nn.Conv2d(192, 384, kernel_size=3, padding=1),
#                 nn.ReLU(),
#                 nn.Conv2d(384, 256, kernel_size=3, padding=1),
#                 nn.ReLU(),
#                 nn.Conv2d(256, 256, kernel_size=3, padding=1),
#                 nn.ReLU(),
#                 nn.BatchNorm2d(256),
#                 nn.MaxPool2d(kernel_size=3, stride=2),
#             )
#         self.classifier = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(2304, 256),
#             nn.ReLU(),
#             nn.Dropout(0.25),
#             nn.ReLU(),
#             nn.Linear(256, 64),
#             nn.ReLU(),
#             nn.Dropout(0.25),
#             nn.Linear(64, num_classes),
#             nn.Softmax()
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x
    
class myModle(nn.Module):
    def __init__(self,num_classes):
        super(myModle,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,64,3,1,1),nn.ReLU(),nn.BatchNorm2d(64),nn.MaxPool2d(2,2))
        self.conv2 = nn.Sequential(nn.Conv2d(64,128,3,padding=1),nn.ReLU(),nn.MaxPool2d(2,2),nn.Dropout2d(0.3))
        self.dense_1 = nn.Sequential(nn.Linear(131072,128),nn.ReLU())
        self.dense_2 = nn.Sequential(nn.Linear(128,num_classes))

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0],-1)
        x = self.dense_1(x)
        x = self.dense_2(x)
        return x

    
def loadImage(train_path):
    train_x = []
    train_y = []
    class_path = os.listdir(train_path)
    for class_name in class_path:
        img_names = os.listdir(os.path.join(train_path,class_name))
        for img in img_names:
            img_path = os.path.join(train_path+"/"+class_name,img)
            train_x.append(cv2.imread(img_path)/255.0)
            y = np.zeros(3)
            y[int(class_name)] = 1
            train_y.append(y)
    return np.asarray(train_x),np.asarray(train_y)


def main():
    ################# 加载模型  ####################################
    model_path = "./dog_cat_horse_class/dog_cat_horse_class.pth"
    writer = SummaryWriter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = myModle(num_classes=2)  
    if os.path.exists(model_path):
        model = torch.load(model_path) 
    model.to(device)

    ################## 构建数据集  ##########################
    data_transform = transforms.Compose([
        transforms.Resize((128,128)),
        # transforms.RandomRotation(20),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])

    train_set = datasets.ImageFolder("G:/Course/Data_Science/keras-screen-on-off-master/project/spiders/downloads/train",data_transform)
    test_set = datasets.ImageFolder("G:/Course/Data_Science/keras-screen-on-off-master/project/spiders/downloads/test",data_transform)
    batch_size = 10
    dataloaders = DataLoader(train_set,batch_size,shuffle=True,num_workers=4)
    test_loaders = DataLoader(test_set,batch_size,shuffle=True,num_workers=4)
    test_loaders_size = len(test_loaders)
    dataloaders_size = len(dataloaders)
    epochs = 200

    ############  选择优化器和损失函数 ########################
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    
    ############  开始训练和评估  #################################
    for epoch in range(epochs):
        model.train()
        for idx,(inputs,labels) in enumerate(dataloaders):
            inputs = inputs.to(device)
            labels = labels.to(device)
            out = model(inputs)
            
            loss = loss_criterion(out,labels)
            #writer.add_scalar("train_loss",loss.item(),idx+epoch*dataloaders_size)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("epoch: {0}/{1}, batch:{2}/{3} loss: {4}".format(epoch,epochs,idx,dataloaders_size,loss.item()))
        model.eval()
        for idx,(inputs,labels) in enumerate(test_loaders):
            inputs = inputs.to(device)
            labels = labels.to(device)
            out = model(inputs)
            loss = loss_criterion(out,labels)
            index1 = torch.argmax(out,1)
            
            correct_sum = torch.sum(index1 == labels)
            print("epoch: {0}/{1}; batch:{2}/{3}; loss: {4}; accuracy:{5}".format(epoch,epochs,idx,test_loaders_size,loss.item(),1.0*correct_sum.item()/len(labels)))
      
        torch.save(model,"./dog_cat_horse_class/dog_cat_horse_class.pth")


if __name__ == '__main__':
    main()



