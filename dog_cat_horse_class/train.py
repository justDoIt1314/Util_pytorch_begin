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

entroy=nn.CrossEntropyLoss()
inputs=torch.Tensor([[-0.7715, -0.6205,-0.2562]])
target = torch.tensor([1])
output = entroy(inputs, target)
print(output)

# s1 = torch.tensor([[2,3,4],[3,4,6]])
# s2 = torch.tensor([[2,3,4],[1,32,4]])
# indx1 = torch.argmax(s1,1)
# indx2 = torch.argmax(s2,1)
# sum = torch.sum(indx1 == indx2)
# sum.item()
class ResNet_50(nn.Module):
    def __init__(self,num_classes):
        super(ResNet_50,self).__init__()
        # self.vgg = models.vgg16()
        # self.vgg.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, num_classes)
        # )
        self.resnet = models.resnet50()
        num_frts =  self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_frts,num_classes)
    def forward(self,x):
        x = self.resnet(x)
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
    model_path = "./dog_cat_horse_class/class_model.pth"
    writer = SummaryWriter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet_50(num_classes=3)  
    if os.path.exists(model_path):
        model = torch.load(model_path) 
    model.to(device)

    ################## 构建数据集  ##########################
    data_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomRotation(1.0),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])

    train_set = datasets.ImageFolder("G:/Course/Data_Science/keras-image-recognition-master-update/project/spiders/downloads/train",data_transform)
    test_set = datasets.ImageFolder("G:/Course/Data_Science/keras-image-recognition-master-update/project/spiders/downloads/test",data_transform)
    batch_size = 16
    dataloaders = DataLoader(train_set,batch_size,shuffle=True,num_workers=4)
    test_loaders = DataLoader(test_set,batch_size,shuffle=True,num_workers=4)
    test_loaders_size = len(test_loaders)
    dataloaders_size = len(dataloaders)
    epochs = 200

    ############  选择优化器和损失函数 ########################
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),1e-3)
    

    ############  开始训练和评估  #################################
    for epoch in range(epochs):
        model.train()
        for idx,(inputs,labels) in enumerate(dataloaders):
            inputs = inputs.to(device)
            labels = labels.to(device)
            out = model(inputs)
            loss = loss_criterion(out,labels)
            writer.add_scalar("train_loss",loss.item(),idx+epoch*dataloaders_size)
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
            print("epoch: {0}/{1}; batch:{2}/{3}; loss: {4}; accuracy:{5}".format(epoch,epochs,idx,test_loaders_size,loss.item(),1.0*correct_sum.item()/batch_size))
        if epoch %10 == 0:
            torch.save(model,"./dog_cat_horse_class/class_model.pth")


if __name__ == '__main__':
    main()



