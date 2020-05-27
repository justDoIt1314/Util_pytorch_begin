from __future__ import print_function,division
import torch
from torch import optim,nn,device
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import numpy as np
import torchvision
from torchvision import datasets,models,transforms
import matplotlib.pyplot as plt
from transfer_model import imshow,visualize_model
import os


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                inp = inputs.cpu().data[j].numpy().transpose((1,2,0))
                mean = np.array([0.485,0.456,0.406])
                std = np.array([0.229,0.224,0.225])
                inp = std*inp + mean
                inp = np.clip(inp,0,1)
                
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                plt.imshow(inp)
                plt.pause(0.1)
                #imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    plt.savefig('fig.jpg')
                    return
        model.train(mode=was_training)

if __name__ == '__main__':
    data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    device = device("cuda:0" if torch.cuda.is_available() else 'cpu')
    data_dir = "G:/Course/Algorithm/Util/Transfer_Learning/hymenoptera_data/"
    image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in ['train','val']}
    dataloaders = {x:DataLoader(image_datasets[x],batch_size=20,shuffle=True,num_workers=4) for x in ['train','val']}
    class_names = image_datasets['train'].classes
    model_ft = torch.load('binClass.pth')
    visualize_model(model_ft)