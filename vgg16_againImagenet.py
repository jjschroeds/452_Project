import numpy as np
import os
from torch.utils.data import Dataset
import torch
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

t1 = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),#torchvision.transforms.Resize(32),

    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    torchvision.transforms.RandomHorizontalFlip(p=0.25)
])



class N_conv2(nn.Module):
    """
    This a class for defining the N convolution
    Attributes
    ----------
    conv  : nn.Sequential
        defines the train model
    """

    def __init__(self, in_channels, out_channels, N=2):
        super(N_conv2, self).__init__()
        model = []
        model.append(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)))
        model.append(nn.ReLU(True))
        model.append(nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)))
        model.append(nn.ReLU(True))
        model.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        self.conv = nn.Sequential(*model)
    def forward(self, x):
        x = self.conv(x)
        return x

class N_conv3(nn.Module):
    """
    This a class for defining the N convolution
    Attributes
    ----------
    conv  : nn.Sequential
        defines the train model
    """

    def __init__(self, in_channels, out_channels, N=2):
        super(N_conv3, self).__init__()
        model = []
        model2 = []
        model3 = []
        model.append(nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)))
        model.append(nn.ReLU(True))
        self.conv = nn.Sequential(*model)
        for i in range(N - 1):
            model2.append(nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)))
            model2.append(nn.ReLU(True))
        self.conv2 = nn.Sequential(*model2)
        model3.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        self.conv3 = nn.Sequential(*model3)
    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2 + x1)
        return x3

class Vgg16(nn.Module):
    """
    This a class for defing the VGG16 model
    Attributes
    ----------
    conv  : nn.Sequential
        defines the train model
    """
    def __init__(self,in_channels=3,out_channels=1,init_weights=True):
        super(Vgg16,self).__init__()
        self.conv1 = N_conv2(3,64)
        self.conv2 = N_conv2(64,128)
        self.conv3 = N_conv3(128,256,N=3)
        self.conv4 = N_conv3(256,512,N=3)
        self.conv5 = N_conv3(512,512,N=3)
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.linear1 = nn.Linear(512*7*7,4096)
        self.linear2 = nn.Linear(4096,4096)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.3)
        self.linear3 = nn.Linear(4096,100)
        if init_weights:
            self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        return x

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)


    model.train()
    return num_correct/num_samples
def get_val_loss(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    loss = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            loss += nn.CrossEntropyLoss(scores, y)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
    loss = loss/len(loader)
    model.train()
    return loss,num_correct/num_samples
if __name__ == '__main__':
    #train_batch,test_batch = get_dogandcat(train_data_path,transforms=t1,batch_size=16,shuffle=True,pin_memory=True)
    train_batch, test_batch = get_imageNet1k(train_data_path,transforms=t1,batch_size=32,shuffle=True,pin_memory=True)

    for i,j in train_batch:
        img = np.transpose(i[0,:,:,:], (1,2,0))
        print(j)
        plt.imshow(img)
        plt.show()
        break
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    learning_rate = 1e-4
    num_epochs = 30
    model = Vgg16(3,2).to(device)
    from torchsummary import summary
    summary(model, (3, 224, 224))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min',factor = 0.1,patience = 4)
    print("beginning training...")
    for epoch in range(num_epochs):
        loop = tqdm(enumerate(train_batch), total=len(train_batch))
        loss = 0
        for batch_idx, (data, targets) in loop:
            data = data.to(device=device)
            targets = targets.to(device=device)

            # backward
            optimizer.zero_grad()

            scores = model(data)
            #print(scores)
            #print(targets)
            loss = criterion(scores, targets)
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            # Update Progress bar
            loop.set_description(f'Epoch [{epoch + 1}]')
            loop.set_postfix(loss=loss.item())
        val_loss, val_accuracy = get_val_loss(test_batch, model)
        scheduler.step(val_loss)
        print("current lr -" + str(optimizer.param_groups[0]['lr']))
        #print("\ntrain accuracy-"+str(check_accuracy(train_batch, model)))
        print("\ntest accuracy-" + str(val_accuracy))

