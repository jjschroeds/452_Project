import torch
import torch.nn as nn

class N_conv(nn.Module):
    """
    This a class for defining the N convolution
    Attributes
    ----------
    conv  : nn.Sequential
        defines the train model
    """
    def __init__(self, in_channels, out_channels, N = 2):
        super(N_conv,self).__init__()
        model = []
        model2 = []
        model3 = []
        model.append(nn.Conv2d(in_channels,out_channels,kernel_size=(3,3),padding=(1,1)))
        model.append(nn.ReLU(True))
        self.conv = nn.Sequential(*model)
        for i in range(N-1):
            model2.append(nn.Conv2d(out_channels,out_channels,kernel_size=(3, 3),padding=(1, 1)))
            model2.append(nn.ReLU(True))
        model3.append(nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)))
        self.conv = nn.Sequential(*model)
        self.conv2 = nn.Sequential(*model2)
        self.conv3 = nn.Sequential(*model3)
    def forward(self,x):
        x1 = self.conv(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2 + x1)
        return self.conv(x3)
class Vgg16(nn.Module):
    """
    This a class for defing the VGG16 model
    Attributes
    ----------
    conv  : nn.Sequential
        defines the train model
    """
    def __init__(self, init_weights=True):
        super(Vgg16, self).__init__()
        self.conv1 = N_conv(3,64)
        self.conv2 = N_conv(64,128)
        self.conv3 = N_conv(128,256,N=3)
        self.conv4 = N_conv(256,512,N=3)
        self.conv5 = N_conv(512,512,N=3)
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.linear1 = nn.Linear(51277,4096)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    with torch.nograd():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)


    model.train()
    return num_correct/num_samples