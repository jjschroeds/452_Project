import torch
import torch.nn as nn
import Vgg16 as model
import torchsummary
from torchsummary import summary
import torchvision
import torchvision.transforms as transforms
import tqdm
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 10

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(device)
learning_rate = 1e-4
num_epochs = 30
model = model.Vgg16().to(device)

summary(model, (3, 224, 224))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if __name__ == '__main__':
    print("beginning training...")
    for epoch in range(num_epochs):
        loop = tqdm(enumerate(trainloader), total=len(trainloader))
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
        print("\ntrain accuracy-"+str(model.check_accuracy(trainloader, model)))
        print("\ntest accuracy-" + str(model.check_accuracy(testloader, model)))
