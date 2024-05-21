import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

class NaiveFourierKANLayer(nn.Module):
    def __init__(self, inputdim, outdim, initial_gridsize, addbias=True):
        super(NaiveFourierKANLayer, self).__init__()
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim
        self.gridsize_param = nn.Parameter(torch.tensor(initial_gridsize, dtype=torch.float32))
        self.fouriercoeffs = nn.Parameter(torch.empty(2, outdim, inputdim, initial_gridsize))
        nn.init.xavier_uniform_(self.fouriercoeffs)
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))

    def forward(self, x):
        gridsize = torch.clamp(self.gridsize_param, min=1).round().int()
        outshape = x.shape[:-1] + (self.outdim,)
        x = torch.reshape(x, (-1, self.inputdim))
        k = torch.reshape(torch.arange(1, gridsize + 1, device=x.device), (1, 1, 1, gridsize))
        xrshp = torch.reshape(x, (x.shape[0], 1, x.shape[1], 1))
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)
        y = torch.sum(c * self.fouriercoeffs[0:1, :, :, :gridsize], (-2, -1))
        y += torch.sum(s * self.fouriercoeffs[1:2, :, :, :gridsize], (-2, -1))
        if self.addbias:
            y += self.bias
        y = torch.reshape(y, outshape)
        return y

class ConvFourierKANLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initial_gridsize=10, addbias=True):
        super(ConvFourierKANLayer, self).__init__()
        self.addbias = addbias
        self.stride = stride
        self.padding = (0, 0, padding, padding, padding, padding)
        self.gridsize_param = nn.Parameter(torch.tensor(initial_gridsize, dtype=torch.float32))
        self.fouriercoeffs = nn.Parameter(torch.empty(2, out_channels, in_channels, kernel_size, kernel_size, initial_gridsize))
        nn.init.xavier_uniform_(self.fouriercoeffs)
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
    
    def forward(self, x):
        gridsize = torch.clamp(self.gridsize_param, min=1).round().int()
        k = torch.reshape(torch.arange(1, gridsize + 1, device=x.device), (1, 1, 1, 1, gridsize))
        xrshp = torch.reshape(x, x.shape + (1,))
        c = F.pad(torch.cos(k * xrshp), self.padding)
        s = F.pad(torch.sin(k * xrshp), self.padding)
        y = F.conv3d(c, self.fouriercoeffs[0, :, :, :, :, :gridsize], stride = self.stride)
        y += F.conv3d(s, self.fouriercoeffs[1, :, :, :, :, :gridsize], stride = self.stride)
        y = torch.reshape(y, y.shape[:-1])
        if self.addbias:
            y += self.bias
        return y

class ConvKAN(nn.Module):
    def __init__(self):
        super(ConvKAN, self).__init__()
        self.conv1 = ConvFourierKANLayer(1, 13, kernel_size=5, padding=2, initial_gridsize=5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = ConvFourierKANLayer(13, 21, kernel_size=13, padding=6, initial_gridsize=8)
        self.pool2 = nn.MaxPool2d(2)
        self.fourierkan1 = NaiveFourierKANLayer(21*7*7, 89, initial_gridsize=13)
        self.fourierkan2 = NaiveFourierKANLayer(89, 10, initial_gridsize=8)
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fourierkan1(x)
        x = self.fourierkan2(x)
        return x

def eval():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    subset_indices = np.random.choice(len(train_dataset), int(len(train_dataset) * 0.1), replace=False)
    train_subset = Subset(train_dataset, subset_indices)
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ConvKAN().to(device)
    opt = torch.optim.LBFGS(net.parameters(), lr = .001)
    for epoch in range(5):
        net.train()
        for i, (image, label) in enumerate(train_loader):
            def closure():
                opt.zero_grad()
                output = net(image.to(device))
                loss = nn.CrossEntropyLoss()(output, label.to(device))
                loss.backward()
                return loss
            opt.step(closure)
            if i % 10 == 0:
                loss = closure()
                print(f'Train Epoch: {epoch} [{i * len(image)}/{len(train_loader.dataset)} ({100. * i / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    net.eval()
    with torch.no_grad():
        acc, tot = 0, 0
        for image, label in test_loader:
            output = net(image.to(device))
            _, predicted = torch.max(output.data, 1)
            tot += label.size(0)
            acc += (predicted == label.to(device)).sum().item()
    print('Test Accuracy: {:.2f}%'.format(100 * acc / tot))

if __name__ == '__main__':
    eval()
