import torch
from torch import nn
import numpy as np

def weight_init(m):

    if (isinstance(m,nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif (isinstance(m,nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    # print(m)

class MainNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.p_layer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.MaxPool2d(3, 3),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(3, 3),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.Conv2d(128, 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 256, 3, 1, groups=256),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(256, 128, 1, 1),
            nn.BatchNorm2d(128),
            nn.PReLU()
        )
        self.liner1 = nn.Linear(128 * 3 * 3, 128)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.liner2 = nn.Linear(128, 1)
        self.liner3 = nn.Linear(128 * 3 * 3, 512)
        self.liner4 = nn.Linear(512, 200)
        self.apply(weight_init)

    def forward(self, x):
        c_out = self.p_layer(x)
        c_out = c_out.view(c_out.size(0), -1)
        y1 = self.liner1(c_out)
        y1 = self.prelu1(y1)
        cls = torch.sigmoid(self.liner2(y1))
        y2 = self.liner3(c_out)
        y2 = self.prelu2(y2)
        offset = self.liner4(y2)

        return cls,offset

class P_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.p_layer = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, stride=1),
            # nn.BatchNorm2d(10),
            nn.PReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(10, 16, kernel_size=3, stride=1),
            # nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.PReLU()
        )
        self.conv1 = nn.Conv2d(32, 1, 1, 1)
        self.conv2 = nn.Conv2d(32, 14, 1, 1)
        self.apply(weight_init)

    def forward(self, x):
        y = self.p_layer(x)
        cls = torch.sigmoid(self.conv1(y))
        offset = self.conv2(y)
        return cls, offset

if __name__ == '__main__':
    img = np.random.random([10,3,128,128])
    print(img.shape)
    img = torch.tensor(img,dtype=torch.float32)
    net = MainNet()
    x,y = net(img)
    print(x.shape,y.shape)