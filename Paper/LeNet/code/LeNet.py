import torch
import torch.nn as nn


# LeNet Architecture 1x32x32 -> (5,5),s=1,p=0 -> avg pool s=2,p=0 -> (5,5) s=1,p=0 -> avg pool s=2,p=0
# -> conv 5x5 to 120 -> Linear 120 x Linear 80 x Linear 10

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.linear1 = nn.Linear(in_features=120, out_features=80)
        self.linear2 = nn.Linear(in_features=80, out_features=10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))  # [batch,120,1,1]
        print("shape of x ", x.shape)
        x = x.reshape(x.shape[0], -1)  # [batch,120]
        x = self.linear1(x)
        x = self.linear2(x)
        return x


x = torch.randn(64, 1, 32, 32)
model = LeNet()
print(model(x).shape)
