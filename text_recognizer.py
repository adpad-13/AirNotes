import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32,5)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32,64,5)
        self.FC1 = nn.Linear(1024,256)
        self.FC2 = nn.Linear(256,47)
        self.flatten = nn.Flatten()
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.flatten(x)
        x = self.FC1(x)
        x = self.relu(x)
        x = self.FC2(x)
        return x
    
model = CNN()
dummy = torch.randn(1,1,28,28)
output = model(dummy)
print(output.shape)
