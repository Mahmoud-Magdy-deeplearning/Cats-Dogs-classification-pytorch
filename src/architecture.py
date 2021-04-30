import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 =  nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        criterion = nn.NLLLoss()

    def forward(self, x,labels):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.LogSoftmax(self.fc3(x),dim=1)
        return x
