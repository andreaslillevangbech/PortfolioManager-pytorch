# Cols are time and rows are coins
# Channels first

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, in_features, rows, cols, layers, device = torch.device("cpu")):
        super(CNN, self).__init__()

        out1 = 2
        out2 = 20
        kernel1 = (1,3)
        kernel2 = (1,cols-2) # cols - (kernel1[1] - 1)

        self.conv1 = nn.Conv2d(in_features, out1, kernel1)
        self.conv2 = nn.Conv2d(out1, out2, kernel2)
        self.votes = nn.Conv2d(out2+1, 1, (1,1)) # input features is out2 plus the appended last_weights
        # BTC bias
        b = torch.zeros((1,1))
        self.b = nn.Parameter(b)

    def forward(self, x, w):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = torch.cat((x, w),dim=1)
        x = self.votes(x)
        x = torch.squeeze(x)

        cash = self.b.repeat(x.size()[0], 1)
        x = torch.cat((cash, x), dim=1)
        return F.softmax(x, dim=1)

