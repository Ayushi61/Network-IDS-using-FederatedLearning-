'''
Author: Benzil
Reference: https://ieeexplore-ieee-org.prox.lib.ncsu.edu/stamp/stamp.jsp?tp=&arnumber=6149943
https://arxiv.org/pdf/1804.07474.pdf

'''
import torch.nn as nn
import torch
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        input_dim: number of input features.
        output_dim: number of labels.
        """
        super(Net, self).__init__()
        #self.linear = nn.Linear(input_dim, output_dim)
        self.fc1 = nn.Linear(input_dim, 25)
        #self.dropout1 = torch.nn.Dropout(0.05)
        self.fc2 = nn.Linear(25,30)
        self.fc3=nn.Linear(30,output_dim)
        #self.dropout2 = torch.nn.Dropout(0.01)
        #self.dropout3 = torch.nn.Dropout(0.01)

    def forward(self, x):
        #outputs = self.linear(x)
        x=self.fc1(x)
        x=self.fc2(x)
        outputs=self.fc3(x)

        #x = x.view(-1, 33)
        #x = F.relu(self.fc1(x))
        #x = self.dropout1(x)
        #x = f.relu(self.fc2(x))
        return outputs
