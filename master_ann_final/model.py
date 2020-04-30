import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        input_dim: number of input features.
        output_dim: number of labels.
        """
        super(Net, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        #self.fc1 = torch.nn.Linear(input_dim, 1024)
        #self.dropout1 = torch.nn.Dropout(0.01)
        #self.fc2 = torch.nn.Linear(1024, 768)
        #self.dropout2 = torch.nn.Dropout(0.01)
        #self.fc4 = torch.nn.Linear(768, 512)
        #self.dropout3 = torch.nn.Dropout(0.01)
        #self.fc3 = torch.nn.Linear(512, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        #x = x.view(-1, 33)
        #x = F.relu(self.fc1(x))
        #x = self.dropout1(x)
        #x = F.relu(self.fc2(x))
        #x = self.dropout2(x)
        #x = F.relu(self.fc4(x))
        #x = self.dropout3(x)
        #x = F.sigmoid(self.fc3(x))
        return outputs
