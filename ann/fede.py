import pandas as pd
pd.set_option("display.max_columns", 10)
import plotly.graph_objects as go
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import syft as sy
import numpy as np
import torch.nn.functional as F
from syft.frameworks.torch.fl import utils
from syft.workers.websocket_client import WebsocketClientWorker
#matplotlib.use('GTK')
# We use the KDD CUP 1999 data (https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
# 41 column names can be found at https://kdd.ics.uci.edu/databases/kddcup99/kddcup.names
colnames = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
            'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
            'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
            'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate']

# We take 10% of the original data which can be found at 
# http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz
# We select the first 100K records from this data
df = pd.read_csv("http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz",
        names=colnames+["threat_type"])#[:100000]

print(df.head(3))
threat_count_dict = Counter(df["threat_type"])
threat_types = list(threat_count_dict.keys())
threat_counts = [threat_count_dict[threat_type] for threat_type in threat_types]
print("Total distinct number of threat types : ",len(threat_types))
#fig = go.Figure([go.Bar(x=threat_types, y=threat_counts,text=threat_counts,textposition='auto')])
#fig.show()
numerical_colmanes = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot',
                      'num_failed_logins', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
                      'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'count',
                      'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                      'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                      'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                      'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                      'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']
numerical_df = df[numerical_colmanes].copy()
# Lets remove the numerical columns with constant value
numerical_df = numerical_df.loc[:, (numerical_df != numerical_df.iloc[0]).any()]
# lets scale the values for each column from [0,1]
# N.B. we dont have any negative values]
final_df = numerical_df/numerical_df.max()
X = final_df.values
# final dataframe has 33 features
print("Shape of feature matrix : ",X.shape)
threat_types = df["threat_type"].values
encoder = LabelEncoder()
# use LabelEncoder to encode the threat types in numeric values
y = encoder.fit_transform(threat_types)
print("Shape of target vector : ",y.shape)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4, random_state=42, stratify=y)
print("Number of records in training data : ", X_train.shape[0])
print("Number of records in test data : ", X_test.shape[0])
print("Total distinct number of threat types in training data : ",len(set(y_train)))
print("Total distinct number of threat types in test data : ",len(set(y_test)))

# Hook PyTorch ie add extra functionalities to support Federated Learning
hook = sy.TorchHook(torch)
# Sets the seed for generating random numbers.
torch.manual_seed(1)
# Select CPU computation, in case you want GPU use "cuda" instead
device = torch.device("cpu")
# Data will be distributed among these VirtualWorkers.
# Remote training of the model will happen here.

kwargs_websocket_bob = {"host": "10.128.0.16", "hook": hook}
#gatway2 = sy.VirtualWorker(hook, id="gatway2")
gatway2 = WebsocketClientWorker(id="bob", port=8778, **kwargs_websocket_bob)

kwargs_websocket_alice = {"host": "10.128.0.12", "hook": hook}
#gatway1 = sy.VirtualWorker(hook, id="gatway1")
gatway1= WebsocketClientWorker(id="alice", port=8777, **kwargs_websocket_alice)
# Number of times we want to iterate over whole training data
BATCH_SIZE = 1000
EPOCHS = 2
LOG_INTERVAL = 5
lr = 0.01

n_feature = X_train.shape[1]
n_class = np.unique(y_train).shape[0]

print("Number of training features : ",n_feature)
print("Number of training classes : ",n_class)
# Create pytorch tensor from X_train,y_train,X_test,y_test
train_inputs = torch.tensor(X_train,dtype=torch.float).tag("#iot", "#network","#data","#train")
train_labels = torch.tensor(y_train).tag("#iot", "#network","#target","#train")
test_inputs = torch.tensor(X_test,dtype=torch.float).tag("#iot", "#network","#data","#test")
test_labels = torch.tensor(y_test).tag("#iot", "#network","#target","#test")

# Send the training and test data to the gatways in equal proportion.
train_idx = int(len(train_labels)/2)
test_idx = int(len(test_labels)/2)
gatway1_train_dataset = sy.BaseDataset(train_inputs[:train_idx], train_labels[:train_idx]).send(gatway1)
gatway2_train_dataset = sy.BaseDataset(train_inputs[train_idx:], train_labels[train_idx:]).send(gatway2)
gatway1_test_dataset = sy.BaseDataset(test_inputs[:test_idx], test_labels[:test_idx]).send(gatway1)
gatway2_test_dataset = sy.BaseDataset(test_inputs[test_idx:], test_labels[test_idx:]).send(gatway2)

# Create federated datasets, an extension of Pytorch TensorDataset class
federated_train_dataset = sy.FederatedDataset([gatway1_train_dataset, gatway2_train_dataset])
federated_test_dataset = sy.FederatedDataset([gatway1_test_dataset, gatway2_test_dataset])

# Create federated dataloaders, an extension of Pytorch DataLoader class
federated_train_loader = sy.FederatedDataLoader(federated_train_dataset, shuffle=True, batch_size=BATCH_SIZE)
federated_test_loader = sy.FederatedDataLoader(federated_test_dataset, shuffle=False, batch_size=BATCH_SIZE)

import torch.nn as nn
class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        input_dim: number of input features.
        output_dim: number of labels.
        """
        super(Net, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
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

import torch.nn.functional as F

def train(model, device, federated_train_loader, optimizer, epoch):
    model.train()
    # Iterate through each gateway's dataset
    for idx, (data, target) in enumerate(federated_train_loader):
        batch_idx = idx+1
        # Send the model to the right gateway
        model.send(data.location)
        # Move the data and target labels to the device (cpu/gpu) for computation
        data, target = data.to(device), target.to(device)
        # Clear previous gradients (if they exist)
        optimizer.zero_grad()
        # Make a prediction
        output = model(data)
        # Calculate the cross entropy loss [We are doing classification]
        loss = F.cross_entropy(output, target)
        # Calculate the gradients
        loss.backward()
        # Update the model weights
        optimizer.step()
        # Get the model back from the gateway
        model.get()
        if batch_idx==len(federated_train_loader) or (batch_idx!=0 and batch_idx % LOG_INTERVAL == 0):
            # get the loss back
            loss = loss.get()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * BATCH_SIZE, len(federated_train_loader) * BATCH_SIZE,
                100. * batch_idx / len(federated_train_loader), loss.item()))
import torch.nn.functional as F

def test(model, device, federated_test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(federated_test_loader):
            # Send the model to the right gateway
            model.send(data.location)
            # Move the data and target labels to the device (cpu/gpu) for computation
            data, target = data.to(device), target.to(device)
            # Make a prediction
            output = model(data)
            # Get the model back from the gateway
            model.get()
            # Calculate the cross entropy loss
            loss = F.cross_entropy(output, target)
            # Get the index of the max log-probability 
            pred = output.argmax(1, keepdim=True)
            # Get the number of instances correctly predicted
            correct += pred.eq(target.view_as(pred)).sum().get()
                
    # get the loss back
    loss = loss.get()
    print('Test set: Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        loss.item(), correct, len(federated_test_loader.federated_dataset),
        100. * correct / len(federated_test_loader.federated_dataset)))
import torch.optim as optim

# Initialize the model
model = Net(n_feature,n_class)

#Initialize the SGD optimizer
optimizer = optim.SGD(model.parameters(), lr=lr)

for epoch in range(1, EPOCHS + 1):
    # Train on the training data in a federated way
    train(model, device, federated_train_loader, optimizer, epoch)
    # Check the test accuracy on unseen test data in a federated way
    test(model, device, federated_test_loader)

# Save the model
torch.save(model.state_dict(), "binaize-threat-model.pt")
# Reload the model in a new model object
model_new = Net(n_feature,n_class)
model_new.load_state_dict(torch.load("binaize-threat-model.pt"))
model_new.eval()

# Take the 122th record from the test data
idx = 122
data = test_inputs[idx]
pred = model_new(data)
pred_label = int(pred.argmax().data.cpu().numpy())
pred_threat = encoder.inverse_transform([pred_label])[0]
print("Predicted threat type : ", pred_threat)
actual_label = int(test_labels[idx].data.cpu().numpy())
actual_threat = encoder.inverse_transform([actual_label])[0]
print("Actual threat type : ", actual_threat)


# Take the 159th record from the test data
idx = 159
data = test_inputs[idx]
pred = model_new(data)
pred_label = int(pred.argmax().data.cpu().numpy())
pred_threat = encoder.inverse_transform([pred_label])[0]
print("Predicted threat type : ", pred_threat)
actual_label = int(test_labels[idx].data.cpu().numpy())
actual_threat = encoder.inverse_transform([actual_label])[0]
print("Actual threat type : ", actual_threat)
