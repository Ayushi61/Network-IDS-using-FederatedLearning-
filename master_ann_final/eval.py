import torch.optim as optim
from model import *
from train import *
from test import *
import torch
import syft as sy
import numpy as np
import torch.nn.functional as F
from syft.frameworks.torch.fl import utils
from syft.workers.websocket_client import WebsocketClientWorker
import os
import datetime
def run_train_test(workers,ports,ids,X_train, X_test, y_train, y_test,encoder):
    # Hook PyTorch ie add extra functionalities to support Federated Learning
    hook = sy.TorchHook(torch)
    # Sets the seed for generating random numbers.
    torch.manual_seed(1)
    # Select CPU computation, in case you want GPU use "cuda" instead
    device = torch.device("cpu")
    # Data will be distributed among these VirtualWorkers.
    # Remote training of the model will happen here.
    
    kwargs_websocket_bob = {"host": workers[1], "hook": hook}
    #gatway2 = sy.VirtualWorker(hook, id="gatway2")
    gatway2 = WebsocketClientWorker(id=ids[1], port=ports[1], **kwargs_websocket_bob)
    
    kwargs_websocket_alice = {"host": workers[0], "hook": hook}
    #gatway1 = sy.VirtualWorker(hook, id="gatway1")
    gatway1= WebsocketClientWorker(id=ids[0], port=ports[0], **kwargs_websocket_alice)
    #kwargs_websocket_bob2 = {"host": workers[2], "hook": hook}
    #gatway2 = sy.VirtualWorker(hook, id="gatway2")
    #gatway3 = WebsocketClientWorker(id=ids[2], port=ports[2], **kwargs_websocket_bob2)
    
    #kwargs_websocket_alice2 = {"host": workers[3], "hook": hook}
    #gatway1 = sy.VirtualWorker(hook, id="gatway1")
    #gatway4= WebsocketClientWorker(id=ids[3], port=ports[3], **kwargs_websocket_alice2)
    # Number of times we want to iterate over whole training data
    #gatway3=sy.VirtualWorker(hook, id="gatway3")
    #gatway4=sy.VirtualWorker(hook, id="gatway4")
    BATCH_SIZE = 1000
    EPOCHS = 5
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
    gatway2_train_dataset = sy.BaseDataset(train_inputs[train_idx:2*train_idx], train_labels[train_idx:2*train_idx]).send(gatway2)
    #gatway3_train_dataset = sy.BaseDataset(train_inputs[2*train_idx:3*train_idx], train_labels[2*train_idx:3*train_idx]).send(gatway3)
    #gatway4_train_dataset = sy.BaseDataset(train_inputs[3*train_idx:], train_labels[3*train_idx:]).send(gatway4)
    gatway1_test_dataset = sy.BaseDataset(test_inputs[:test_idx], test_labels[:test_idx]).send(gatway1)
    gatway2_test_dataset = sy.BaseDataset(test_inputs[test_idx:2*test_idx], test_labels[test_idx:2*test_idx]).send(gatway2)
    #gatway3_test_dataset = sy.BaseDataset(test_inputs[2*test_idx:2*test_idx], test_labels[2*test_idx:3*test_idx]).send(gatway3)
    #gatway4_test_dataset = sy.BaseDataset(test_inputs[3*test_idx:], test_labels[3*test_idx:]).send(gatway3)
    
    # Create federated datasets, an extension of Pytorch TensorDataset class
    federated_train_dataset = sy.FederatedDataset([gatway1_train_dataset, gatway2_train_dataset)]#,gatway3_train_dataset,gatway4_train_dataset])
    federated_test_dataset = sy.FederatedDataset([gatway1_test_dataset, gatway2_test_dataset)]#,gatway3_test_dataset,gatway4_test_dataset])
    
    # Create federated dataloaders, an extension of Pytorch DataLoader class
    federated_train_loader = sy.FederatedDataLoader(federated_train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    federated_test_loader = sy.FederatedDataLoader(federated_test_dataset, shuffle=False, batch_size=BATCH_SIZE)
    # Initialize the model
    model = Net(n_feature,n_class)
    
    #Initialize the SGD optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    for epoch in range(1, EPOCHS + 1):
        # Train on the training data in a federated way
        start1=datetime.datetime.now()
        model=train(model, device, federated_train_loader, optimizer, epoch,BATCH_SIZE,LOG_INTERVAL)
        end1=datetime.datetime.now()
        print("Time Taken to train epoch %d is ",end1-start1)
        if epoch==1:
            total=end1-start1
        else:
            total+=end1-start1
        # Check the test accuracy on unseen test data in a federated way
        start2=datetime.datetime.now()
        test(model, device, federated_test_loader)
        end2=datetime.datetime.now()
        print("Time Taken to test epoch %d is ",end2-start2)
    print("Total training time = ",total)
    
    # Save the model
    torch.save(model.state_dict(), "binaize-threat-model_10.pt")
    # Reload the model in a new model object
    model_new = Net(n_feature,n_class)
    model_new.load_state_dict(torch.load("binaize-threat-model_10.pt"))
    model_new.eval()
    process = os.popen("sudo scp -i /root/.ssh/id_rsa -o stricthostkeychecking=no binaize-threat-model_fully_trained.pt root@%s:/home/ayush/ADS/predict_workers" %(workers[0]))
    output=process.read()
    process = os.popen("sudo scp -i /root/.ssh/id_rsa -o stricthostkeychecking=no binaize-threat-model_fully_trained.pt root@%s:/home/ayush/ADS/predict_workers" %(workers[1]))
    output=process.read()
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
