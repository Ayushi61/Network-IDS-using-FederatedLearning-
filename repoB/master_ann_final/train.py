'''
Author : Benzil
reference :https://arxiv.org/pdf/1804.07474.pdf
'''
import torch.nn.functional as F

def train(model, device, federated_train_loader, optimizer, epoch,BATCH_SIZE,LOG_INTERVAL):
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
    return model
