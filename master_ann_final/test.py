import torch.nn.functional as F
import torch
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
