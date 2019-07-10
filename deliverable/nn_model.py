import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

class JobPerformance(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, dropout_percentage):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dropout_percentage = dropout_percentage
        
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=dropout_percentage)
 
        
    def forward(self, x):
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        # No dropout on output
        x = self.output(x)
        
        return x

def validation(model, test_loader, criterion):
    accuracy = 0
    test_loss = 0
    for images, labels in test_loader:

#         images = images.resize_(images.size()[0], 784)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        equals = output - labels
        accuracy += torch.mean(equals.type(torch.FloatTensor))

    return test_loss, accuracy

# def predict(model,inputs):
#     model.eval()
#     with torch.no_grad():
#         outputs = model.forward(inputs)

#     return outputs


def train(model, train_loader, test_loader, criterion1, criterion2, optimizer, epochs=5, print_every=40):
    steps = 0
    train_losses, test_losses = [], []
    
    # Make sure dropout and grads are on for training
    model.train()
    for i in range(epochs):
        train_loss = 0
        for images, labels in train_loader:
            steps += 1
            
            # Flatten images into a 784 long vector
#             images.resize_(images.size()[0], 784)
            
            optimizer.zero_grad()
            
            output = model.forward(images)
            loss = criterion1(output, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        else:
            # Model in inference mode, dropout is off
            model.eval()

            # Turn off gradients for validation, will speed up inference
            with torch.no_grad():
                test_loss, accuracy = validation(model, test_loader, criterion2)
            
            train_losses.append(train_loss/len(train_loader))
            test_losses.append(test_loss/len(test_loader))
            
            print("Epoch:",i+1)
            print(f"  Train Loss (MSE): {train_loss/len(train_loader):>12,.1f}")
            print(f"  Test Loss  (MSE): {test_loss/len(test_loader):>12,.1f}")
            print(f"  Accuracy: {accuracy/len(test_loader):>20,.1f}")

            # Make sure dropout and grads are on for training
            model.train()
    
    return train_losses, test_losses


def predict(model, test_loader):
    model.eval()
    with torch.no_grad():
        current_output = torch.tensor([])
        output_tensor = torch.tensor([])

        for rows, labels in test_loader:
            current_output = model.forward(rows)
            output_tensor = torch.cat((output_tensor,current_output),0)

    return output_tensor

