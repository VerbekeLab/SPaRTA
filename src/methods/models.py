import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(
            self, 
            num_layers: int,
            input_size: int, 
            hidden_size: int, 
            output_size: int
            ):
        super().__init__()
        #self.flatten = nn.Flatten() # Flattens the 2D image into a 1D array
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.linear_relu_stack = nn.Sequential(
            self.layer1,
            nn.ReLU(),
            *[layer for hidden_layer in self.hidden_layers for layer in (hidden_layer, nn.ReLU())],
            self.output_layer
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 8, 3, padding=1) 
        self.conv2 = nn.Conv2d(8, 16, 2)
        self.fc = nn.Linear(16 * 2 * 2, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc(x)
        return x.squeeze()
    
class CNN_time(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1) 
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.fc = nn.Linear(16 * 6 * 5, 1)
        self.pool = nn.MaxPool2d(2, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc(x)
        return x.squeeze()