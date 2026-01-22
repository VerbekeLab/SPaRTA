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
    def __init__(self, num_channels=2):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 8, 3, padding=1) 
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
    
class CNN_visual_VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding='same')
        self.conv2 = nn.Conv2d(64, 64, 3, padding='same')
        self.conv3 = nn.Conv2d(64, 128, 3, padding='same')
        self.conv4 = nn.Conv2d(128, 128, 3, padding='same')
        self.conv5 = nn.Conv2d(128, 256, 3, padding='same')
        self.conv6 = nn.Conv2d(256, 256, 3, padding='same')
        self.conv7 = nn.Conv2d(256, 256, 3, padding='same')
        self.conv8 = nn.Conv2d(256, 512, 3, padding='same')
        self.conv9 = nn.Conv2d(512, 512, 3, padding='same')
        self.conv10 = nn.Conv2d(512, 512, 3, padding='same')
        self.conv11 = nn.Conv2d(512, 512, 3, padding='same')
        self.conv12 = nn.Conv2d(512, 512, 3, padding='same')
        self.conv13 = nn.Conv2d(512, 512, 3, padding='same')
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 7 * 7, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.pool(x)

        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.pool(x)

        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = self.pool(x)

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x.squeeze()