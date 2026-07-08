import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(
            self,
            num_layers: int,
            input_size: int,
            hidden_size: int,
            output_size: int,
            dropout: float = 0.0
            ):
        super().__init__()
        # dropout defaults to 0.0 so the static-features path (experiment_features.py), which
        # does not pass it, is bit-identical to before: at 0.0 NO Dropout layer is inserted, so
        # the module list and its forward RNG stream are exactly the old ones. The baseline MLP
        # tunes dropout via Optuna, adding a Dropout after each ReLU only when it is > 0.
        #self.flatten = nn.Flatten() # Flattens the 2D image into a 1D array
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, output_size)
        # Build the stack explicitly so a fresh Dropout follows each ReLU only when dropout > 0
        # (at 0.0 the list is exactly [Linear, ReLU, (Linear, ReLU)*, Linear] as before).
        layers = [self.layer1, nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        for hidden_layer in self.hidden_layers:
            layers.extend([hidden_layer, nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(self.output_layer)
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
class CNN(nn.Module):
    def __init__(
            self, 
            num_channels=2, 
            hidden_channels=8, 
            num_layers=3,
            kernel_size=3, 
            max_pool=False,
            half_final_layer=True):
        super().__init__()

        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.conv_layers.append(nn.Conv2d(num_channels, hidden_channels, kernel_size, padding='same'))
            elif (i == num_layers - 1) and half_final_layer:
                self.conv_layers.append(nn.Conv2d(hidden_channels, hidden_channels//2, kernel_size, padding='same'))
            else:
                self.conv_layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding='same'))

        self.pool = nn.MaxPool2d(2, 1) if max_pool else nn.Identity()

        if max_pool:
            if half_final_layer:
                self.fc = nn.Linear(hidden_channels//2 * 2 * 2, 1)
            else:
                self.fc = nn.Linear(hidden_channels * 2 * 2, 1)

        else:
            if half_final_layer:
                self.fc = nn.Linear(hidden_channels//2 * 3 * 3, 1)
            else:
                self.fc = nn.Linear(hidden_channels * 3 * 3, 1)

    def forward(self, x):
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        x = self.pool(x)
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

# --- Time-series models (SPaRTA snapshots) --------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class LSTMClassifier(nn.Module):
    def __init__(self, n_features=90, hidden=32, num_layers=1, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers, batch_first=True,
                            dropout=(dropout if num_layers > 1 else 0.0))
        self.drop = nn.Dropout(dropout)           # regularize the last-step hidden state
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, mask):                  # x:(B,T,F)  mask:(B,T) bool
        out, _ = self.lstm(x)                     # (B,T,H); padded steps fed as zeros
        last = mask.float().cumsum(1).argmax(1)   # index of the last valid (True) step
        idx = last.view(-1, 1, 1).expand(-1, 1, out.size(-1))
        h = out.gather(1, idx).squeeze(1)         # (B,H) hidden state at last valid step
        return self.head(self.drop(h)).squeeze(-1)


class TransformerClassifier(nn.Module):
    def __init__(self, n_features=90, head_dim=8, nhead=4, num_layers=1,
                 dim_feedforward=None, dropout=0.2):
        super().__init__()
        d_model = head_dim * nhead                # ensures d_model % nhead == 0
        self.proj = nn.Linear(n_features, d_model)
        self.pe = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead,
                                           dim_feedforward=(dim_feedforward or 2 * d_model),
                                           dropout=dropout, batch_first=True)
        # enable_nested_tensor=False: the padding-mask nested-tensor fast path uses an op
        # (_nested_tensor_from_mask_left_aligned) unimplemented on MPS, so eval() forward
        # crashes on Apple Silicon. Disabling it costs nothing for tiny K and runs everywhere.
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.drop = nn.Dropout(dropout)                   # regularize the pooled representation
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, mask):                       # mask:(B,T) True = valid
        h = self.pe(self.proj(x))
        h = self.encoder(h, src_key_padding_mask=~mask)   # True = ignore this position
        m = mask.float().unsqueeze(-1)
        h = (h * m).sum(1) / m.sum(1).clamp(min=1)        # mean-pool over valid tokens
        return self.head(self.drop(h)).squeeze(-1)