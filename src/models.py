import torch
from torch import nn
import torch.nn.functional as F


class VariantCNN(nn.Module):
    def __init__(self, input_dim=5120, num_channels=32, output_dim=1, start_k_size=5):
        super(VariantCNN, self).__init__()
        
        # Reshape features into a format suitable for CNN
        # We'll treat the features as a 1D sequence with 32 channels
        #self.num_channels = 32
        
        self.num_channels = num_channels
        self.seq_length = input_dim // self.num_channels  # 5120 // 32 = 160
        
        # CNN layers
        self.conv1 = nn.Conv1d(self.num_channels, 64, kernel_size=start_k_size, stride=1, padding=int((start_k_size-1)/2))
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Reshape input from (batch_size, 5120) to (batch_size, 32, 160)
        x = x.view(-1, self.num_channels, self.seq_length)
        
        # CNN blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global average pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class Ensemble(nn.Module):   
    def __init__(self):
        super().__init__()
        self.modelA = VariantCNN(input_dim=5120)
        self.modelB = VariantCNN(input_dim=5120)
        self.modelC = VariantCNN(input_dim=5120, start_k_size=8)
        self.modelD = VariantCNN(input_dim=5120, start_k_size=10)
        self.modelE = VariantCNN(input_dim=5120, start_k_size=16)
        self.modelF = VariantCNN(input_dim=5120, start_k_size=32)
        self.modelG = VariantCNN(input_dim=5120, start_k_size=20)
        self.classifier = nn.Linear(7, 1)
        
    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        x3 = self.modelC(x)
        x4 = self.modelD(x)
        x5 = self.modelE(x)
        x6 = self.modelE(x)
        x7 = self.modelE(x)
        x = torch.cat((x1, x2, x3, x4, x5, x6, x7), dim=1)
        out = self.classifier(x)
        return out