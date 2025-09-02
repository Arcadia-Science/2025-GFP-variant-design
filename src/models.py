import torch
from torch import nn
import torch.nn.functional as F


class VariantCNN(nn.Module):
    def __init__(self, input_dim=5120, num_channels=32, output_dim=1, start_k_size=5):
        """
        A Convolutional Neural Network (CNN) model for predicting variant effects from ESM-2 embeddings.
        Args:
            input_dim (int): Dimension of the input features (default: 5120 for ESM-2 15b parameter embeddings).
            num_channels (int): Number of channels to reshape the input features into (default: 32).
            output_dim (int): Dimension of the output (default: 1 for regression).
            start_k_size (int): Kernel size for the first convolutional layer (default: 5).
        """
        super(VariantCNN, self).__init__()
        self.num_channels = num_channels

        # Reshape features into a format suitable for CNN
        # Convert flat ESM-2 embeddings (5120-dim) into a 2D tensor (32 channels × 160 sequence length)
        # This allows the CNN to capture local patterns and spatial relationships within the embedding space
        # 32 channels provides a good balance between preserving information and computational efficiency
        self.seq_length = input_dim // self.num_channels  # 5120 // 32 = 160

        # CNN layers
        self.conv1 = nn.Conv1d(
            self.num_channels,
            64,
            kernel_size=start_k_size,
            stride=1,
            padding=int((start_k_size - 1) / 2),
        )
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
        # This converts the flat ESM-2 embeddings into the multi-channel format expected by Conv1d layers
        # Each of the 32 channels contains a 160-length sequence that the CNN can process spatially
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
        # The first two CNNs are the same size to place more focus locally.
        self.modelA = VariantCNN(input_dim=5120, start_k_size=5)
        self.modelB = VariantCNN(input_dim=5120, start_k_size=5)
        self.modelC = VariantCNN(input_dim=5120, start_k_size=8)
        self.modelD = VariantCNN(input_dim=5120, start_k_size=10)
        self.modelE = VariantCNN(input_dim=5120, start_k_size=16)
        self.classifier = nn.Linear(5, 1)

    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        x3 = self.modelC(x)
        x4 = self.modelD(x)
        x5 = self.modelE(x)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        out = self.classifier(x)
        return out
