import torch
import torch.nn as nn

class OrientationCNN(nn.Module):
    """CNN for predicting agent orientation as continuous angle"""
    
    def __init__(self, input_size=64):
        """Initialize orientation CNN
        
        Args:
            input_size: Input image size (assumed square)
        """
        super(OrientationCNN, self).__init__()
        
        # Feature extraction (similar to classification CNN)
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Calculate size after convolutions
        conv_output_size = input_size // 8
        flatten_size = 128 * conv_output_size * conv_output_size
        
        # Regression head for angle prediction
        self.regressor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(flatten_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)  # Output: [sin(θ), cos(θ)]
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        # Normalize to unit circle
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x

