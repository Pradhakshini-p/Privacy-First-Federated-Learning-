import torch
import torch.nn as nn
import torch.nn.functional as F

class FraudDetectionMLP(nn.Module):
    """
    Multi-Layer Perceptron for Credit Card Fraud Detection.
    Architecture optimized for tabular financial data.
    """
    
    def __init__(self, input_dim=30, hidden_dims=[128, 64, 32], output_dim=2, dropout_rate=0.3):
        super(FraudDetectionMLP, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build hidden layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)
    
    def predict_proba(self, x):
        """Return class probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities

class FraudDetectionCNN(nn.Module):
    """
    1D CNN for Fraud Detection.
    Useful when treating transaction features as sequential data.
    """
    
    def __init__(self, input_dim=30, output_dim=2):
        super(FraudDetectionCNN, self).__init__()
        
        # Reshape input for 1D CNN (batch, channels, length)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Calculate the flattened size after conv layers
        self._calculate_conv_output_size(input_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.conv_output_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, output_dim)
        )
    
    def _calculate_conv_output_size(self, input_dim):
        """Calculate output size after convolution layers."""
        # Create a dummy tensor to pass through conv layers
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            conv_output = self.conv_layers(dummy)
            self.conv_output_size = conv_output.view(1, -1).size(1)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Reshape for 1D CNN: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        
        # Pass through convolutional layers
        x = self.conv_layers(x)
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

def create_model(model_type="mlp", input_dim=30, **kwargs):
    """
    Factory function to create different model types.
    
    Args:
        model_type: "mlp" or "cnn"
        input_dim: Number of input features
        **kwargs: Additional model-specific arguments
    
    Returns:
        PyTorch model instance
    """
    if model_type.lower() == "mlp":
        return FraudDetectionMLP(input_dim=input_dim, **kwargs)
    elif model_type.lower() == "cnn":
        return FraudDetectionCNN(input_dim=input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Model testing function
def test_model():
    """Test the model with dummy data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test MLP
    print("Testing MLP model...")
    mlp_model = FraudDetectionMLP(input_dim=30).to(device)
    dummy_input = torch.randn(32, 30).to(device)
    
    with torch.no_grad():
        output = mlp_model(dummy_input)
        print(f"MLP output shape: {output.shape}")
        print(f"MLP output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Test CNN
    print("\nTesting CNN model...")
    cnn_model = FraudDetectionCNN(input_dim=30).to(device)
    
    with torch.no_grad():
        output = cnn_model(dummy_input)
        print(f"CNN output shape: {output.shape}")
        print(f"CNN output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    print("\n✅ Models are working correctly!")

if __name__ == "__main__":
    test_model()
