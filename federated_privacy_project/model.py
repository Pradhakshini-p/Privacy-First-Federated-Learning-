import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """Simple feed-forward neural network for digit classification."""

    def __init__(self, input_dim: int = 64, hidden_dims=None, output_dim: int = 10):
        super(SimpleMLP, self).__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=0.3))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SimpleCNN(nn.Module):
    """Small CNN for digit-like image data."""

    def __init__(self, input_channels: int = 1, output_dim: int = 10):
        super(SimpleCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        return self.classifier(x)


def create_model(model_type: str = "mlp") -> nn.Module:
    """Factory for model selection."""
    model_type = model_type.lower()
    if model_type == "mlp":
        return SimpleMLP()
    if model_type == "cnn":
        return SimpleCNN()
    raise ValueError(f"Unknown model type '{model_type}'. Use 'mlp' or 'cnn'.")
