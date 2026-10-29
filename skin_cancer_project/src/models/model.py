import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes, model_name='efficientnet_b0', pretrained=True):
    """
    Creates an EfficientNet model with a custom classifier head.
    """
    if model_name == 'efficientnet_b0':
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        
        # Replace the classifier head
        # EfficientNet-B0 has 1280 input features for the classifier
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    else:
        raise ValueError(f"Model {model_name} not implemented.")
        
    return model

if __name__ == "__main__":
    # Test model creation
    m = get_model(num_classes=8)
    print(m)
    print("Model successfully created.")
