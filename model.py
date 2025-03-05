import torch
import torch.nn as nn
import torchvision.models as models

def create_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # Use pretrained weights
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Modify the final layer
    return model
