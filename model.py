import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as mod

nclasses = 20 

model = mod.resnet152(pretrained=True)
model.fc = torch.nn.Linear(2048, nclasses)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Make last block trainable
for param in model.layer4[2].parameters():
    param.requires_grad = True
