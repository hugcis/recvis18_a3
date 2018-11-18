import zipfile
import os

import torchvision.transforms as transforms

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set
data_transforms = {
    'train': transforms.Compose([
        transforms.ColorJitter(brightness=0.6, contrast=0.6),
        transforms.RandomAffine(30, shear=30, scale=(0.6, 1.2)),
        transforms.RandomResizedCrop(224, scale=(0.3, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Create a transformation without normalization for visualization
    'sample': transforms.Compose([
        transforms.ColorJitter(brightness=0.6, contrast=0.6),
        transforms.RandomAffine(30, shear=30, scale=(0.6, 1.2)),
        transforms.RandomResizedCrop(224, scale=(0.3, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}