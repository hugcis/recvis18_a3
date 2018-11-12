import zipfile
import os

import torchvision.transforms as transforms

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set
data_transforms = {
    'train': transforms.Compose([
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.RandomAffine(15, shear=15),
        transforms.RandomResizedCrop(224, scale=(0.5, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'sample': transforms.Compose([
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.RandomAffine(15, shear=15),
        transforms.RandomResizedCrop(224, scale=(0.5, 1)),
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