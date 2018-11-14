import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as mod

nclasses = 20 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5)
        self.fc1 = nn.Linear(2048, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #print(x.shape)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        #print(x.shape)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        #print(x.shape)
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        #print(x.shape)
        x = x.view(-1, 2048)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = mod.vgg16_bn(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.classifier[6] = torch.nn.Linear(4096, 20)
for param in model.classifier.parameters():
    param.requires_grad = True

for lay in model.features[40:]:
    for param in lay.parameters():
        param.requires_grad = True

model_new = model