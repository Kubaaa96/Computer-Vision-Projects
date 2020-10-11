from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset 
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

# Hyperparameters
batch_size = 64
image_size = 64

# Data Preparation
transform = transforms.Compose([transforms.Scale(image_size), 
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), )])

data_set = dset.CIFAR10(root='./data', download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(data_set, 
                                          batch_size=batch_size, 
                                          shuffle = True, 
                                          num_workers=2)