import torch
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils import data

print(torch.cuda.is_available())

# mnist_trainset = datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform)
# mnist_testset = datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)
#
# dataset = datasets.ImageFolder(root='./classify/dataset/training_set/', transform=transforms.ToTensor())
# loader = data.DataLoader(dataset, batch_size=8, shuffle=True)
#
#
# class NeuralNet(nn.Module):
#     def __init__(self):
#         super(NeuralNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.fc1 = nn.Linear(9216, 128)
#         self.fc2 = nn.Linear(128, 10)
#         x = nn.functional.max_pool2d(x, 2)
