import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# read and prepare data _____________________________________________________________________
X = np.loadtxt('C:/Users/Qba Liu/Documents/NAUKA_WLASNA/FEATURE_SELECTION_IDEA/LearnableFeatureSelection/PREPARATORY_STUFF/PYTORCH_RECAP/X_sim.txt', dtype = np.float32)
Y = np.loadtxt('C:/Users/Qba Liu/Documents/NAUKA_WLASNA/FEATURE_SELECTION_IDEA/LearnableFeatureSelection/PREPARATORY_STUFF/PYTORCH_RECAP/Y_sim.txt', dtype = np.float32)
X = torch.tensor(X)
Y = torch.tensor(Y).reshape(-1, 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)