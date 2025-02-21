import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchsummary import summary


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


# create a class of our custom layer__________________________________________________________________________________________
"""
Our custom layer will just be a masked fully connected layer.
In our case the mask is just an identity matrix.
(Since the input and output layers have the same number of nodes and each node is connected
only to its one corresponding node.)
"""
class CustomLinear(nn.Linear):
    def __init__(self, *args, mask, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = mask

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)*self.mask

# define the mask  
mask = np.eye(X_train.shape[1], dtype = np.int8)
mask = torch.from_numpy(mask)



# define the model architecture__________________________________________________________________

input_dim = X_train.shape[1]   # the number of features/columns


class Model_with_CustomLayer(nn.Module):
     
    def __init__(self):
        super(Model_with_CustomLayer, self).__init__()
        self.custom = CustomLinear(input_dim, input_dim, mask=mask) # this is the custom layers
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,1)

    def forward(self, x):
         x = self.custom(x)
         x = torch.relu(x)
         x = self.fc1(x)
         x = torch.relu(x)
         x = self.fc2(x)
         x = torch.relu(x)
         x = self.fc3(x)

         return(x)


# visualize the model __________________________________________________________________________________________________

custom_model = Model_with_CustomLayer()

summary(custom_model, (1, X_train.shape[1]))  