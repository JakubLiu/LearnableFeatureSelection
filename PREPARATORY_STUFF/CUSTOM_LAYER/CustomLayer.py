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
Our custom linear layer will not be a fully connected layer. It will have N input and N output neurons
and each neuron in the input layer will be connected to only one neuron in the output layer (look at the pdf).
There will be no biases (only weights). In addition the weights will be only binary (0 or 1).
"""
class CustomLinear(nn.Module):

    def __init__(self, *args, **kwargs):
        super(CustomLinear, self).__init__()
        self.weight = nn.Parameter(torch.ones(input_dim)) # initialize random weigths

    def binarize(self, weight):  # function that sets the weights to 0 or 1
        weight_bin = (weight >= 0).float()
        return weight_bin

    def forward(self, input):
        binary_weight = self.binarize(self.weight)  # binarize ({0,1}) the weights
        output =  input * binary_weight  # notice no bias
        return output




# define the model architecture__________________________________________________________________

input_dim = X_train.shape[1]   # the number of features/columns

# define a model with our custom layer
class Model_with_CustomLayer(nn.Module):
     
    def __init__(self):
        super(Model_with_CustomLayer, self).__init__()
        self.custom = CustomLinear(input_dim, input_dim) # this is the custom layers
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


# define a 'normal' model with only fully connected layers, for comparison
class StandardFCModel(nn.Module):
    def __init__(self):
        super(StandardFCModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim,32)
        self.fc3 = nn.Linear(32,1)

    def forward(self, x):
         x = self.fc1(x)
         x = torch.relu(x)
         x = self.fc2(x)
         x = torch.relu(x)
         x = self.fc3(x)

         return(x)


# visualize the model __________________________________________________________________________________________________

custom_model = Model_with_CustomLayer()
standard_model = StandardFCModel()

# compare the two models
summary(custom_model, (1, X_train.shape[1])) 
summary(standard_model, (1, X_train.shape[1]))



# output_______________________________________________________________________________________________________________-
"""

========================================= MODEL WITH CUSTOM LAYER  ==================================================

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
      CustomLinear-1               [-1, 1, 100]             100   
            Linear-2                [-1, 1, 64]           6,464
            Linear-3                [-1, 1, 32]           2,080
            Linear-4                 [-1, 1, 1]              33
================================================================
Total params: 8,677
Trainable params: 8,677
Non-trainable params: 0

As can be seen, the CustomLinear-1 has 100 parameters. This is as expected, since the input size is 100 (features)
and our CustomLinear-1 layer has 100 connections between the input and output nodes and one weights (and no bias)
per connection.



========================= STANDARD MODEL WIH ONLY FULLY CONNECTED LAYERS ================================================

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1               [-1, 1, 100]          10,100
            Linear-2                [-1, 1, 32]           3,232
            Linear-3                 [-1, 1, 1]              33
================================================================
Total params: 13,365
Trainable params: 13,365
Non-trainable params: 0

As can be seen although the Linear-1 layer has the same input dimensions as the CustomLinear-1 layer, it has much
more parameters since it is a fully connected layer. It has 100 connections per neuron (and there are 100 neurons
in the input and output laters), so 100x100 + 100(biases) = 10 100.
"""