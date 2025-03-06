import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

# read and prepare data _____________________________________________________________________
X = np.loadtxt('C:/Users/Qba Liu/Documents/NAUKA_WLASNA/FEATURE_SELECTION_IDEA/LearnableFeatureSelection/PREPARATORY_STUFF/PYTORCH_RECAP/X_sim.txt', dtype = np.float32)
Y = np.loadtxt('C:/Users/Qba Liu/Documents/NAUKA_WLASNA/FEATURE_SELECTION_IDEA/LearnableFeatureSelection/PREPARATORY_STUFF/PYTORCH_RECAP/Y_sim.txt', dtype = np.float32)
X = torch.tensor(X)
Y = torch.tensor(Y).reshape(-1, 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)


# define the model architecture__________________________________________________________________

input_dim = X_train.shape[1]   # the number of features/columns

class NN_Model(nn.Module):

    # layers
    def __init__(self):
        super(NN_Model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)  # one output node since we have a prediction/regression task

    # architecture
    def forward(self, x):
            x = self.fc1(x)
            x = torch.relu(x)
            x = self.fc2(x)
            x = torch.relu(x)
            x = self.fc3(x)
            x = torch.relu(x)
            x = self.fc4(x)  # no activatuon function at the output
            return x




# define the custom loss function___________________________________________________________________________________________
class CustomLoss(nn.Module):
    def __init__(self, hyperparam):
        super(CustomLoss, self).__init__()
        self.hyperparam = hyperparam

    def forward(self, input, target):
        loss = torch.mean(self.hyperparam * (input - target) ** 2)
        return loss
    
"""
======================================= NOTES ABOUT THE CUSTOM LOSS FUNCTION =====================================================

class CustomLoss(nn.Module):
    def __init__(self, hyperparam):
        super(CustomLoss, self).__init__()
        self.hyperparam = hyperparam

    def forward(self, input, target):
        loss = torch.max(self.hyperparam * (input - target) ** 2)
        return loss

        
hyperparam --> a non-learnable hyperparameter
torch.max(self.hyperparam * (input - target) ** 2) --> input & target are vectors, so the get a scalar value for the loss we need
                                                        to apply an aggregation function (for example the max function).
"""

# define the training function (batch gradient descent)_______________________________________     

def train(model, X_train, X_test, Y_train, Y_test, loss_function, optimizer,num_epochs):
     
    global train_acc_array  # delcare as global so we can use it outside the function
    global test_acc_array
    train_acc_array = np.arange(num_epochs, dtype = np.float64)
    test_acc_array = np.arange(num_epochs, dtype = np.float64)

    for epoch in range(0, num_epochs):
          
          model.train()  # set model to training mode
          optimizer.zero_grad()   # remove all gradients
          Y_pred_train = model(X_train)  # make a prediction based on the training set
          loss = loss_function(Y_pred_train, Y_train)  # calculate the losss
          train_acc = loss  # save the training accuracy for the current epoch
          train_acc_array[epoch] = train_acc  # save to array for plotting
          loss.backward() # based on the loss use the chain rule to get the gradient
          optimizer.step()  # make a step based on the gradient

          model.eval()  # set model to evaluation mode
          Y_pred_test = model(X_test)  # make prediction based on the test set
          loss = loss_function(Y_pred_test, Y_test)  # calculate testing loss
          test_acc = loss  # save the testing accuracy for the current epoch
          test_acc_array[epoch] = test_acc  # save for plotting

          # print status for current epoch
          status = "epoch: {}\n training loss: {}\n testing loss: {}\n".format(epoch, train_acc, test_acc)
          print(status)



# train the model_________________________________________________________________________________________________________________
model = NN_Model()
hyperparam_ = 0.1
loss_fn = CustomLoss(hyperparam=hyperparam_)  # specify out custom loss function as the loss function
optim = optim.Adam(model.parameters())
n_epochs = 1000
train(model, X_train, X_test, Y_train, Y_test, loss_function = loss_fn, optimizer = optim, num_epochs = n_epochs)


plt.plot(range(0, n_epochs), train_acc_array, label="Train loss", color='blue', linestyle='-')
plt.plot(range(0, n_epochs), test_acc_array, label="Test loss", color='red', linestyle='--')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()