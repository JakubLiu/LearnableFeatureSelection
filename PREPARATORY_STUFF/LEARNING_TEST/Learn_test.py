import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchsummary import summary
from matplotlib import pyplot as plt

# read and prepare data _____________________________________________________________________
X = np.loadtxt('C:/Users/Qba Liu/Documents/NAUKA_WLASNA/FEATURE_SELECTION_IDEA/LearnableFeatureSelection/PREPARATORY_STUFF/LEARNING_TEST/X.txt', dtype = np.float32)
Y = np.loadtxt('C:/Users/Qba Liu/Documents/NAUKA_WLASNA/FEATURE_SELECTION_IDEA/LearnableFeatureSelection/PREPARATORY_STUFF/LEARNING_TEST/y.txt', dtype = np.float32)
X = torch.tensor(X)
Y = torch.tensor(Y).reshape(-1, 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


class CustomLinear(nn.Module):

    def __init__(self, *args, **kwargs):
        super(CustomLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(input_dim)) # initialize random weigths

    def binarize(self, weight):  # function that sets the weights to 0 or 1
        weight_bin = (weight >= 0).float()
        return (weight_bin - weight).detach() + weight # this is the STE procedure

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
model = Model_with_CustomLayer()
loss_fn = nn.MSELoss()
optim = optim.Adam(model.parameters())
n_epochs = 1000
train(model, X_train, X_test, Y_train, Y_test, loss_function = loss_fn, optimizer = optim, num_epochs = n_epochs)

plt.plot(np.arange(n_epochs), train_acc_array, color = 'blue', label = 'Train MSE')
plt.plot(np.arange(n_epochs), test_acc_array, color = 'red', label = 'Test MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.grid()
plt.legend()
plt.show()