import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

# read and prepare data _____________________________________________________________________
X = np.loadtxt('C:/Users/Qba Liu/Documents/NAUKA_WLASNA/FEATURE_SELECTION_IDEA/LearnableFeatureSelection/PREPARATORY_STUFF/LEARNING_TEST/X.txt', dtype = np.float32)
Y = np.loadtxt('C:/Users/Qba Liu/Documents/NAUKA_WLASNA/FEATURE_SELECTION_IDEA/LearnableFeatureSelection/PREPARATORY_STUFF/LEARNING_TEST/y.txt', dtype = np.float32)
X = torch.tensor(X)
Y = torch.tensor(Y).reshape(-1, 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)



# build the custom model _____________________________________________________________________________________
input_dim = X_train.shape[1]

class CustomLinear(nn.Module):

    def __init__(self, input_dim, output_dim, cutoff = 0.5, *args, **kwargs):
        super(CustomLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(input_dim)) # initialize random weigths
        self.cutoff = cutoff

    def binarize(self, weight):  # function that sets the weights to 0 or 1
        weight_bin = (torch.abs(weight) >= self.cutoff).float() # set the weight to 0 if its abs() is lower than 0.5, else set the weight to 1
        return (weight_bin - weight).detach() + weight # this is the STE procedure

    def forward(self, input):
        binary_weight = self.binarize(self.weight)  # binarize ({0,1}) the weights
        output =  input * binary_weight  # notice no bias
        return output
    

class Model_with_CustomLayer(nn.Module):
     
    def __init__(self):
        super(Model_with_CustomLayer, self).__init__()
        self.custom = CustomLinear(input_dim, input_dim, cutoff=0.5) # this is the custom layer
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
    

# define the custom loss function___________________________________________________________________________________________
# define the custom loss function___________________________________________________________________________________________
class CustomLoss(nn.Module):
    def __init__(self, penalty_strength):
        super(CustomLoss, self).__init__()
        self.penalty_strength = penalty_strength

    def forward(self, y_pred, y_true, penalty):
        loss = torch.mean((y_pred - y_true) ** 2) + penalty*self.penalty_strength
        return loss
    
"""
======================================= NOTES ABOUT THE CUSTOM LOSS FUNCTION =====================================================

loss = torch.mean((y_pred - y_true) ** 2) + penalty*penalty_strength

The loss function is basically a MSE + a penalty for the number of kept (non-zero) features.
The 'penalty' term is just the number of kept features and the 'penalty_strength' term is the coefficient
by which we multiply the 'penalty' term (so the higher that coefficient the more severe the penalty).

"""


# define the training function___________________________________________________________________________________________-
def train(model, X_train, X_test, Y_train, Y_test, loss_function, optimizer,num_epochs):
     
    global train_acc_array  # delcare as global so we can use it outside the function
    global test_acc_array
    train_acc_array = np.arange(num_epochs, dtype = np.float64)
    test_acc_array = np.arange(num_epochs, dtype = np.float64)

    global binary_weight_matrix
    binary_weight_matrix = np.zeros((num_epochs, X_train.shape[1]), dtype = np.float16)  # saving the weights (of the binary mask) at each epoch

    for epoch in range(0, num_epochs):
          
          
          # the differentialble weights from the STE procedure
          custom_layer_weights_for_backprop = model.custom.weight
          
          # the binarized weights from the forward pass (the binary feature mask)
          custom_layer_weights_for_forward_pass = model.custom.binarize(custom_layer_weights_for_backprop).detach().cpu().numpy()

          # get the sum of the binary mask layer, this  sum corresponds to the number of 'kept' features
          num_features_kept = np.sum(custom_layer_weights_for_forward_pass)

          binary_weight_matrix[epoch,:] = custom_layer_weights_for_forward_pass # save the weights
          

          model.train()  # set model to training mode
          optimizer.zero_grad()   # remove all gradients
          Y_pred_train = model(X_train)  # make a prediction based on the training set

          loss = loss_function(y_pred = Y_pred_train, y_true = Y_train, penalty = num_features_kept)  # calculate the losss
          train_acc = loss  # save the training accuracy for the current epoch
          train_acc_array[epoch] = train_acc  # save to array for plotting
          loss.backward() # based on the loss use the chain rule to get the gradient
          optimizer.step()  # make a step based on the gradient

          model.eval()  # set model to evaluation mode
          Y_pred_test = model(X_test)  # make prediction based on the test set
          loss = loss_function(y_pred = Y_pred_test, y_true = Y_test, penalty = num_features_kept)  # calculate testing loss
          test_acc = loss  # save the testing accuracy for the current epoch
          test_acc_array[epoch] = test_acc  # save for plotting

          status = 'Epoch: {}/{}'.format(epoch, num_epochs)
          print(status)
          print('='*50)



# train the model and inspect the weigths at each epoch_____________________________________________________________________________
# here I dont care about the performance of the model, I just want to check how the weights behave.
model = Model_with_CustomLayer()
loss_fn = CustomLoss(penalty_strength = 1)
optim = optim.Adam(model.parameters())
n_epochs = 10000
train(model, X_train, X_test, Y_train, Y_test, loss_function = loss_fn, optimizer = optim, num_epochs = n_epochs)


np.savetxt("C:/Users/Qba Liu/Documents/NAUKA_WLASNA/FEATURE_SELECTION_IDEA/LearnableFeatureSelection/PREPARATORY_STUFF/CUSTOM_LOSS_FUNCTION/binary_weigth_matrix.csv",
           binary_weight_matrix,
           delimiter = ',',
           fmt="%d")

# this prints the amount of binary weight 'flips'
num_flips = np.sum(np.abs(binary_weight_matrix[1,:] - binary_weight_matrix[(n_epochs-1),:]))
diag = 'Number of binary weight flips between the 1st and last epoch: {}'.format(num_flips)
print(diag)





plt.plot(range(0, n_epochs), train_acc_array, label="Train loss", color='blue', linestyle='-')
plt.plot(range(0, n_epochs), test_acc_array, label="Test loss", color='red', linestyle='--')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()


