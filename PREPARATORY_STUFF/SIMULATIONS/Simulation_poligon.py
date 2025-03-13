import sys
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
sys.path.append("C:/Users/Qba Liu/Documents/NAUKA_WLASNA/FEATURE_SELECTION_IDEA/LearnableFeatureSelection/PREPARATORY_STUFF/SOURCECODE/")
import sourcecode as src


# read and prepare data _____________________________________________________________________
X = np.loadtxt('C:/Users/Qba Liu/Documents/NAUKA_WLASNA/FEATURE_SELECTION_IDEA/LearnableFeatureSelection/PREPARATORY_STUFF/LEARNING_TEST/X.txt', dtype = np.float32)
Y = np.loadtxt('C:/Users/Qba Liu/Documents/NAUKA_WLASNA/FEATURE_SELECTION_IDEA/LearnableFeatureSelection/PREPARATORY_STUFF/LEARNING_TEST/y.txt', dtype = np.float32)
X = torch.tensor(X)
Y = torch.tensor(Y).reshape(-1, 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
input_dim = X_train.shape[1]


# create the model__________________________________________________________________________________________________________
class Model_with_CustomLayer(nn.Module):
     
    def __init__(self):
        super(Model_with_CustomLayer, self).__init__()
        self.custom = src.CustomLinear(input_dim, input_dim, cutoff=0.7) # this is the custom layer
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


# define the training parameters_____________________________________________________________________________________________________

model = Model_with_CustomLayer()
loss_fn = src.CustomLoss(penalty_strength = 80)
optim = optim.Adam(model.parameters())
n_epochs = 10000

# train the model_____________________________________________________________________________________________________________
train(model, X_train, X_test, Y_train, Y_test, loss_function = loss_fn, optimizer = optim, num_epochs = n_epochs)

# calculate the dataset-specific performacne metrics
performance_metrics = src.performance_on_simdata1(binary_weight_matrix[(n_epochs-1),:])
correct_inclusion_rate = performance_metrics[0]
correct_exclusion_rate = performance_metrics[1]

res = 'Correct inclusion rate: {}\nCorrect exclusion rate: {}'.format(np.round(correct_inclusion_rate,4),
                                                                      np.round(correct_exclusion_rate,4))

print(res)


# plot the learning curve__________________________________________________________________________________________
plt.plot(range(0, n_epochs), train_acc_array, label="Train loss", color='blue', linestyle='-')
plt.plot(range(0, n_epochs), test_acc_array, label="Test loss", color='red', linestyle='--')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

"""
cutoff = 0.5

Correct inclusion rate: 0.2673
Correct exclusion rate: 0.5714

___________________________________________________________________________________________

cutoff = 0.2

Correct inclusion rate: 0.5545
Correct exclusion rate: 0.3469

_________________________________________________________________________________________

cutoff = 0.7 -----------------------> best for now

Correct inclusion rate: 0.2871
Correct exclusion rate: 0.6327

_______________________________________________________________________________________

cutoff = 0.9

Correct inclusion rate: 0.2475
Correct exclusion rate: 0.6327

______________________________________________________________________________________

cutoff = 1.5

Correct inclusion rate: 0.0693
Correct exclusion rate: 0.9184
"""

"""
penalty = 10

Correct inclusion rate: 0.297
Correct exclusion rate: 0.6735

_________________________________________________________________________________________________________________

penalty = 20  ---------------------------> best so far

Correct inclusion rate: 0.3267
Correct exclusion rate: 0.7143

____________________________________________________________________________________________________________


penalty = 50

Correct inclusion rate: 0.2772
Correct exclusion rate: 0.5714

____________________________________________________________________________________________________________

penalty = 80

Correct inclusion rate: 0.3861
Correct exclusion rate: 0.551
"""