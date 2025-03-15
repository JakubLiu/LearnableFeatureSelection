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


# create the model__________________________________________________________________________________________________________
class Model_with_CustomLayer(nn.Module):
     
    def __init__(self, input_dim, thresh):
        super(Model_with_CustomLayer, self).__init__()
        self.input_dim = input_dim
        self.thresh = thresh
        self.custom = src.CustomLinear(self.input_dim, self.input_dim, cutoff=self.thresh) # this is the custom layer
        self.fc1 = nn.Linear(self.input_dim, 64)
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

          #status = 'Epoch: {}/{}'.format(epoch, num_epochs)
          #print(status)
          #print('='*50)


def Wrapper(thresh, penalty_strength, optimizer, n_epochs, n_rep, X_train, X_test, Y_train, Y_test):

    num_features = X_train.shape[1]
    model = Model_with_CustomLayer(input_dim=num_features, thresh=thresh)
    loss_fn = src.CustomLoss(penalty_strength = penalty_strength)

    # for all these optimizers, their default parameters will be used
    global optimi
    if optimizer == 'Adam':   
        optimi = optim.Adam(model.parameters())
    
    elif optimizer == 'AdamW':
        optimi = optim.AdamW(model.parameters())

    elif optimizer == 'RMSprop':
        optimi = optim.RMSprop(model.parameters())
    
    elif optimizer == 'Adagram':
        optimi = optim.Adagrad(model.parameters())
    
    elif optimizer == 'Adadelta':
        optimi = optim.Adadelta(model.parameters())
    
    elif optimizer == 'NAdam':
        optimi = optim.NAdam(model.parameters())
    
    elif optimizer == 'ASGD':
        optimi = optim.ASGD(model.parameters())
    
    else:  # use stochastic gradient descent by default
        optimizer = optim.SGD(model.parameters())

    n_epochs = n_epochs

    inclusion_rates = np.arange(n_rep, dtype = np.float64)
    exclusion_rates = np.arange(n_rep, dtype = np.float64)


    for i in range(0, n_rep):
        status = str(i/n_rep*100) + '%'
        print(status)
        train(model, X_train, X_test, Y_train, Y_test, loss_function = loss_fn, optimizer = optimi, num_epochs = n_epochs)
        performance_metrics = src.performance_on_simdata1(binary_weight_matrix[(n_epochs-1),:])
        correct_inclusion_rate = performance_metrics[0]
        correct_exclusion_rate = performance_metrics[1]
        inclusion_rates[i] = correct_inclusion_rate
        exclusion_rates[i] = correct_exclusion_rate


    report = {'threshold': thresh,
              'penalty strength' : penalty_strength,
              'optimizer': optimizer,
              'n_epochs': n_epochs,
              'n_repetitions': n_rep}


    return (inclusion_rates, exclusion_rates, report)


# ===================================================================================================================================

# read and prepare data _____________________________________________________________________
X = np.loadtxt('C:/Users/Qba Liu/Documents/NAUKA_WLASNA/FEATURE_SELECTION_IDEA/LearnableFeatureSelection/PREPARATORY_STUFF/LEARNING_TEST/X.txt', dtype = np.float32)
Y = np.loadtxt('C:/Users/Qba Liu/Documents/NAUKA_WLASNA/FEATURE_SELECTION_IDEA/LearnableFeatureSelection/PREPARATORY_STUFF/LEARNING_TEST/y.txt', dtype = np.float32)
X = torch.tensor(X)
Y = torch.tensor(Y).reshape(-1, 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

out = Wrapper(thresh=0.5,
        penalty_strength=10,
        optimizer='AdamW',
        n_epochs=1000,
        n_rep=20,
        X_train=X_train,
        X_test=X_test,
        Y_train=Y_train,
        Y_test=Y_test)

mean_inclustion_rates = np.mean(out[0])
mean_exclusion_rates = np.mean(out[1])

print(mean_inclustion_rates)
print(mean_exclusion_rates)