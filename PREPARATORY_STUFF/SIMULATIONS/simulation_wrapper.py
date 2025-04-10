import sys
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
import multiprocessing as mp
sys.path.append("C:/Users/Qba Liu/Documents/NAUKA_WLASNA/FEATURE_SELECTION_IDEA/LearnableFeatureSelection/PREPARATORY_STUFF/SOURCECODE/")
import sourcecode as src


# create the model__________________________________________________________________________________________________________

# medium depth model
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
    
# shallow model
class Model_with_CustomLayer_shallow(nn.Module):
     
    def __init__(self, input_dim, thresh):
        super(Model_with_CustomLayer_shallow, self).__init__()
        self.input_dim = input_dim
        self.thresh = thresh
        self.custom = src.CustomLinear(self.input_dim, self.input_dim, cutoff=self.thresh) # this is the custom layer
        self.fc1 = nn.Linear(self.input_dim, 32)
        self.fc2 = nn.Linear(32,1)

    def forward(self, x):
         x = self.custom(x)
         x = torch.relu(x)
         x = self.fc1(x)
         x = torch.relu(x)
         x = self.fc2(x)

         return(x)

# deep model
class Model_with_CustomLayer_deep(nn.Module):
     
    def __init__(self, input_dim, thresh):
        super(Model_with_CustomLayer_deep, self).__init__()
        self.input_dim = input_dim
        self.thresh = thresh
        self.custom = src.CustomLinear(self.input_dim, self.input_dim, cutoff=self.thresh) # this is the custom layer
        self.fc1 = nn.Linear(self.input_dim, self.input_dim)
        self.fc2 = nn.Linear(self.input_dim, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 32)
        self.fc6 = nn.Linear(32,1)

    def forward(self, x):
         x = self.custom(x)
         x = torch.relu(x)
         x = self.fc1(x)
         x = torch.relu(x)
         x = self.fc2(x)
         x = torch.relu(x)
         x = self.fc3(x)
         x = torch.relu(x)
         x = self.fc4(x)
         x = torch.relu(x)
         x = self.fc5(x)
         x = torch.relu(x)
         x = self.fc6(x)

         return(x)


# very deep model
class Model_with_CustomLayer_super_deep(nn.Module):
     
    def __init__(self, input_dim, thresh):
        super(Model_with_CustomLayer_super_deep, self).__init__()
        self.input_dim = input_dim
        self.thresh = thresh
        self.custom = src.CustomLinear(self.input_dim, self.input_dim, cutoff=self.thresh) # this is the custom layer
        self.fc1 = nn.Linear(self.input_dim, self.input_dim)
        self.fc2 = nn.Linear(self.input_dim, self.input_dim)
        self.fc3 = nn.Linear(self.input_dim, self.input_dim)
        self.fc4 = nn.Linear(self.input_dim, 64)
        self.fc5 = nn.Linear(64, 64)
        self.fc6 = nn.Linear(64, 64)
        self.fc7 = nn.Linear(64, 64)
        self.fc8 = nn.Linear(64, 32)
        self.fc9 = nn.Linear(32, 32)
        self.fc10 = nn.Linear(32, 32)
        self.fc11 = nn.Linear(32, 32)
        self.fc12 = nn.Linear(32,32)
        self.fc13 = nn.Linear(32,16)
        self.fc14 = nn.Linear(16,16)
        self.fc15 = nn.Linear(16,16)
        self.fc16 = nn.Linear(16,1)

    def forward(self, x):
         x = self.custom(x)
         x = torch.relu(x)
         x = self.fc1(x)
         x = torch.relu(x)
         x = self.fc2(x)
         x = torch.relu(x)
         x = self.fc3(x)
         x = torch.relu(x)
         x = self.fc4(x)
         x = torch.relu(x)
         x = self.fc5(x)
         x = torch.relu(x)
         x = self.fc6(x)
         x = torch.relu(x)
         x = self.fc7(x)
         x = torch.relu(x)
         x = self.fc8(x)
         x = torch.relu(x)
         x = self.fc9(x)
         x = torch.relu(x)
         x = self.fc10(x)
         x = torch.relu(x)
         x = self.fc11(x)
         x = torch.relu(x)
         x = self.fc12(x)
         x = torch.relu(x)
         x = self.fc13(x)
         x = torch.relu(x)
         x = self.fc14(x)
         x = torch.relu(x)
         x = self.fc15(x)
         x = torch.relu(x)
         x = self.fc16(x)

         return(x)



# the following models will be there to look how far we can go into the model depth to get a performance gain on this specific dataset

# deeep model 1
class DeepModel1(nn.Module):
    def __init__(self, input_dim, thresh):
        super(DeepModel1, self).__init__()
        self.input_dim = input_dim
        self.thresh = thresh

        self.custom = src.CustomLinear(self.input_dim, self.input_dim, cutoff=self.thresh)

        self.fc1 = nn.Linear(self.input_dim, self.input_dim)
        self.fc2 = nn.Linear(self.input_dim, self.input_dim)
        self.fc3 = nn.Linear(self.input_dim, self.input_dim)
        self.fc4 = nn.Linear(self.input_dim, 64)
        self.fc5 = nn.Linear(64, 64)
        self.fc6 = nn.Linear(64, 64)
        self.fc7 = nn.Linear(64, 64)
        self.fc8 = nn.Linear(64, 64)
        self.fc9 = nn.Linear(64, 64)
        self.fc10 = nn.Linear(64, 64)
        self.fc11 = nn.Linear(64, 32)
        self.fc12 = nn.Linear(32, 32)
        self.fc13 = nn.Linear(32, 32)
        self.fc14 = nn.Linear(32, 32)
        self.fc15 = nn.Linear(32, 32)
        self.fc16 = nn.Linear(32, 32)
        self.fc17 = nn.Linear(32, 32)
        self.fc18 = nn.Linear(32, 32)
        self.fc19 = nn.Linear(32, 16)
        self.fc20 = nn.Linear(16, 16)
        self.fc21 = nn.Linear(16, 16)
        self.fc22 = nn.Linear(16, 16)
        self.fc23 = nn.Linear(16, 16)
        self.fc24 = nn.Linear(16, 16)
        self.fc25 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.custom(x)
        x = torch.relu(x)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = torch.relu(self.fc8(x))
        x = torch.relu(self.fc9(x))
        x = torch.relu(self.fc10(x))
        x = torch.relu(self.fc11(x))
        x = torch.relu(self.fc12(x))
        x = torch.relu(self.fc13(x))
        x = torch.relu(self.fc14(x))
        x = torch.relu(self.fc15(x))
        x = torch.relu(self.fc16(x))
        x = torch.relu(self.fc17(x))
        x = torch.relu(self.fc18(x))
        x = torch.relu(self.fc19(x))
        x = torch.relu(self.fc20(x))
        x = torch.relu(self.fc21(x))
        x = torch.relu(self.fc22(x))
        x = torch.relu(self.fc23(x))
        x = torch.relu(self.fc24(x))
        x = self.fc25(x)

        return x


# deep model 2
class DeepModel2(nn.Module):
    def __init__(self, input_dim, thresh):
        super(DeepModel2, self).__init__()
        self.input_dim = input_dim
        self.thresh = thresh

        self.custom = src.CustomLinear(self.input_dim, self.input_dim, cutoff=self.thresh)

        self.fc1 = nn.Linear(self.input_dim, self.input_dim)
        self.fc2 = nn.Linear(self.input_dim, self.input_dim)
        self.fc3 = nn.Linear(self.input_dim, self.input_dim)
        self.fc4 = nn.Linear(self.input_dim, 64)
        self.fc5 = nn.Linear(64, 64)
        self.fc6 = nn.Linear(64, 64)
        self.fc7 = nn.Linear(64, 64)
        self.fc8 = nn.Linear(64, 64)
        self.fc9 = nn.Linear(64, 64)
        self.fc10 = nn.Linear(64, 64)
        self.fc11 = nn.Linear(64, 64)
        self.fc12 = nn.Linear(64, 64)
        self.fc13 = nn.Linear(64, 64)
        self.fc14 = nn.Linear(64, 64)
        self.fc15 = nn.Linear(64, 64)
        self.fc16 = nn.Linear(64, 64)
        self.fc17 = nn.Linear(64, 32)
        self.fc18 = nn.Linear(32, 32)
        self.fc19 = nn.Linear(32, 32)
        self.fc20 = nn.Linear(32, 32)
        self.fc21 = nn.Linear(32, 32)
        self.fc22 = nn.Linear(32, 32)
        self.fc23 = nn.Linear(32, 32)
        self.fc24 = nn.Linear(32, 32)
        self.fc25 = nn.Linear(32, 32)
        self.fc26 = nn.Linear(32, 16)
        self.fc27 = nn.Linear(16, 16)
        self.fc28 = nn.Linear(16, 16)
        self.fc29 = nn.Linear(16, 16)
        self.fc30 = nn.Linear(16, 16)
        self.fc31 = nn.Linear(16, 16)
        self.fc32 = nn.Linear(16, 16)
        self.fc33 = nn.Linear(16, 16)
        self.fc34 = nn.Linear(16, 16)
        self.fc35 = nn.Linear(16, 16)
        self.fc36 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.custom(x)
        x = torch.relu(x)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = torch.relu(self.fc8(x))
        x = torch.relu(self.fc9(x))
        x = torch.relu(self.fc10(x))
        x = torch.relu(self.fc11(x))
        x = torch.relu(self.fc12(x))
        x = torch.relu(self.fc13(x))
        x = torch.relu(self.fc14(x))
        x = torch.relu(self.fc15(x))
        x = torch.relu(self.fc16(x))
        x = torch.relu(self.fc17(x))
        x = torch.relu(self.fc18(x))
        x = torch.relu(self.fc19(x))
        x = torch.relu(self.fc20(x))
        x = torch.relu(self.fc21(x))
        x = torch.relu(self.fc22(x))
        x = torch.relu(self.fc23(x))
        x = torch.relu(self.fc24(x))
        x = torch.relu(self.fc25(x))
        x = torch.relu(self.fc26(x))
        x = torch.relu(self.fc27(x))
        x = torch.relu(self.fc28(x))
        x = torch.relu(self.fc29(x))
        x = torch.relu(self.fc30(x))
        x = torch.relu(self.fc31(x))
        x = torch.relu(self.fc32(x))
        x = torch.relu(self.fc33(x))
        x = torch.relu(self.fc34(x))
        x = torch.relu(self.fc35(x))
        x = self.fc36(x)

        return x


# deep model 3
class DeepModel3(nn.Module):
    def __init__(self, input_dim, thresh):
        super(DeepModel3, self).__init__()
        self.input_dim = input_dim
        self.thresh = thresh

        self.custom = src.CustomLinear(self.input_dim, self.input_dim, cutoff=self.thresh)

        self.fc1 = nn.Linear(self.input_dim, self.input_dim)
        self.fc2 = nn.Linear(self.input_dim, self.input_dim)
        self.fc3 = nn.Linear(self.input_dim, self.input_dim)
        self.fc4 = nn.Linear(self.input_dim, 64)
        self.fc5 = nn.Linear(64, 64)
        self.fc6 = nn.Linear(64, 64)
        self.fc7 = nn.Linear(64, 64)
        self.fc8 = nn.Linear(64, 64)
        self.fc9 = nn.Linear(64, 64)
        self.fc10 = nn.Linear(64, 64)
        self.fc11 = nn.Linear(64, 64)
        self.fc12 = nn.Linear(64, 64)
        self.fc13 = nn.Linear(64, 64)
        self.fc14 = nn.Linear(64, 64)
        self.fc15 = nn.Linear(64, 64)
        self.fc16 = nn.Linear(64, 64)
        self.fc17 = nn.Linear(64, 64)
        self.fc18 = nn.Linear(64, 64)
        self.fc19 = nn.Linear(64, 64)
        self.fc20 = nn.Linear(64, 64)
        self.fc21 = nn.Linear(64, 64)
        self.fc22 = nn.Linear(64, 64)
        self.fc23 = nn.Linear(64, 64)
        self.fc24 = nn.Linear(64, 64)
        self.fc25 = nn.Linear(64, 64)
        self.fc26 = nn.Linear(64, 32)
        self.fc27 = nn.Linear(32, 32)
        self.fc28 = nn.Linear(32, 32)
        self.fc29 = nn.Linear(32, 32)
        self.fc30 = nn.Linear(32, 32)
        self.fc31 = nn.Linear(32, 32)
        self.fc32 = nn.Linear(32, 32)
        self.fc33 = nn.Linear(32, 32)
        self.fc34 = nn.Linear(32, 32)
        self.fc35 = nn.Linear(32, 32)
        self.fc36 = nn.Linear(32, 16)
        self.fc37 = nn.Linear(16, 16)
        self.fc38 = nn.Linear(16, 16)
        self.fc39 = nn.Linear(16, 16)
        self.fc40 = nn.Linear(16, 16)
        self.fc41 = nn.Linear(16, 16)
        self.fc42 = nn.Linear(16, 16)
        self.fc43 = nn.Linear(16, 16)
        self.fc44 = nn.Linear(16, 16)
        self.fc45 = nn.Linear(16, 16)
        self.fc46 = nn.Linear(16, 16)
        self.fc47 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.custom(x)
        x = torch.relu(x)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = torch.relu(self.fc8(x))
        x = torch.relu(self.fc9(x))
        x = torch.relu(self.fc10(x))
        x = torch.relu(self.fc11(x))
        x = torch.relu(self.fc12(x))
        x = torch.relu(self.fc13(x))
        x = torch.relu(self.fc14(x))
        x = torch.relu(self.fc15(x))
        x = torch.relu(self.fc16(x))
        x = torch.relu(self.fc17(x))
        x = torch.relu(self.fc18(x))
        x = torch.relu(self.fc19(x))
        x = torch.relu(self.fc20(x))
        x = torch.relu(self.fc21(x))
        x = torch.relu(self.fc22(x))
        x = torch.relu(self.fc23(x))
        x = torch.relu(self.fc24(x))
        x = torch.relu(self.fc25(x))
        x = torch.relu(self.fc26(x))
        x = torch.relu(self.fc27(x))
        x = torch.relu(self.fc28(x))
        x = torch.relu(self.fc29(x))
        x = torch.relu(self.fc30(x))
        x = torch.relu(self.fc31(x))
        x = torch.relu(self.fc32(x))
        x = torch.relu(self.fc33(x))
        x = torch.relu(self.fc34(x))
        x = torch.relu(self.fc35(x))
        x = torch.relu(self.fc36(x))
        x = torch.relu(self.fc37(x))
        x = torch.relu(self.fc38(x))
        x = torch.relu(self.fc39(x))
        x = torch.relu(self.fc40(x))
        x = torch.relu(self.fc41(x))
        x = torch.relu(self.fc42(x))
        x = torch.relu(self.fc43(x))
        x = torch.relu(self.fc44(x))
        x = torch.relu(self.fc45(x))
        x = torch.relu(self.fc46(x))
        x = self.fc47(x)

        return x




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
    
    elif optimizer == 'Adagrad':
        optimi = optim.Adagrad(model.parameters())
    
    elif optimizer == 'Adadelta':
        optimi = optim.Adadelta(model.parameters())
    
    elif optimizer == 'NAdam':
        optimi = optim.NAdam(model.parameters())
    
    elif optimizer == 'ASGD':
        optimi = optim.ASGD(model.parameters())

    elif optimizer == 'SGD':
        optimi = optim.SGD(model.parameters())
    
    else:  # use stochastic gradient descent by default
        optimizer = optim.SGD(model.parameters())

    n_epochs = n_epochs

    inclusion_rates = np.arange(n_rep, dtype = np.float64)
    exclusion_rates = np.arange(n_rep, dtype = np.float64)


    for i in range(0, n_rep):
        status = str(i/n_rep*100) + '%'
        print(status, flush=True)
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



def Wrapper_model(model_depth, thresh, penalty_strength, optimizer, n_epochs, n_rep, X_train, X_test, Y_train, Y_test):

    num_features = X_train.shape[1]

    if model_depth == 'shallow':
        model = Model_with_CustomLayer_shallow(input_dim=num_features, thresh=thresh)
    elif model_depth == 'standard':
        model = Model_with_CustomLayer(input_dim=num_features, thresh=thresh)
    elif model_depth == 'deep':
        model = Model_with_CustomLayer_deep(input_dim=num_features, thresh=thresh)
    elif model_depth == 'very_deep':
        model = Model_with_CustomLayer_super_deep(input_dim=num_features, thresh=thresh)
    elif model_depth == 'DeepModel1':
        model = DeepModel1(input_dim=num_features, thresh = thresh)
    elif model_depth == 'DeepModel2':
        model = DeepModel2(input_dim=num_features, thresh = thresh)
    elif model_depth == 'DeepModel3':
        model = DeepModel3(input_dim=num_features, thresh = thresh)
    else:
        sys.exit('Wrong model specified.')

    
    loss_fn = src.CustomLoss(penalty_strength = penalty_strength)

    # for all these optimizers, their default parameters will be used
    global optimi
    if optimizer == 'Adam':   
        optimi = optim.Adam(model.parameters())
    
    elif optimizer == 'AdamW':
        optimi = optim.AdamW(model.parameters())

    elif optimizer == 'RMSprop':
        optimi = optim.RMSprop(model.parameters())
    
    elif optimizer == 'Adagrad':
        optimi = optim.Adagrad(model.parameters())
    
    elif optimizer == 'Adadelta':
        optimi = optim.Adadelta(model.parameters())
    
    elif optimizer == 'NAdam':
        optimi = optim.NAdam(model.parameters())
    
    elif optimizer == 'ASGD':
        optimi = optim.ASGD(model.parameters())

    elif optimizer == 'SGD':
        optimi = optim.SGD(model.parameters())
    
    else:  # use stochastic gradient descent by default
        optimizer = optim.SGD(model.parameters())

    n_epochs = n_epochs

    inclusion_rates = np.arange(n_rep, dtype = np.float64)
    exclusion_rates = np.arange(n_rep, dtype = np.float64)


    for i in range(0, n_rep):
        status = str(i/n_rep*100) + '%'
        print(status, flush=True)
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




# Wrapper(thresh, penalty_strength, optimizer, n_epochs, n_rep, X_train, X_test, Y_train, Y_test):

def WrapperScalar(j,thresh, penalty_strength, optimizer, n_epochs, X_train, X_test, Y_train, Y_test):
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

    train(model, X_train, X_test, Y_train, Y_test, loss_function = loss_fn, optimizer = optimi, num_epochs = n_epochs)
    performance_metrics = src.performance_on_simdata1(binary_weight_matrix[(n_epochs-1),:])
    #correct_inclusion_rate = performance_metrics[0]
    correct_exclusion_rate = performance_metrics[1]

    return correct_exclusion_rate


# def WrapperScalar(j,thresh, penalty_strength, optimizer, n_epochs, X_train, X_test, Y_train, Y_test):

def WrapperArray(num_cores, n_rep, thresh, penalty_strength, optimizer, n_epochs, X_train, X_test, Y_train, Y_test):

    with mp.Pool(processes=num_cores) as pool:
        correct_exclusion_rate_array = pool.starmap(WrapperScalar, [(j, thresh, penalty_strength, optimizer, n_epochs, X_train, X_test, Y_train, Y_test) 
                                           for j in range(n_rep)])
        

    return np.array(correct_exclusion_rate_array)
    

    



# ===================================================================================================================================
'''
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
'''