import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from matplotlib import pyplot as plt



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

# this function is specific to the results obtained on these simulated datasets:
# "C:\Users\Qba Liu\Documents\NAUKA_WLASNA\FEATURE_SELECTION_IDEA\LearnableFeatureSelection\PREPARATORY_STUFF\LEARNING_TEST\X.txt"
# "C:\Users\Qba Liu\Documents\NAUKA_WLASNA\FEATURE_SELECTION_IDEA\LearnableFeatureSelection\PREPARATORY_STUFF\LEARNING_TEST\y.txt"
# The perfect scores are: 1.0 and 1.0
def performance_on_simdata1(bin_weights_last_layer):
    true_predictors = bin_weights_last_layer[:101]
    noise_predictors = bin_weights_last_layer[101:]

    correct_true = np.sum(true_predictors)/len(true_predictors) # proportion of correctly included actual features
    correct_noise = np.sum(np.abs(1-noise_predictors))/len(noise_predictors) # porportion of correctly omitted noisy features

    return (correct_true, correct_noise)

