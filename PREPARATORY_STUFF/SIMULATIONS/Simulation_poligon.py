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
sys.path.append("C:/Users/Qba Liu/Documents/NAUKA_WLASNA/FEATURE_SELECTION_IDEA/LearnableFeatureSelection/PREPARATORY_STUFF/SIMULATIONS/simulation_wrapper.py")
import simulation_wrapper as wrp
import pandas as pd


# read and prepare data _____________________________________________________________________
X = np.loadtxt('C:/Users/Qba Liu/Documents/NAUKA_WLASNA/FEATURE_SELECTION_IDEA/LearnableFeatureSelection/PREPARATORY_STUFF/LEARNING_TEST/X.txt', dtype = np.float32)
Y = np.loadtxt('C:/Users/Qba Liu/Documents/NAUKA_WLASNA/FEATURE_SELECTION_IDEA/LearnableFeatureSelection/PREPARATORY_STUFF/LEARNING_TEST/y.txt', dtype = np.float32)
X = torch.tensor(X)
Y = torch.tensor(Y).reshape(-1, 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
input_dim = X_train.shape[1]


thresh_values = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1])
cols = ['threshold value', 'mean_correct_inclusion_rate', 'mean_correct_exclusion_rate']
simulation_table = pd.DataFrame(index=range(len(thresh_values)), columns=cols)


for i in range(0, len(thresh_values)):

    print('===============================threshold: {}========================================='.format(thresh_values[i]))
    simulation_table.iloc[i,0] = thresh_values[i]

    sim = wrp.Wrapper(thresh=thresh_values[i],
        penalty_strength=10,
        optimizer='AdamW',
        n_epochs=1000,
        n_rep=20,
        X_train=X_train,
        X_test=X_test,
        Y_train=Y_train,
        Y_test=Y_test)
    
    mean_inclustion_rates = np.mean(sim[0])
    mean_exclusion_rates = np.mean(sim[1])

    simulation_table.iloc[i,1] = mean_inclustion_rates
    simulation_table.iloc[i,2] = mean_exclusion_rates
    



print(simulation_table)