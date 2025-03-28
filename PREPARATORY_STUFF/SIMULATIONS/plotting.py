import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

'''
data = np.loadtxt('Simulation_thresh.txt')
param_value = data[:,0]
correct_inclusion_rate = data[:,1]
correct_exlcusion_rate = data[:,2]
sum_score = data[:,3]

plt.plot(param_value, correct_inclusion_rate, label = 'mean correct_inclusion_rate')
plt.plot(param_value, correct_exlcusion_rate, label = 'mean correct_exlcusion_rate')
plt.plot(param_value, sum_score, label = 'sum_score')
plt.grid()
plt.legend()
plt.title('Threshold')
plt.xlabel('Threshold value')
plt.ylabel('Scores')
plt.show()




data = np.loadtxt('Simulation_penalty_strength.txt')
param_value = data[:,0]
correct_inclusion_rate = data[:,1]
correct_exlcusion_rate = data[:,2]
sum_score = data[:,3]

plt.plot(param_value, correct_inclusion_rate, label = 'mean correct_inclusion_rate')
plt.plot(param_value, correct_exlcusion_rate, label = 'mean correct_exlcusion_rate')
plt.plot(param_value, sum_score, label = 'sum_score')
plt.grid()
plt.legend()
plt.title('Penalty strength')
plt.xlabel('Penalty strength')
plt.ylabel('Scores')
plt.show()



'''

data = pd.read_csv('Simulation_optimizers.txt', sep = ' ')

plt.plot(data.iloc[:,0], data.iloc[:,1], label = 'mean correct inclusion rate')
plt.plot(data.iloc[:,0], data.iloc[:,2], label = 'mean correct exclusion rate')
plt.plot(data.iloc[:,0], data.iloc[:,3], label = 'mean summary score')
plt.grid()
plt.legend()
plt.xlabel('Optimizers')
plt.ylabel('metrics')
plt.show()



