import numpy as np

n = 1000
p = 100
true_coefs = np.random.uniform(-5,5,(p+1))
noise = np.random.normal(0,1,n)
X = np.random.normal(0,1,(n,p))
Y = X@true_coefs[:p]
Y = Y + true_coefs[-1]
Y = Y + noise

np.savetxt("C:/Users/Qba Liu/Documents/NAUKA_WLASNA/FEATURE_SELECTION_IDEA/LearnableFeatureSelection/PREPARATORY_STUFF/PYTORCH_RECAP/X_sim.txt", X, delimiter = ' ')
np.savetxt("C:/Users/Qba Liu/Documents/NAUKA_WLASNA/FEATURE_SELECTION_IDEA/LearnableFeatureSelection/PREPARATORY_STUFF/PYTORCH_RECAP/Y_sim.txt", Y, delimiter = ' ')