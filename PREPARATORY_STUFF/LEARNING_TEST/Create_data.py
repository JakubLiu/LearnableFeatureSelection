import numpy as np

num_samples = 1000
num_actual_features = 100
num_noise_features = 50

predictors = np.random.normal(0,1,(num_samples, num_actual_features))
true_coefs = np.random.uniform(-5,5,num_actual_features+1)
y = predictors@true_coefs[1:]
y = y + true_coefs[0]
noise = np.random.normal(0, np.std(y)*0.1, num_samples)
y = y + noise

uncorrelated_predictors = np.random.normal(0,1,(num_samples, num_noise_features))
X = np.concatenate((predictors, uncorrelated_predictors), axis = 1)

np.savetxt("C:/Users/Qba Liu/Documents/NAUKA_WLASNA/FEATURE_SELECTION_IDEA/LearnableFeatureSelection/PREPARATORY_STUFF/LEARNING_TEST/X.txt", X)
np.savetxt("C:/Users/Qba Liu/Documents/NAUKA_WLASNA/FEATURE_SELECTION_IDEA/LearnableFeatureSelection/PREPARATORY_STUFF/LEARNING_TEST/y.txt", y)
print('done.')