import numpy as np

num_samples = 1000
num_actual_features = 100
num_noise_features = 50

# Create the actual predictors which are important for the explanation of the response
# They should have low correlations amongst eachothers
covariance_matrix = np.zeros((num_actual_features, num_actual_features), dtype = np.float64)
np.fill_diagonal(covariance_matrix, 1.0)
predictors = np.random.multivariate_normal(mean = np.arange(num_actual_features),
                                        cov = covariance_matrix,
                                        size = num_samples)

# based on the actual predictors create the response (+ some noise)
intercept = 10.0
coefs = np.random.uniform(100,150,num_actual_features)
y = predictors@coefs
y = y + intercept
y = y + np.random.normal(0, np.std(y)*0.05, num_samples)

# add noise predictors which are uncorrelated with the response (they will be the last 50 predictors)
uncorrelated_predictors = np.random.normal(0,100,(num_samples, num_noise_features))
X = np.concatenate((predictors, uncorrelated_predictors), axis = 1)


np.savetxt("C:/Users/Qba Liu/Documents/NAUKA_WLASNA/FEATURE_SELECTION_IDEA/LearnableFeatureSelection/PREPARATORY_STUFF/LEARNING_TEST/X.txt", X)
np.savetxt("C:/Users/Qba Liu/Documents/NAUKA_WLASNA/FEATURE_SELECTION_IDEA/LearnableFeatureSelection/PREPARATORY_STUFF/LEARNING_TEST/y.txt", y)
