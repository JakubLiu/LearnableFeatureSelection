library(data.table)

X <- fread("C:/Users/Qba Liu/Documents/NAUKA_WLASNA/FEATURE_SELECTION_IDEA/LearnableFeatureSelection/PREPARATORY_STUFF/LEARNING_TEST/X.txt")
y <- scan("C:/Users/Qba Liu/Documents/NAUKA_WLASNA/FEATURE_SELECTION_IDEA/LearnableFeatureSelection/PREPARATORY_STUFF/LEARNING_TEST/y.txt")
X <- as.matrix(X)

# Ideally the 1st 100 predictors should be significant and the last 50 non-significant
summary(lm(y~X))


