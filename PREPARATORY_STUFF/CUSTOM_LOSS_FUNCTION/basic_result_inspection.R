library(data.table)

data <- fread("C:/Users/Qba Liu/Documents/NAUKA_WLASNA/FEATURE_SELECTION_IDEA/LearnableFeatureSelection/PREPARATORY_STUFF/CUSTOM_LOSS_FUNCTION/binary_weigth_matrix.csv")



# look how the binary weights flip across the epochs
plot(1:nrow(data), data[[1]], type = "l", ylim = c(0,1), col = 1, 
     xlab = "epoch", ylab = "weight")

for(i in 2:ncol(data)) {
  lines(1:nrow(data), data[[i]], col = i)
}


# look at the distribution of the binary weights in the last epochs
last_ep <- data[nrow(data),]
plot(1:ncol(data), last_ep)
