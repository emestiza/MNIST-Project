
# Note: This code takes several minutes to run.

# Install packages
install.packages(c("dslabs", "caret", "matrixStats", "e1071", "rmarkdown","knitr"))

# Load package and data
library(dslabs)
library(caret)
library(matrixStats)
library(e1071)
library(rmarkdown)
library(knitr)

# MNIST Data
mnist <- read_mnist()
names(mnist)

# Matrix column features 
dim(mnist$train$images)

# Vector class
class(mnist$train$labels)

# Table with digit labels
table(mnist$train$labels)

# Training dataset
set.seed(123)
index <- sample(nrow(mnist$train$images), 20000)
x <- mnist$train$images[index,]
y <- factor(mnist$train$labels[index])

# Training subset dataset
index <- sample(nrow(mnist$train$images), 2000)
x_validation <- mnist$train$images[index,]
y_validation <- factor(mnist$train$labels[index])

# Preprocessing MNIST Data
sds <- colSds(x)
qplot(sds, bins = 256, color = I("black"))

# Columns with near zero variance are removed
nzv <- nearZeroVar(x)
col_index <- setdiff(1:ncol(x), nzv)
length(col_index)

# Model Fitting for MNIST Data
colnames(x) <- 1:ncol(mnist$train$images)
colnames(x_validation) <- colnames(mnist$train$images)

# Knn for training dataset
control <- trainControl(method = "cv", number = 10, p = 0.9)
train_knn <- train(x[ ,col_index], y, method = "knn", tuneGrid = data.frame(k = c(1,3,5,7)), trControl = control)
plot(train_knn)

# Fitting entire training dataset
fit_knn <-  knn3(x[ ,col_index], y, k = 3) 
y_hat_knn <- predict(fit_knn, x_validation[, col_index], type = "class")
cm <- confusionMatrix(y_hat_knn, factor(y_validation))
cm$overall["Accuracy"]  

# Training dataset specificity and sensitivity output
cm$byClass[, 1:2]

# Knn for test dataset
index <- sample(nrow(mnist$test$images), 2000)
x_test <- mnist$test$images[index,]
y_test <- factor(mnist$test$labels[index])
colnames(x_test) <- colnames(mnist$test$images)

fit_knn <-  knn3(x[ ,col_index], y, k = 3) 
y_hat_knn <- predict(fit_knn, x_test[, col_index], type = "class")
cm_test <- confusionMatrix(y_hat_knn, factor(y_test))
cm_test$overall["Accuracy"]  

# Test dataset specificity and sensitivity output
cm_test$byClass[, 1:2]