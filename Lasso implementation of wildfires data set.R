#Data set: 
#https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength

library(readxl)  #For latex output
library(glmnet)  #For Lasso regression
library(tidyverse)  #For data manupiliation

df = read.table('/Users/patrikmirzai/Desktop/forestfires.csv', sep=",", header = T)
df$area = log(df$area+1)  #Transform the variables

set.seed(2)

#Index for our train data
train_index = sample(1:nrow(df), size = nrow(df)*0.7, replace = F)


#Select train and test set
train_data = df[train_index,]
test_data = df[-train_index,]


#Model matrix for train data
x = model.matrix(area~., train_data)[,-1]
y = train_data$area

#Plotting the coefficients against different values of lambda
fit = glmnet(x, y)
plot(fit)

#Create a sequence of our tuninig parameter used in the cross validation
set.seed(2)
lambda_seq = 10^seq(2, -2, by = -.1)

#Train model with different tuning parameters
cv_output = cv.glmnet(x, y, alpha = 1, lambda = lambda_seq, type.measure="mse")

#Cross validation plot
plot(cv_output)
best_lam = cv_output$lambda.min

#Fit lasso model again with the best lambda
best_lasso = glmnet(x, y, alpha = 1, lambda = best_lam)

aal = coef(best_lasso) #Get coefficients
xtable(as.matrix(aal))  #Exporting table to latex (optional)

#Predicting
x_test = model.matrix(area~., test_data)[,-1]
pred = predict(best_lasso, x_test)

actual_test = test_data$area
mse = mean((actual_test - pred)^2)


#------------------------
#Linear regression  (Optional)

lm_model <- lm(area ~ ., data = train_data)
pred_lm <- predict(lm_model, newdata = test_data)

MSE_lm = mean((pred_lm - test_data$area)^2) 

#Now performing stepwise regression
#-------------------------
library(leaps)
step.model <- stepAIC(lm_model, direction = "both", 
                      trace = FALSE)
summary(step.model)
stepwise_pred = predict(step.model, newdata = test_data)
mse_stepwise = mean((test_data$area - stepwise_pred)^2)
#---------------------
#Regression trees
library(rpart)
library(rpart.plot)

# grow tree
fit <- rpart(area~.,
             method="anova", data=train_data)

# plot tree
rpart.plot(fit, uniform=TRUE)

predicted_vals = predict(fit, test_data)
mse_tree = mean((actual_test - predicted_vals)^2)

#Compare Lasso, regression, tree and stepwise
cbind(mse, MSE_lm, mse_tree, mse_stepwise)
