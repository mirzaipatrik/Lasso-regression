# Lasso-regression
Regularized regression of a forest fire data

### Authors: Patrik Mirzai and Huixing Zhong

This project aims at predicting the burned area of wildfires using Lasso regression. Moreover, a comparison to Multiple regression and regression trees is also carried out. A summary of the Lasso procedure is given below. See the attached source code "Lasso implementation of wildfires data set.R" for full details on the project. 

## Upload packages and read the data set

```r
#Upload packages
library(readxl)  #For latex output
library(glmnet)  #For Lasso regression
library(tidyverse)  #For data manupiliation

df = read.table('forestfires.csv', sep=",", header = T)
df$area = log(df$area+1)  #Transform the variables

set.seed(2)

```

Let's divide the data into a train- and test set

```r
#Index for our train data
train_index = sample(1:nrow(df), size = nrow(df)*0.7, replace = F)

#Select train and test set
train_data = df[train_index,]
test_data = df[-train_index,]

#Model matrix for train data
x = model.matrix(area~., train_data)[,-1]
y = train_data$area
```

Let's plot the coefficients against the L1 norm
![grouped](https://github.com/mirzaipatrik/Lasso-regression/blob/master/coefficients.png)

Now let's choose the tuning parameter lambda through cross-validation

```r
#Create a sequence of our tuninig parameter used in the cross validation
lambda_seq = 10^seq(2, -2, by = -.1)

#Train model with different tuning parameters
set.seed(2)
cv_output = cv.glmnet(x, y, alpha = 1, lambda = lambda_seq, type.measure="mse")

#Cross validation plot
plot(cv_output)
best_lam = cv_output$lambda.min

#Fit lasso model again with the best lambda
best_lasso = glmnet(x, y, alpha = 1, lambda = best_lam)

coef(best_lasso) #Get coefficients
```
The plot displays the mean squared error using 10-fold cross validation
![grouped](https://github.com/mirzaipatrik/Lasso-regression/blob/master/cv_error.png)

Finally, let's compute the mean squared error of the test data
```r
#Predicting
x_test = model.matrix(area~., test_data)[,-1]
pred = predict(best_lasso, x_test)

actual_test = test_data$area
mse = mean((actual_test - pred)^2)  #mse is 2.049039
```
