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


