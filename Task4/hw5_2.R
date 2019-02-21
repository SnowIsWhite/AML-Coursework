#preprocess: Delete the second row
rm(list = ls())
data <- read.csv('C:/Users/Jae/Desktop/default of credit card clients.csv', header= TRUE)
features <- data[,-c(25)]
y <- data[,c(25)]

library(pscl)
############regularize############
#ridge regression(misclassification error, deviance, auc in order)
fit_ridge = cv.glmnet(as.matrix(features), as.matrix(y), family = 'binomial', alpha = 0, type.measure = 'class', nfold = 10)
fit_ridge2 = cv.glmnet(as.matrix(features), as.matrix(y), family = 'binomial', alpha = 0, type.measure = 'deviance', nfold = 10)
fit_ridge3 = cv.glmnet(as.matrix(features), as.matrix(y), family = 'binomial', alpha = 0, type.measure = 'auc', nfold = 10)

#lasso regression
fit_lasso = cv.glmnet(as.matrix(features), as.matrix(y), family = 'binomial', alpha = 1, type.measure = 'class', nfold = 10)
fit_lasso2 = cv.glmnet(as.matrix(features), as.matrix(y), family = 'binomial', alpha = 1, type.measure = 'deviance', nfold = 10)
fit_lasso3 = cv.glmnet(as.matrix(features), as.matrix(y), family = 'binomial', alpha = 1, type.measure = 'auc', nfold = 10)

#elastic net
fit_elastic = cv.glmnet(as.matrix(features), as.matrix(y), family = 'binomial', alpha = 0.5, type.measure = 'class', nfold = 10)
fit_elastic2 = cv.glmnet(as.matrix(features), as.matrix(y), family = 'binomial', alpha = 0.5, type.measure = 'deviance', nfold = 10)
fit_elastic3 = cv.glmnet(as.matrix(features), as.matrix(y), family = 'binomial', alpha = 0.5, type.measure = 'auc', nfold = 10)

