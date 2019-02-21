rm(list = ls())
data <- read.table('C:/Users/Jae/Desktop/default_features_1059_tracks.txt', header= FALSE, sep = ',')
latitude <- data[,c(69)]
longitude <- data[,c(70)]
attributes <- data[,-c(69,70)]

library(glmnet)
library(MASS)
#############linear regression############
data_lat <- data.frame(latitude,attributes)
reg_fit1 <-lm(latitude ~ ., data = data_lat)

data_long <- data.frame(longitude, attributes)
reg_fit2 <- lm(longitude~.,data= data_long)
#plot(reg_fit1, which = 5)
#plot(reg_fit2, which = 5)

#R squared
r_fit1 <- var(reg_fit1$fitted.values)/var(latitude)
r_fit2 <- var(reg_fit2$fitted.values)/var(longitude)
#############Box Cox transformation############
#transfrom dependent variables to be positive
trans_lat <- latitude + 360
trans_long <- longitude + 360
data_trans_lat <- data.frame(trans_lat, attributes)
data_trans_long <- data.frame(trans_long, attributes)
#boxcox
bc_lat <- boxcox(lm(trans_lat ~., data= data_trans_lat), lambda = seq(-20,20))
bc_long <- boxcox(lm(trans_long~., data= data_trans_long), lambda = seq(-10,10))

#get lambda with the maximum log likelihood
lat_lambda <- bc_lat$x[which.max(bc_lat$y)]
long_lambda <- bc_long$x[which.max(bc_long$y)]

#regrerssion with lambda
reg_fit1_new <- lm(((trans_lat^lat_lambda)-1)/lat_lambda~.,data= data_lat)
reg_fit2_new <- lm(((trans_long^long_lambda)-1)/long_lambda~., data= data_long)
#r squared(boxcox)
r_fit1_new <- var(reg_fit1_new$fitted.values)/var(((trans_lat^lat_lambda)-1)/lat_lambda)
r_fit2_new <- var(reg_fit2_new$fitted.values)/var(((trans_long^long_lambda)-1)/long_lambda)

#############regression regularization #############
#ridge regression
lat_cv0 <- cv.glmnet(as.matrix(attributes),as.matrix(((trans_lat^lat_lambda)-1)/lat_lambda), alpha = 0, nfold = 10)
plot(lat_cv0)
long_cv0 <- cv.glmnet(as.matrix(attributes),as.matrix(((trans_long^long_lambda)-1)/long_lambda), alpha = 0, nfold = 10)
plot(long_cv0)

#predicted value
p1 <-predict(lat_cv0, newx=as.matrix(attributes), s = "lambda.min")
p2 <- predict(long_cv0, newx= as.matrix(attributes), s="lambda.min")
#r sqaure
r_lat_ridge <- var(p1)/var(((trans_lat^lat_lambda)-1)/lat_lambda)
r_long_ridge <- var(p2)/var(((trans_long^long_lambda)-1)/long_lambda)

#lasso regression
lat_cv1 <- cv.glmnet(as.matrix(attributes), as.matrix(((trans_lat^lat_lambda)-1)/lat_lambda), alpha = 1, nfold = 10)
plot(lat_cv1)
long_cv1 <- cv.glmnet(as.matrix(attributes), as.matrix(((trans_long^long_lambda)-1)/long_lambda),alpha = 1, nfold = 10)
plot(long_cv1)

#predict values
p3 <- predict(lat_cv1, newx = as.matrix(attributes), s = "lambda.min")
p4 <- predict(long_cv1, newx= as.matrix(attributes), s = "lambda.min")
r_lat_lasso <- var(p3)/var(((trans_lat^lat_lambda)-1)/lat_lambda)
r_long_lasso <- var(p4)/var(((trans_long^long_lambda)-1)/long_lambda)
