rm(list = ls())
data <- read.csv('C:/Users/Jae/Desktop/I2000.txt', sep=" ", header = FALSE)
y <- read.csv('C:/Users/Jae/Desktop/tissue.txt', header= FALSE)

#preprocess data: classify tumor/normal into 1/0
for(i in 1:62){
  if (y[i,1] >0){
    y[i,1]<- 0
  }else{
    y[i,1]<- 1
  }
}

fit_auc = cv.glmnet(as.matrix(t(data)), as.matrix(y), alpha = 1, family = 'binomial', type.measure = 'auc', nfolds = 5)
fit_deviance = cv.glmnet(as.matrix(t(data)), as.matrix(y), alpha = 1, family = 'binomial', type.measure = 'deviance', nfolds = 5)

plot(fit_auc)
plot(fit_deviance)
