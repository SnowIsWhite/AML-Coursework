setwd('C:/Users/Jae/Documents/rdata')
rm(list=ls())
pima <- read.csv("pima-indians-diabetes.txt", header = FALSE)
library(klaR)
library(caret)
features <- pima[,-c(9)]
label <- as.factor(pima[,9])

test_score <- array(dim=10)
#10 folds
for(index in 1:10){
  #divide data into train set and test set
  partition <- createDataPartition(y=label, p=0.8, list = FALSE)
  
  #call svmlight and train svm
  svm <- svmlight(features[partition,], label[partition], pathsvm = 'C:/Users/Jae/Downloads/svm_light_windows64/')
  
  #predict on test set
  prediction <- predict(svm, features[-partition,])
  #take class prediction
  result <- prediction$class
  #calculate accuracy 
  test_score[index] <-sum(result ==label[-partition])/(sum(result==label[-partition])+sum(!(result==label[-partition])))
}
