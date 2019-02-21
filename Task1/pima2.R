setwd("C:/Users/Jae/Documents/rdata")
pima <- read.csv("pima-indians-diabetes.txt", header = FALSE)
setwd("C:/Users/Jae/Documents/rscript")
source("naive_classifier.R")
library(caret)
library(klaR)

label <- pima[,9]
data1 <- pima[,-c(9)]
for(i in c(3,4,6,8)){
  temp <- pima[,i]==0
  pima[temp,i]=NA
}
data2 <- pima[,-c(9)]

#without considering missing value
train_score1 = array(dim=10)
test_score1 = array(dim=10)
#with missing value
train_score2 = array(dim=10)
test_score2 = array(dim=10)
#accuracy difference
train_acc_diff = array(dim=10)
test_acc_diff = array(dim=10)
#without missing value
for(index in 1:10){
  #divide data into partitions for a cross validation 
  partition <- createDataPartition(y=label, p=0.8, list=FALSE)
  train_x <- data1[partition,]
  train_y <- label[partition]
  test_x <- data1[-partition,]
  test_y <- label[-partition]
  
  #train
  train_score1[index] = naive_bayes(pima, index, train_x,train_y, train_x, train_y)
  #test
  test_score1[index] = naive_bayes(pima, index, train_x, train_y, test_x, test_y)
}

#with missing value set to NA
for(index in 1:10){
  #divide data into partitions for a cross validation 
  partition <- createDataPartition(y=label, p=0.8, list=FALSE)
  train_x <- data2[partition,]
  train_y <- label[partition]
  test_x <- data2[-partition,]
  test_y <- label[-partition]
  
  #train
  train_score2[index] = naive_bayes(pima, index, train_x,train_y, train_x, train_y)
  #test
  test_score2[index] = naive_bayes(pima, index, train_x, train_y, test_x, test_y)
}


train_acc_diff = train_score2/train_score1*100
test_acc_diff = test_score2/test_score1*100
