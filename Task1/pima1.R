setwd("C:/Users/Jae/Documents/rdata")
pima <- read.csv('pima-indians-diabetes.txt', header = FALSE)
library(klaR)
library(caret)
setwd("C:/Users/Jae/Documents/rscript")
source("naive_classifier.R")
#label from data
label <- pima[,9]
#exclude label from data
tdata<- pima[,-c(9)]

train_score <- array(dim =10)
test_score <- array(dim=10)

#train 10 times
for(index in 1:10){
  #divide data into partitions for a cross validation 
  partition <- createDataPartition(y=label, p=0.8, list=FALSE)
  train_x <- tdata[partition,]
  train_y <- label[partition]
  test_x <- tdata[-partition,]
  test_y <- label[-partition]
 
  #train
  train_score[index] = naive_bayes(pima, index, train_x,train_y, train_x, train_y)
  #test
  test_score[index] = naive_bayes(pima, index, train_x, train_y, test_x, test_y)
}