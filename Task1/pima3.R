setwd('C:/Users/Jae/Documents/rdata')
pima <- read.csv("pima-indians-diabetes.txt", header = FALSE)
library(klaR)
library(caret)
#divide the data into label and other attributes
features <- pima[,-c(9)]
label <- as.factor(pima[,9])

test_score<- array(dim=10)
for(index in 1:10){
  #divide data into test set and train set
  partition <- createDataPartition(y=label, p=0.8, list = FALSE)
  train_x <- features[partition,]
  train_y <- label[partition]

  #cross-validation, train
  model <- train(train_x, train_y, 'nb', trControl=trainControl(method = 'cv', number= 10))
  
  #test to test set
  test <- predict(model, newdata=features[-partition,])
  stat <-confusionMatrix(data = test, label[-partition])
  
  #get accuracy
  test_score[index] <- stat$overall['Accuracy']
}