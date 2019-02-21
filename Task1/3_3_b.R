setwd("C:/Users/Jae/Documents/rdata")
data <- read.csv('processed.cleveland.txt', header = FALSE)
library(klaR)
library(caret)

#drop the row with missing value
for(i in 1:13){
  temp <- data[,i]!='?'
  data = data[temp,]
}
tdata <- data[,-c(14)]
label <- as.factor(data[,14])

test_score <- array(dim = 10)
#10 splits
for(index in 1:10){
  partition <- createDataPartition(y=label, p=0.85, list = FALSE)
  train_x <- tdata[partition,]
  train_y <- label[partition]
  
  #10 fold cross-falidation
  model <- train(train_x, train_y, 'nb', trControl = trainControl(method = 'cv', number =10))
  test <- predict(model, newdata=tdata[-partition,])
  #compute accuracy
  stat <- confusionMatrix(data = test, label[-partition])
  #get accuracy
  test_score[index] <- stat$overall['Accuracy']
}
#compute mean and standard deviation
mean <- mean(test_score)
sd <- sd(test_score)
