setwd('C:/Users/Jae/Documents/rdata')
library(klaR)
library(caret)
require(ggplot2)
#data
rawdata <- read.csv("k9.csv", header=FALSE, nrow=10000, colClasses = c(rep("numeric", 5408), "character", NA), na.strings="?")
#drop na column
data <- rawdata[-c(5410)]
#drop missing values
temp <- is.na(data[,1])
data = data[!temp,]

#change class contriubute into +1/-1
temp <- data[,5409]=='active'
data[temp,5409] = 1
temp <- data[,5409]=='inactive'
data[temp,5409] = -1
data[,5409] <- sapply(data[,5409],as.integer)
features <- data[,-c(5409)]
label <- data[,5409]

#variables
test_acc <- array(dim=1)
a <- list()
b <- array(dim=3)
num_lambda<-3
lambda <- c(0.001, 0.05, 0.1)
step_len <- 0.01
seasons <- 100
steps <- 50
graph_acc1 <- array(dim=seasons)
graph_acc2 <- array(dim=seasons)
graph_acc3 <- array(dim=seasons)
for(i in 1){
  #data partition in random
  #seperate1(final check)
  
  partition1 <- createDataPartition(y= label, p=0.9, list = FALSE )
  train_x1 <- features[partition1,]
  train_y1 <- label[partition1]
  test_x <- features[-partition1,]
  test_y <- label[-partition1]
  #normalize
  
  for(j in 1:5408){
    mean <- sum(train_x1[,j])/nrow(train_x1)
    sd <- sd(train_x1[,j])
    if(sd!=0){
    train_x1[,j]<-(train_x1[,j]-mean)/sd
    test_x[,j]<-(test_x[,j]-mean)/sd
    }else{
      train_x1[,j] <- 0
      test_x[,j]<-0
    }
  }
  #seperate2(lambda check)

  partition2 <- createDataPartition(y = train_y1, p=0.85, list=FALSE)
  train_x2 <- train_x1[partition2,]
  train_y2 <- train_y1[partition2]
  validate_x <- train_x1[-partition2,]
  validate_y <- train_y1[-partition2]
  
  for(index in 1:num_lambda){
    #choose a,b at random(start point)
    start <- Sys.time()
    a[[index]]<-runif(ncol(features),0.5,5.0)
    b[index]<-runif(1,0.5,5.0)
    end <- Sys.time()
    sprintf("time:%f",end-start)
    for(i3 in 1:seasons){
      #update step length
      step_len <- 1/(0.01*seasons+50)
      #seperate3(a,b check)
      partition3 <- createDataPartition(y = train_y2, p=0.85, list = FALSE)
      train_x3 <- train_x2[partition3,]
      train_y3 <- train_y2[partition3]
      check_x <- train_x2[-partition3,]
      check_y <- train_y2[-partition3]
      
      for(i4 in 1:steps){
        #choose one sample uniformly and randomly
        sample <- sample(1:nrow(train_x3), 1, replace=TRUE)
        sample_x <- (train_x3[sample,])
        sample_y <- train_y3[sample]
        #calculate gradient(a,b) and update
        if(nrow(as.matrix(t(a[[index]])))==1){
          yx <- sample_y*(((as.matrix(t(a[[index]]))) %*% as.matrix(t(sample_x)))+b[index])
        }else{
          yx <- sample_y*(((as.matrix((a[[index]]))) %*% as.matrix(t(sample_x)))+b[index])
        }
        if(yx>=1){
          a[[index]]<-a[[index]]*(1- step_len*lambda[index])
        }else{
          a[[index]]<-a[[index]]*(1- step_len*lambda[index])+step_len*(sample_y*sample_x)
          b[index] <- b[index] + step_len*sample_y
        }
      }
      
      #check accuracy of updated a,b
      acc <- 0
      for(i4 in 1:nrow(check_x)){
        if(nrow(as.matrix(t(a[[index]])))==1){
          predict <- (as.matrix(t(a[[index]]))%*%as.matrix(t(check_x[i4,])))+b[index]
        }else{
          predict <- (as.matrix((a[[index]]))%*%as.matrix(t(check_x[i4,])))+b[index]
        }
        if((predict*check_y[i4])>=0){
          #correct
          acc <- acc+1
        }else{}
      }
      if(index ==1){#lambda1
        graph_acc1[i3]<- acc/nrow(check_x)*100
      }else if(index ==2){#lambda2
        graph_acc2[i3]<- acc/nrow(check_x)*100
      }else{#lambda3
        graph_acc3[i3]<- acc/nrow(check_x)*100
      }
    }
  }#for index ends
  
  
  #plot graph
  epoch <- seq(1,seasons,1)
  graph <- data.frame(epoch,graph_acc1,graph_acc2,graph_acc3)  #ggplot object
  graph<-ggplot(graph, aes(epoch)) +geom_line(aes(y=graph_acc1), colour="red") + geom_line(aes(y=graph_acc2), colour="green")+geom_line(aes(y=graph_acc3), colour="blue")
  graph <- graph+ylim(0,100)
  
  #test lambda,a,b in partition2 validation set
  max_index <- 1
  max_val <-0
  for(index in 1:num_lambda){
    valid_acc <- array(dim=3)
    acc <- 0
    #get accuracy in validation set
    for(i2 in 1:nrow(validate_x)){
      if(nrow(as.matrix(t(a[[index]])))==1){
        predict <- (as.matrix(t(a[[index]]))%*%as.matrix(t(validate_x[i2,])))+b[index]
      }else{
        predict <- (as.matrix((a[[index]]))%*%as.matrix(t(validate_x[i2,])))+b[index]
      }
      if((predict*validate_y[i2])>=0){
        #correct
        acc <- acc+1
      }else{}
    }
    valid_acc[index] <- acc/nrow(validate_x)*100
    #get best lambda(index)
    if(max_val<valid_acc[index]){
      max_val <-valid_acc[index]
      max_index<-index
    }
  }
  
  #test accuracy in partition1 test set
  acc<-0
  for(i2 in 1:nrow(test_x)){
    if(nrow(as.matrix(t(a[[index]])))==1){
      predict <- (as.matrix(t(a[[max_index]]))%*%as.matrix(t(test_x[i2,])))+b[max_index]
    }else{
      predict <- (as.matrix((a[[max_index]]))%*%as.matrix(t(test_x[i2,])))+b[max_index]
    }
    if((predict*test_y[i2])>=0){
      #correct
      acc <- acc+1
    }else{}
  }
  test_acc[i]<-acc/nrow(test_x)*100
}


#naive_bayes
test_score = array(dim=1)
label <- sapply(label, as.character)
nbpartition <- createDataPartition(y=label, p=0.9, list = FALSE)
nbtrain_x <- features[nbpartition,]
nbtrain_y <- label[nbpartition]

#cross-validation, train
model <- train(nbtrain_x, nbtrain_y, 'nb', trControl=trainControl(method = 'cv', number= 10))

#test to test set
test <- predict(model, newdata=features[-nbpartition,])
stat <-confusionMatrix(data = test, label[-nbpartition])

#get accuracy
test_score[1] <- stat$overall['Accuracy']
