naive_bayes<- function(data,index, train_x,train_y, x, y){
  #select data with label =1
  maskflag <- train_y>0
  posdat <- train_x[maskflag,]
  #select data with label = 0
  negdat <- train_x[!maskflag,]
  
  #evaluate mean and sd, ignore NA values
  pos_dat_mean <- sapply(posdat, mean, na.rm=TRUE)
  pos_dat_sd <- sapply(posdat, sd, na.rm = TRUE)
  
  neg_dat_mean <- sapply(negdat, mean, na.rm=TRUE)
  neg_dat_sd <- sapply(negdat, sd, na.rm=TRUE)
  
  #assume normal distribution and calculate P(x_i|+)
  ptemp1<- t(t(x)-pos_dat_mean)
  ptemp2<- t(t(ptemp1)/pos_dat_sd)
  pval <- -(1/2)*rowSums(apply(ptemp2,c(1,2),function(x) x^2), na.rm=TRUE)-sum(log(pos_dat_sd))+log(nrow(posdat)/nrow(data))
  
  #P(x_i|-)
  ntemp1<- t(t(x)-neg_dat_mean)
  ntemp2<- t(t(ntemp1)/neg_dat_sd)
  nval <- -(1/2)*rowSums(apply(ntemp2,c(1,2),function(x) x^2), na.rm=TRUE)-sum(log(neg_dat_sd))+log(nrow(negdat)/nrow(data))
  
  isdiabetes <- pval>nval
  isright <- isdiabetes == y
  
  score <- sum(isright)/(sum(isright)+sum(!isright))
  return(score)
}