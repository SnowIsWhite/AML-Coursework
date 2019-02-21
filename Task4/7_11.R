ab_data <- read.csv('C:/Users/Jae/Desktop/abalone.csv', header = FALSE, stringsAsFactors = FALSE)
colnames(ab_data) <- c('Sex', 'Leng', 'Diam', 'Height', 'Whole', 'Shuck', 'Visc','Shell','Rings')

################ problem A ########################
data_a<- data.frame(ab_data['Sex'],ab_data['Leng'],ab_data['Diam'],ab_data['Height'],ab_data['Whole'],ab_data['Shuck'],ab_data['Visc'],ab_data['Shell'],ab_data['Rings'])
fit1 <- lm(Rings~Leng+Diam+Height+Whole+Shuck+Visc+Shell, data = data_a)
plot(fit1, which = 1) #first graph

################ problem B ########################
#change values in the gender column
data_a$Sex[which(data_a$Sex=='M')] <- 1
data_a$Sex[which(data_a$Sex=='F')] <- -1
data_a$Sex[which(data_a$Sex=='I')] <- 0
as.numeric(data_a$Sex)

fit2 <- lm(Rings~Sex+Leng+Diam+Height+Whole+Shuck+Visc+Shell, data = data_a)
plot(fit2, which =1)

################ problem C ########################
#lof of age
data_c <- data.frame(ab_data['Sex'],ab_data['Leng'],ab_data['Diam'],ab_data['Height'],ab_data['Whole'],ab_data['Shuck'],ab_data['Visc'],ab_data['Shell'],log10(ab_data['Rings']))
#remove outlier
data_c <- data_c[-c(2052,237),]
fit3 <- lm(Rings~Leng+Diam+Height+Whole+Shuck+Visc+Shell, data = data_c)
plot(fit3, which = 1)
################ problem D ########################
data_c$Sex[which(data_c$Sex=='M')] <- 1
data_c$Sex[which(data_c$Sex=='F')] <- -1
data_c$Sex[which(data_c$Sex=='I')] <- 0
as.numeric(data_c$Sex)
fit4 <- lm(Rings~Sex+Leng+Diam+Height+Whole+Shuck+Visc+Shell, data = data_c)
plot(fit4, which=1)


############### RELATIONSHIP BETWEEN GENDER AND AGE ##############

plot(data_c$Rings,data_c$Sex, xlab = 'Age', ylab = 'Gender', main = 'Relationship between Age and Gender',col = 'red', pch = 16 )


################ problem F ########################
require(glmnet)
input_x <- ab_data[,1:8]
input_y <- ab_data[,9]
input_x[,1][which(input_x[,1]=='M')] <- 1
input_x[,1][which(input_x[,1]=='F')] <- -1
input_x[,1][which(input_x[,1]=='I')] <- 0
input_x[,1]<-as.numeric(input_x[,1])
input_x[,1] <- as.double(input_x[,1])
input_x <- as.matrix(input_x)
input_y <- as.vector(as.double(input_y))

#a
input_x_a <- input_x[,-c(1)]
cvfita <- cv.glmnet(input_x_a,input_y)
#b
cvfitb <- cv.glmnet(input_x, input_y)
#c
input_y_c <- log(input_y)
cvfitc <- cv.glmnet(input_x_a, input_y_c)
#d
cvfitd <- cv.glmnet(input_x,input_y_c)

#remove outliers
input_x_a <- input_x_a[-c(2052,237),]
input_x <- input_x[-c(2052,237),]
input_y_c <- input_y_c[-c(2052,237)]
#c
cvfitc2 <- cv.glmnet(input_x_a, input_y_c)
#d
cvfitd2 <- cv.glmnet(input_x,input_y_c)

plot(cvfita)
plot(cvfitb)
plot(cvfitc)
plot(cvfitd)
plot(cvfitc2)
plot(cvfitd2)
