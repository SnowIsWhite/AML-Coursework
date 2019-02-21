data <- read.csv('c:/Users/Jae/Desktop/brunhild.csv', header = FALSE)
#data preprocessed to remove name of the fields
x <- data[1]
y <- data[2]

#change data into log
logx <- log10(x)
logy <- log10(y)

brunhild <- data.frame('time' = logx, 'concentration' = logy)
colnames(brunhild)<- c('time','concentration')

#regression
brunhild.lm <- lm(concentration ~ time, data = brunhild)

#plot graph
plot(brunhild, xlab = 'log(time)', ylab = 'log(concentration)', col='red', pch=16)
abline(brunhild.lm, col='blue')


#problem b

#non log data
brunhild_b <- data.frame('time' = x, 'concentration' = y)
colnames(brunhild_b) <- c('time', 'concentration')
plot(brunhild_b, xlab = 'time', ylab = 'concentration', col = 'red', pch = 16)

x_curve <- t(x)
y_curve <-10^(brunhild.lm$coefficients[1] + brunhild.lm$coefficients[2]*log10(x_curve))
lines(x_curve, y_curve)

#problem c
res1 <- brunhild.lm$residuals
fit1 <- brunhild.lm$fitted.values 
plot(fit1, res1, xlab='fitted value(log-log)', ylab = 'residuals', col= 'red', pch = 16)
abline(0,0)

fit2 <- 10^(brunhild.lm$coefficients[1] + brunhild.lm$coefficients[2]*log10(t(x)))
res2 <- t(y) - fit2
plot(fit2, res2, xlab = 'fitted value(original)', ylab = 'residuals', col = 'red', pch = 16)
abline(0,0)
