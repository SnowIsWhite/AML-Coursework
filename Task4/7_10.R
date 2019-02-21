ms305data <- read.csv('C:/Users/Jae/Desktop/ms305data.csv', header = TRUE)

#problem a
#regression
data <- data.frame(ms305data['Mass'],ms305data['Fore'],ms305data['Bicep'],ms305data['Chest'],ms305data['Neck'],ms305data['Shoulder'],ms305data['Waist'],ms305data['Height'],ms305data['Calf'],ms305data['Thigh'],ms305data['Head'])
fit <- lm(Mass~Fore+Bicep+Chest+Neck+Shoulder+Waist+Height+Calf+Thigh+Head, data= data)
plot(fit$fitted.values,fit$residuals, col='red', pch=16)
abline(0,0)
#standard residual
R3 <- var(fit$fitted.values)/var(ms305data['Mass'])
#problem b
#regress the cube root of mass
cube_root_mass <- ms305data['Mass']^(1/3)
data2 <- data.frame(cube_root_mass,ms305data['Fore'],ms305data['Bicep'],ms305data['Chest'],ms305data['Neck'],ms305data['Shoulder'],ms305data['Waist'],ms305data['Height'],ms305data['Calf'],ms305data['Thigh'],ms305data['Head'])
fit2 <- lm(cube_root_mass~Fore+Bicep+Chest+Neck+Shoulder+Waist+Height+Calf+Thigh+Head, data= data2)

#plot in cube root coordinate
plot(fit2$fitted.values, fit2$residuals,xlab= 'fitted value', ylab = 'Residual', col = 'red', pch = 16, main='cube root coordinate')
abline(0,0)

#Standard residual close to 1
R <- var(fit2$fitted.values)/var(ms305data['Mass']^(1/3))
#plot in original coordinate
fit3 <- fit2$fitted.values^3
res <- ms305data['Mass'] - t(fit3)
plot(fit3,t(res),xlab= 'fitted value', ylab = 'Residual', col= 'red', pch =16, main = 'original coordinate')
abline(0,0)

#R square close to 1
R2 <- var(fit3)/var(ms305data['Mass'])
