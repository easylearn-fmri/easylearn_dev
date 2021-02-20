library(readxl)
library(rms)
library(survival)
library(pROC) 

data <- read_excel('D:/workstation_b/wangpeng/R-30.xlsx')
data <- as.data.frame(data)

dd=datadist(data)
options(datadist="dd") 

f1 <- lrm(month ~ LTS + ALB + TBil, data = data, x=T, y=T)
f2 <- lrm(month ~ CLIF, data = data, x=T, y=T)
gfit1 <- roc(month~predict(f1), data = data)
gfit2 <- roc(month~predict(f2), data = data)


plot(gfit1)
par(new=T)#在这个图上再画图
plot(gfit2)
