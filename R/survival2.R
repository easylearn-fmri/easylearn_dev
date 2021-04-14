library(survival)
library(rms)

data(package="survival")

dd<-datadist(lung)
options(datadist="dd")

f <- cph(Surv(time, status) ~ age+sex+ph.karno, data=lung,
    x=TRUE, y=TRUE, surv=TRUE)

survival <- survival(f)
survival1 <- function(x)survival(365, x)
survival2 <- function(x)survival(730, x)

## 绘制logisitc回归的风险预测值的nomogram图
nom <- nomogram(f, fun = list(survival1, survival2), 
                fun.at=c(0.05, seq(0.1, 0.9, by=0.05)),
                funlabel=c('1 year, survival, 2 year survival'))
plot(nom)