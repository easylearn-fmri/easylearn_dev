library(readxl)
library(rms)
library(survival)


data <- read_excel('D:/workstation_b/wangpeng/coxRR.xlsx')
head(data)

dd=datadist(data)
options(datadist="dd") 

f2 <- psm(Surv(time,status) ~ ALB+Surgery+Tbil+LTS+BCLC,data = data,dist='lognormal') 
med <- Quantile(f2, q=1) # 计算中位生存时间
surv <- Survival(f2) # 构建生存概率函数

nom <- nomogram(f2, fun=list(function(x) surv(365,x),
                             function(x) med(lp=x)),
                lp=T,
                funlabel=c('1-year survival probability','Median Survival Time'))

plot(nom,xfrac=.2, cex.lab=1, cex.axis=0.8, cex=1)


#
library(readxl)
library(ggDCA)
library(rms)

data <- read_excel('D:/workstation_b/wangpeng/coxRR.xlsx')
data <- as.data.frame(data)

model1 <- coxph(Surv(time,status)~LTS+BCLC+Treatafter + Surgery + ALB+Tbil,data)

d <- dca(model1,model.names = 'y', times = 365)  # 一年即365天
ggplot(d)

d <- dca(model1,model.names = 'y',times = "median")  # 中位时间
ggplot(d)
