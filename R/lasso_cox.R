##加载包 明确每个包的作用
library(glmnet) ##Lasso回归
library(rms)  ## 画列线图；
library(VIM) ## 包中aggr()函数，判断数据缺失情况
library(survival) ##  生存分析包
#读取数据集
dt <-read.csv("cancer.csv")

str(dt)  ##查看每个变量结构
 aggr(dt,prop=T,numbers=T) #判断数据缺失情况，红色表示有缺失。
 dt <- na.omit(dt) 按行删除缺失值

 #用for循环语句将数值型变量转为因子变量
for(i in names(dt)[c(4:9)]) {dt[,i] <- as.factor(dt[,i])}
##筛选变量前，首先将自变量数据（因子变量）转变成矩阵（matrix）
## Lasso要求的数据类型
x.factors <- modtel.matrix(~ dt$sex+dt$trt+dt$bui+dt$ch+dt$p+dt$stage,dt)[,-1]
#将矩阵的因子变量与其它定量边量合并成数据框，定义了自变量。
x <- as.matrix(dtata.frame(x.factors,dt[,3]))
#设置应变量，打包生存时间和生存状态（生存数据）
y <- data.matrix(Surv(dt$time,dt$censor))

#调用glmnet包中的glmnet函数，注意family那里一定要制定是“cox”，如果是做logistic需要换成"binomial"。
fit <-glmnet(x,y,family = "cox",alpha = 1)
plot(fit,label=T)
plot(fit,xvar="lambda",label=T)
#主要在做交叉验证,lasso
fitcv <- cv.glmnet(x,y,family="cox", alpha=1,nfolds=10)
plot(fitcv)
coef(fitcv, s="lambda.min")
##
#9 x 1 sparse Matrix of class "dgCMatrix"                1
##d.sex1    .       
##d.trt1    .       
##d.bui1    .       
##d.ch2     .       
##d.ch3     .       
##d.ch4    -0.330676
##d.p1      .       
##d.stage4  .       
##d...3.    .

#拟合cox回归
coxm <- cph(Surv(time,censor==1)~age+sex+trt+bui+ch+p+stage,x=T,y=T,data=dt,surv=T) 
cox.zph(coxm)#等比例风险假定
##       chisq df     p
##age    1.993  1 0.158
##sex    0.363  1 0.547
##trt    3.735  1 0.053
##bui    2.587  1 0.108
##ch     0.296  1 0.587
##p      0.307  1 0.579
##stage  0.395  1 0.530
##GLOBAL 9.802  7 0.200

###开始cox nomo graph
surv <- Survival(coxm) # 建立生存函数

surv1 <- function(x)surv(1*3,lp=x) # 定义time.inc,3月OS
surv2 <- function(x)surv(1*6,lp=x) # 定义time.inc,6月OS
surv3 <- function(x)surv(1*12,lp=x) # 定义time.inc,1年OS

dd<-datadist(dt) #设置工作环境变量，将数据整合
options(datadist='dd') #设置工作环境变量，将数据整合

plot(nomogram(coxm,
              fun=list(surv1,surv2,surv3),
              lp= F,
              funlabel=c('3-Month Survival','6-Month survival','12-Month survival'),
              maxscale=100,
              fun.at=c('0.9','0.85','0.80','0.70','0.6','0.5','0.4','0.3','0.2','0.1')),
     xfrac=.45)
#maxscale 参数指定最高分数，一般设置为100或者10分
#fun.at 设置生存率的刻度
#xfrac 设置数值轴与最左边标签的距离，可以调节下数值观察下图片变化情况
plot(nomogram)

##模型验证
#Concordance index
f<-coxph(Surv(time,censor==1)~age+sex+trt+bui+ch+p+stage,data=d)
sum.surv<-summary(f)
c_index<-sum.surv$concordance
c_index  ##
##C      se(C) 
##0.55396619 0.07664425