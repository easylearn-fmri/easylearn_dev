library(Hmisc); library(grid); library(lattice);library(Formula); library(ggplot2) 
library(rms)

## 第二步 读取数据，以survival程序包的lung数据来进行演示
## 列举survival程序包中的数据集
library(survival)
data(package = "survival")

## 读取lung数据集
data(lung)

## 显示lung数据集的前6行结果
head(lung)

## 显示lung数据集的变量说明
help(lung)

## 添加变量标签以便后续说明
lung$sex <- 
    factor(lung$sex,
           levels = c(1,2),
           labels = c("male", "female"))

## 第三步 按照nomogram要求“打包”数据，绘制nomogram的关键步骤,??datadist查看详细说明
dd=datadist(lung)
options(datadist="dd") 

## 第四步 构建模型
## 构建logisitc回归模型
f1 <- lrm(status~ age + sex, data = lung) 

## 绘制logisitc回归的风险预测值的nomogram图
nom <- nomogram(f1, fun= function(x)1/(1+exp(-x)), # or fun=plogis
                lp=F, funlabel="Risk")
plot(nom)
