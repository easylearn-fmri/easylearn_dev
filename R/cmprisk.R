library(cmprsk)

set.seed(10)
ftime <- rexp(200)
fstatus <- sample(0:2,200,replace=TRUE)
cov <- matrix(runif(600),nrow=200)
dimnames(cov)[[2]] <- c('x1','x2','x3')
print(z <- crr(ftime,fstatus,cov))
summary(z)
z.p <- predict(z,rbind(c(.1,.5,.8),c(.1,.5,.2)))
plot(z.p,lty=1,color=2:3)
crr(ftime,fstatus,cov,failcode=2)
# quadratic in time for first cov
crr(ftime,fstatus,cov,cbind(cov[,1],cov[,1]),function(Uft) cbind(Uft,Uft^2))
#additional examples in test.R


set.seed(2)
ss <- rexp(100)
gg <- factor(sample(1:3,100,replace=TRUE),1:3,c('a','b','c'))
cc <- sample(0:2,100,replace=TRUE)
strt <- sample(1:2,100,replace=TRUE)
print(xx <- cuminc(ss,cc,gg,strt))
plot(xx,lty=1,color=1:6)
# see also test.R, test.out