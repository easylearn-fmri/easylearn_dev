library(readxl)
library(ggDCA)
library(rms)
library(deLong)

#### Load data ####
data <- read_excel('D:/My_Codes/lc_private_codes/R/demo_data1.xlsx', sheet = 1)
data <- data.frame(data)

#### Classification ####
model1 <- lrm(label ~ Sex + Age, data)
d <- dca(model1,model.names = 'ANLN')
ggplot(d)

model2 <- lrm(status~ANLN+CENPA,LIRI)
d <- dca(model2,model.names = 'ANLN+CENPA')
ggplot(d)

model3 <- lrm(status~ANLN+CENPA+GPR182,LIRI)
d <- dca(model3,model.names = 'ANLN+CENPA+GPR182')
ggplot(d)

model4 <- lrm(status~ANLN+CENPA+GPR182+BCO2,LIRI)
d <- dca(model4,model.names = 'ANLN+CENPA+GPR182+BCO2')
ggplot(d)

d <- dca(model1,model2,model3,model4,
         model.names = c('ANLN',
                         'ANLN+CENPA',
                         'ANLN+CENPA+GPR182',
                         'ANLN+CENPA+GPR182+BCO2'))
ggplot(d,
       linetype = FALSE,
       color = c('blue','green','black','red','gray','gray'))


#### Survial Analysis ####

########## cox regression
# evaluate at median time
model1 <- coxph(Surv(time,status)~ANLN,LIRI)
d <- dca(model1,model.names = 'ANLN')
ggplot(d)

model2 <- coxph(Surv(time,status)~ANLN+CENPA,LIRI)
d <- dca(model2,model.names = 'ANLN+CENPA')
ggplot(d)

model3 <- coxph(Surv(time,status)~ANLN+CENPA+GPR182,LIRI)
d <- dca(model3,model.names = 'ANLN+CENPA+GPR182')
ggplot(d)

model4 <- coxph(Surv(time,status)~ANLN+CENPA+GPR182+BCO2,LIRI)
d <- dca(model4,model.names = 'ANLN+CENPA+GPR182+BCO2')
ggplot(d)

d <- dca(model1,model2,model3,model4,
         model.names = c('ANLN',
                         'LIRI',
                         'ANLN+CENPA',
                         'ANLN+CENPA+GPR182',
                         'ANLN+CENPA+GPR182+BCO2'))
ggplot(d,
       linetype = FALSE,
       color = c('blue','green','black','red','gray','gray'))

# evaluate at different times
qt <- quantile(LIRI$time,c(0.25,0.5,0.75))
qt=round(qt,2)
model1 <- coxph(Surv(time,status)~ANLN,LIRI)
d <- dca(model1,
         model.names = 'ANLN',
         times = qt)
ggplot(d)

model2 <- coxph(Surv(time,status)~ANLN+CENPA,LIRI)
d <- dca(model2,
         model.names = 'ANLN+CENPA',
         times = qt)
ggplot(d)

model3 <- coxph(Surv(time,status)~ANLN+CENPA+GPR182,LIRI)
d <- dca(model3,
         model.names = 'ANLN+CENPA+GPR182',
         times = qt)
ggplot(d)

model4 <- coxph(Surv(time,status)~ANLN+CENPA+GPR182+BCO2,LIRI)
d <- dca(model4,
         model.names = 'ANLN+CENPA+GPR182+BCO2',
         times = qt)
ggplot(d)

d <- dca(model1,model2,model3,model4,
         model.names = c('ANLN',
                         'ANLN+CENPA',
                         'ANLN+CENPA+GPR182',
                         'ANLN+CENPA+GPR182+BCO2'),
         times = qt)
ggplot(d)