library(readxl)
library(rms)
library(survival)

nomo <- function(data, format_str){
  data = data.frame(data)

  # dd=datadist(data)
  # options(datadist="dd")
  
  format_str <- parse(text = format_str)
  f1 <- eval(format_str)
  f <- lrm(f1, data = data)
  
  nomo <- nomogram(f, fun= function(x)1/(1+exp(-x)),
                  lp=T, 
                  # lp.at = seq(-3,4,by=0.5),
                  fun.at = c(seq(0.1,0.9, by=0.1)),
                  funlabel="Risk")

  plot(nomo, col.conf = c('red','green'))
  
}


data <- read_excel('D:/My_Codes/lc_private_codes/R/demo_data1.xlsx', sheet = 1)
dd=datadist(data)
options(datadist="dd")
format_str = "label ~ Sex + Age"
nomo(data, format_str)
# 
