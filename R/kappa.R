require(irr)
library(xlsx)
library(readxl)

file <- 'D:/workstation_b/limengsi/¼ÓÈ¨Kappa.xlsx'

data1 <- read_excel(file, sheet='2D')
kappa2(data1,'equal')

data2 <- read_excel(file, sheet='3D')
kappa2(data2,'equal')
