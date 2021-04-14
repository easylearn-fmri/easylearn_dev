#该数据下载链接：https://www.mskcc.org/sites/default/files/node/4509/documents/decisioncurveanalysis.zip
#该数据就是临床信息数据，若有自己的数据，用自己的数据即可。
data.set <- read.table("dca.txt", header=TRUE, sep="\t")
attach(data.set)
str(data.set)
# 这是一个数据框结构的生存数据，750个观测，10个变量: 
# patientid : 编号 。
# cancer : 是否发生癌症，二分类，1表示罹患癌症，0表示未患癌症。因变量 。
# dead : 是否死亡，二分类，1表示死亡，0表示存活 。
# ttcancer : 从随访开始到发生癌症的时间，连续变量。时间变量 。
# risk_group : 危险因素分组，因子变量，等级变量，3 = “high”, 2 = “intermediate”, 1 =“low” 
# casecontrol : 分组变量，二分类，1 = “case”,0 = “control” $ age : 年龄，连续变量 。
# famhistory : 家族史，0 = no, 1 = yes $ marker : 某标志物水平，连续变量 。
# cancerpredmarker: 肿瘤标志物水平，连续变量。

#使用source()函数载入MSKCC网站上下载的源代码，需提前下载该源代码并保存至当前工作路径中
# 具体下载地址：https://www.mskcc.org/sites/default/files/node/4509/documents/downloadrcode.zip
source("stdca.R")
# 后续我们直接使用该函数定义的生存资料DCA分析的stdca()函数即可。
# 函数用法如下：
# stdca(data, outcome, predictors, timepoint, xstart=0.01, xstop=0.99, xby=0.01, ymin=-0.05, 
# probability=NULL, harm=NULL, graph=TRUE, 
# intervention=FALSE, interventionper=100, smooth=FALSE, loess.span=0.10, cmprsk=FALSE)


#--------------------------------多因素cox的DCA分析-----------------------------------------
library(survival)
#要定义一个生存函数对象，该对象包含研究的结局及发生结局的时间，即本例中即数据框的“cancer”和“ttcancer”两个变量。
Srv = Surv(data.set$ttcancer, data.set$cancer)
#使用survival包中的coxph函数构建Cox回归模型
coxmod <- coxph(Srv ~ age + famhistory + marker, data=data.set)
#根据coxmod生存函数计算1.5年时点的癌症发生率的补数，即未患癌症的发生率
data.set$pr_failure18 <- c(1 - (summary(survfit(coxmod,newdata=data.set), times=1.5)$surv))
#此步骤是必须的，根据前文所述及stdca()函数predictors参数项的规定，此处只能传入一个变量，显然使用模型的预测概率作为新的变量传入反应了整个模型的预测能力。
#此处如果只传入一个预测因素，则仅代表某个因素的因素对于结局的预测能力，而非整个模型的预测能力。

#使用stdca()函数进行DCA分析
stdca(data=data.set, outcome="cancer", ttoutcome="ttcancer", timepoint=1.5, predictors="pr_failure18", xstop=0.5, smooth=TRUE)
#data=data.set指定数据集， outcome=“cancer”定义二分类结局, 
#ttoutcome=“ttcancer”定义时间变量， timepoint=1.5定义时间点1.5年， 
#predictors=“pr_failure18”传入根据Cox回归模型计算的预测概率，此处需要指定这里传入的是概率，probability=TRUE,这也是默认设置。
# 如果用单个因素取值预测，则需设置为FALSE.

#接下来构建两个Cox回归模型
coxmod1 <- coxph(Srv ~ age + famhistory + marker, data=data.set)
coxmod2 <- coxph(Srv ~ age + famhistory + marker + risk_group, data=data.set)

#根据生存函数分别计算2个模型的1.5年时点的癌症发生率的补数，即未患癌症的发生率
data.set$pr_failure19 <- c(1 - (summary(survfit(coxmod1,newdata=data.set), times=1.5)$surv))
data.set$pr_failure20 <- c(1 - (summary(survfit(coxmod2,newdata=data.set), times=1.5)$surv))

#用stdca()函数对两个模型进行DCA分析
stdca(data=data.set, outcome="cancer", ttoutcome="ttcancer", timepoint=1.5,predictors=c("pr_failure19","pr_failure20"), xstop=0.5, smooth=TRUE)


#---------------------------------单因素Cox回归DCA分析------------------------------------------
# 使用MASS包中自带数据集Melanoma.
# 数据框结构，含7个变量，共205观测: 
# time: 时间，连续变量。
# status: 结局变量，1表示死于黑色素瘤，2表示存活，3表示死去其他原因 。
# sex: 性别，1代表男，0代表女 。
# age: 连续变量 $ year: 手术年代，连续变量 。
# thickness: 肿瘤厚度，单位：mm 。
# ulcer: 肿瘤是否溃疡，1代表有溃疡，0代表无溃疡。

source("stdca.R")
library(MASS)
data.set <- Melanoma
data.set$diedcancer = ifelse(data.set$status==1, 1, 0)
stdca(data=data.set, outcome="diedcancer", ttoutcome="time", 
      timepoint=545,predictors="thickness", probability=FALSE, xstop=.25)
# 注意此处我们是用单个变量去预测结局，故“probability=FALSE”,其他参数设置基本相同！
