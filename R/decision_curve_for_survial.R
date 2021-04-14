#�������������ӣ�https://www.mskcc.org/sites/default/files/node/4509/documents/decisioncurveanalysis.zip
#�����ݾ����ٴ���Ϣ���ݣ������Լ������ݣ����Լ������ݼ��ɡ�
data.set <- read.table("dca.txt", header=TRUE, sep="\t")
attach(data.set)
str(data.set)
# ����һ�����ݿ�ṹ���������ݣ�750���۲⣬10������: 
# patientid : ��� ��
# cancer : �Ƿ�����֢�������࣬1��ʾ���֢��0��ʾδ����֢������� ��
# dead : �Ƿ������������࣬1��ʾ������0��ʾ��� ��
# ttcancer : ����ÿ�ʼ��������֢��ʱ�䣬����������ʱ����� ��
# risk_group : Σ�����ط��飬���ӱ������ȼ�������3 = ��high��, 2 = ��intermediate��, 1 =��low�� 
# casecontrol : ��������������࣬1 = ��case��,0 = ��control�� $ age : ���䣬�������� ��
# famhistory : ����ʷ��0 = no, 1 = yes $ marker : ĳ��־��ˮƽ���������� ��
# cancerpredmarker: ������־��ˮƽ������������

#ʹ��source()��������MSKCC��վ�����ص�Դ���룬����ǰ���ظ�Դ���벢��������ǰ����·����
# �������ص�ַ��https://www.mskcc.org/sites/default/files/node/4509/documents/downloadrcode.zip
source("stdca.R")
# ��������ֱ��ʹ�øú����������������DCA������stdca()�������ɡ�
# �����÷����£�
# stdca(data, outcome, predictors, timepoint, xstart=0.01, xstop=0.99, xby=0.01, ymin=-0.05, 
# probability=NULL, harm=NULL, graph=TRUE, 
# intervention=FALSE, interventionper=100, smooth=FALSE, loess.span=0.10, cmprsk=FALSE)


#--------------------------------������cox��DCA����-----------------------------------------
library(survival)
#Ҫ����һ�����溯�����󣬸ö�������о��Ľ�ּ�������ֵ�ʱ�䣬�������м����ݿ�ġ�cancer���͡�ttcancer������������
Srv = Surv(data.set$ttcancer, data.set$cancer)
#ʹ��survival���е�coxph��������Cox�ع�ģ��
coxmod <- coxph(Srv ~ age + famhistory + marker, data=data.set)
#����coxmod���溯������1.5��ʱ��İ�֢�����ʵĲ�������δ����֢�ķ�����
data.set$pr_failure18 <- c(1 - (summary(survfit(coxmod,newdata=data.set), times=1.5)$surv))
#�˲����Ǳ���ģ�����ǰ��������stdca()����predictors������Ĺ涨���˴�ֻ�ܴ���һ����������Ȼʹ��ģ�͵�Ԥ�������Ϊ�µı������뷴Ӧ������ģ�͵�Ԥ��������
#�˴����ֻ����һ��Ԥ�����أ��������ĳ�����ص����ض��ڽ�ֵ�Ԥ����������������ģ�͵�Ԥ��������

#ʹ��stdca()��������DCA����
stdca(data=data.set, outcome="cancer", ttoutcome="ttcancer", timepoint=1.5, predictors="pr_failure18", xstop=0.5, smooth=TRUE)
#data=data.setָ�����ݼ��� outcome=��cancer�������������, 
#ttoutcome=��ttcancer������ʱ������� timepoint=1.5����ʱ���1.5�꣬ 
#predictors=��pr_failure18���������Cox�ع�ģ�ͼ����Ԥ����ʣ��˴���Ҫָ�����ﴫ����Ǹ��ʣ�probability=TRUE,��Ҳ��Ĭ�����á�
# ����õ�������ȡֵԤ�⣬��������ΪFALSE.

#��������������Cox�ع�ģ��
coxmod1 <- coxph(Srv ~ age + famhistory + marker, data=data.set)
coxmod2 <- coxph(Srv ~ age + famhistory + marker + risk_group, data=data.set)

#�������溯���ֱ����2��ģ�͵�1.5��ʱ��İ�֢�����ʵĲ�������δ����֢�ķ�����
data.set$pr_failure19 <- c(1 - (summary(survfit(coxmod1,newdata=data.set), times=1.5)$surv))
data.set$pr_failure20 <- c(1 - (summary(survfit(coxmod2,newdata=data.set), times=1.5)$surv))

#��stdca()����������ģ�ͽ���DCA����
stdca(data=data.set, outcome="cancer", ttoutcome="ttcancer", timepoint=1.5,predictors=c("pr_failure19","pr_failure20"), xstop=0.5, smooth=TRUE)


#---------------------------------������Cox�ع�DCA����------------------------------------------
# ʹ��MASS�����Դ����ݼ�Melanoma.
# ���ݿ�ṹ����7����������205�۲�: 
# time: ʱ�䣬����������
# status: ��ֱ�����1��ʾ���ں�ɫ������2��ʾ��3��ʾ��ȥ����ԭ�� ��
# sex: �Ա�1�����У�0����Ů ��
# age: �������� $ year: ����������������� ��
# thickness: ������ȣ���λ��mm ��
# ulcer: �����Ƿ�����1����������0����������

source("stdca.R")
library(MASS)
data.set <- Melanoma
data.set$diedcancer = ifelse(data.set$status==1, 1, 0)
stdca(data=data.set, outcome="diedcancer", ttoutcome="time", 
      timepoint=545,predictors="thickness", probability=FALSE, xstop=.25)
# ע��˴��������õ�������ȥԤ���֣��ʡ�probability=FALSE��,�����������û�����ͬ��