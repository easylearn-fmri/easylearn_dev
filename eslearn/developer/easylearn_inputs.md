# easylearn项目用户输入
## 概要
此文档描述easylearn中参数加载格式，用户在图形界面中每一次点击或者输入，在确认后都会保存到一个json文件中。该json文件在用户打开软件，选择了工作目录，并初始化时所创建，并由用户命名。此后所有的参数都会被该文件所记录，并用于最后的执行。  
只有充分了解该文件的结构，也就是软件如何记录用户的输入，以后才能方便后续的开发。  

## 例子
假设用户选择./developer为工作目录，并在此目录创建了一个叫做"configuration_file.json"的文件。
那么空的文件格式为：  
{"data_loading": {}, "features_engineering": {}, "machine_learning": {}, "model_evaluation": {}, "statistical_analysis": {}}    

如果用户通过界面加入一个组（group_0:精神分裂症病人组），以及一个模态，并为这个组下的这个模态（modality_0: 数据类型，比如一组人的全脑功能连接）添加了几个包含全脑功能连接数值的文件，那么json文件为：   
{
    "data_loading": 
        {"group_0": 
            {"modality_0": {"file": ["D:/WorkStation_2018/SZ_classification/Data/FC_1322/ROICorrelation_FisherZ_ROISignal_00002.mat", "D:/WorkStation_2018/SZ_classification/Data/FC_1322/ROICorrelation_FisherZ_ROISignal_00003.mat", "D:/WorkStation_2018/SZ_classification/Data/FC_1322/ROICorrelation_FisherZ_ROISignal_00004.mat", "D:/WorkStation_2018/SZ_classification/Data/FC_1322/ROICorrelation_FisherZ_ROISignal_00005.mat", "D:/WorkStation_2018/SZ_classification/Data/FC_1322/ROICorrelation_FisherZ_ROISignal_00006.mat", "D:/WorkStation_2018/SZ_classification/Data/FC_1322/ROICorrelation_FisherZ_ROISignal_00007.mat"], "mask": "", "targets": [], "covariates": []}}, 
        }, 
    "features_engineering": {}, 
    "machine_learning": {}, 
    "model_evaluation": {}, 
    "statistical_analysis": {}
}   

同理，如果用户通过界面再加入一个组（group_1:正常对照组），以及一个模态，并为这个组下的这个模态（modality_1:数据类型，比如一组人的全脑功能连接）添加了几个包含全脑功能连接数值的文件，那么json文件为：  
{
    "data_loading": 
        {"group_0": 
            {"modality_0": {"file": ["D:/WorkStation_2018/SZ_classification/Data/FC_1322/ROICorrelation_FisherZ_ROISignal_00002.mat", "D:/WorkStation_2018/SZ_classification/Data/FC_1322/ROICorrelation_FisherZ_ROISignal_00003.mat", "D:/WorkStation_2018/SZ_classification/Data/FC_1322/ROICorrelation_FisherZ_ROISignal_00004.mat", "D:/WorkStation_2018/SZ_classification/Data/FC_1322/ROICorrelation_FisherZ_ROISignal_00005.mat", "D:/WorkStation_2018/SZ_classification/Data/FC_1322/ROICorrelation_FisherZ_ROISignal_00006.mat", "D:/WorkStation_2018/SZ_classification/Data/FC_1322/ROICorrelation_FisherZ_ROISignal_00007.mat"], "mask": "", "targets": [], "covariates": []}}, 
        "group_1": 
            {"modality_0": {"file": ["D:/WorkStation_2018/SZ_classification/Data/FC_1322/ROICorrelation_FisherZ_ROISignal_00029.mat", "D:/WorkStation_2018/SZ_classification/Data/FC_1322/ROICorrelation_FisherZ_ROISignal_00030.mat", "D:/WorkStation_2018/SZ_classification/Data/FC_1322/ROICorrelation_FisherZ_ROISignal_00031.mat", "D:/WorkStation_2018/SZ_classification/Data/FC_1322/ROICorrelation_FisherZ_ROISignal_00032.mat", "D:/WorkStation_2018/SZ_classification/Data/FC_1322/ROICorrelation_FisherZ_ROISignal_00033.mat", "D:/WorkStation_2018/SZ_classification/Data/FC_1322/ROICorrelation_FisherZ_ROISignal_00034.mat"], "mask": "", "targets": [], "covariates": []}}}, 
    "features_engineering": {}, 
    "machine_learning": {}, 
    "model_evaluation": {}, 
    "statistical_analysis": {}
}  

同理，用户会通过界面输入所有需要的参数，比如每个组每个模态的mask（可以为空）， targets(labels), covariates(可以为空)，以及必要的特征工程信息，机器学习信息（比如交叉验证方式，算法等）。  

## 执行
假如用户输入了必要的信息，接下来可以进行特征工程，机器学习，模型评估和统计分析。拿机器学习为例子，我们要用全脑功能连接作为特征，来区分精神分裂症和正常对照。由于此时，我们已经获取了features，labels以及其它必要信息（参数）,我们便可以通过这些信息来训练机器学习模型。比如以10折交叉验证的方式训练并测试支持向量机模型。或者以胸部CT影像数据为特征，并划分为训练集，验证集和测试集来训练一个卷积神经网络模型，用于新冠肺炎患者的诊断。

