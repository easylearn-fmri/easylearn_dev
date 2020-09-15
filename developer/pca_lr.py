#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""This demo was copy and revised from sklearn

此是为开发者准备的一个简单的开发demo
假设用户希望
    1. 用主成分分析（PCA）来执行特征降维,并且设置了参数优化范围和参数个数，即max_components = 1,
    min_components = 0.5, number = 5
    2. 用方差分析来筛选特征，并且设置了参数优化范围和参数个数，即max_number = 1(占总特征数的比例),
    min_number = 0.5(占总特征数的比例), number = 5
    3. 用逻辑回归来作为分类器，并且设置了参数优化范围和参数个数，即max_l1_ratio = 1,
    min_l1_ratio = 0, number = 5。注：逻辑回归使用了l1正则。

"""


# In[2]:


# 导入相应模块
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from joblib import Memory
from shutil import rmtree

from eslearn.model_evaluation.el_evaluation_model_performances import eval_performance


# In[3]:


# 加载数据集：此处直接使用生成的数据
# 实际使用中，我们需要根据不同的数据类型，来加载不同的数据。并且需要检查用户给定的数据是否符合要求。
# 因此，此处需要我们设计一个检查数据的模块。
X, y = make_classification(n_features=20, n_samples=200, n_classes=2, n_redundant=0, n_informative=2)

# 确定训练集和测试集，在实际中此处会通过交叉验证的方式执行。
# 此处，需要我们设计交叉验证的模块。
X_train, y_train = X[:80], y[:80]
X_test, y_test = X[80:], y[80:]


# In[ ]:


# 数据预处理
# 在实际使用中，在开始机器学习之前，我们需要进行数据的一个标准化，比如z-score标准化，或者scaling归一化。
# easylearn的eslearn.feature_preprocessing.el_preprocessing模块可以实现这两个功能。
# 另外，我们也需要进一步扩展这个模块，以增加更多的标准化方法。


# In[ ]:


# 不平衡的处理
# 在实际使用中，当几类数据很不平衡时，我们需要处理这种不平衡。不然，占多数的类会影响模型的训练。比如正类样本有90个，负类样本只有10个，
# 那么模型会倾向于把所有样本都预测为正类，因为无论如何这样都可以得到90%的accuracy。
# 目前easylearn使用第三方imblearn模块来处理不平衡，比如随机上采样，随机下采样等。


# In[4]:


# Caching transformers within a Pipeline
# 根据sklearn，此处用的作用是为了减少内存，因为在一个pipe里面可以重复使用某个对象
location = 'cachedir'
memory = Memory(location=location, verbose=10)


# In[5]:


# Make pipeline
pipe = Pipeline([
        ('reduce_dim', PCA()),
        ('feature_selection', SelectKBest(f_classif)),
        ('classify', LogisticRegression(solver='saga', penalty='l1'))
    ], 
    memory=memory
)


# In[6]:


# Set paramters according to users inputs
# PCA参数
max_components = 0.99,
min_components = 0.7
number_pc = 3
range_dimreduction = np.linspace(min_components, max_components, number_pc).reshape(number_pc,)

# ANOVA参数
max_number = 1
min_number = 0.7
number_anova = 3
range_feature_selection = np.linspace(min_number, max_number, number_anova).reshape(number_anova,)
# 由于anova检验的特征数必须是整数，所以进行如下的操作，将min/max_number 变为整数
range_feature_selection = np.int16(range_feature_selection * np.shape(X)[1])

# 分类器参数
max_l1_ratio = 1
min_l1_ratio = 0.5
number_l1_ratio = 5
range_l1_ratio = np.linspace(min_l1_ratio, max_l1_ratio, number_l1_ratio).reshape(number_l1_ratio,)

# 整体grid search设置
param_grid = [
    {
        'reduce_dim__n_components': range_dimreduction,
        'feature_selection__k': range_feature_selection,
        'classify__l1_ratio': range_l1_ratio,
    },
]
# P.S. 此处是底层代码逻辑的缩影，其实做起来很简单，例如只需要将PCA换成非负矩阵分解或者其它，只需要将逻辑回归换成SVM或者其它。
# 但是难在我们需要根据不同的目标（分类，回归，聚类）来调试得到稳定的用户能直接使用的代码，并将其封装起来，让用户一键就能使用。


# In[ ]:


# Train
grid = GridSearchCV(pipe, n_jobs=1, param_grid=param_grid)
grid.fit(X_train, y_train)
# 此出我们需要添加模型持久化的方法，即将grid模型保存到本地，方便用户日后使用该模型。
# 另外，如果是线性模型，我们需要将分类的权重保存到本地，方便用户查看或者文章中做报告用
# 用于此处运行后，会产生一个很大的输出，因此我不在此运行结果


# In[8]:


# Delete the temporary cache before exiting
memory.clear(warn=False)
rmtree(location)


# In[9]:


# Prediction
# 此处不同的分类器对象可能会有不同的predict方式，比如逻辑回归要获得decision，需要用model.predict_proba方法
pred = grid.predict(X_test)
dec = grid.predict_proba(X_test)[:,1]


# In[10]:


# Evaluate performances
# 非常欢迎您能贡献您的模型评估代码，比如多分类的评估，回归的评估等。
acc, sens, spec, auc = eval_performance(
    y_test, pred, dec, 
    accuracy_kfold=None, sensitivity_kfold=None, specificity_kfold=None, AUC_kfold=None,
    verbose=1, is_showfig=0
)


# In[ ]:




