from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.datasets import make_classification

X, y = make_classification(n_informative=5, n_redundant=0, random_state=42)

anova_filter = SelectKBest(f_regression, k=5)
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)

anova_svm = Pipeline([('anova', anova_filter), ('svc', clf)])


anova_svm.fit(X, y)

prediction = anova_svm.predict(X)
anova_svm.score(X, y)


selected_feature_bool = anova_svm['anova'].get_support()

sub_pipeline = anova_svm[:1]
sub_pipeline



coef = anova_svm[-1].coef_
anova_svm['svc'] is anova_svm[-1]

coef.shape

sub_pipeline.inverse_transform(coef).shape