#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
Base class for all modules
"""

from eslearn.base import BaseMachineLearning, DataLoader


def test_base():
    base = BaseMachineLearning(configuration_file='./configuration_file.json')
    data_loader = DataLoader(configuration_file='./configuration_file.json')
    data_loader.load_data()
    
    base.get_all_inputs()
    
    print(base.method_feature_preprocessing_)
    print(base.param_feature_preprocessing_)
    print(base.method_dim_reduction_)
    print(base.param_dim_reduction_)
    print(base.method_feature_selection_)
    print(base.param_feature_selection_)
    print(base.method_unbalance_treatment_)
    print(base.param_unbalance_treatment_)
    print(base.method_machine_learning_)
    print(base.param_machine_learning_)
    print(base.method_model_evaluation_)
    print(base.param_model_evaluation_)
    print(base.method_statistical_analysis_)
    print(base.param_statistical_analysis_)

    assert str(base.method_feature_preprocessing_) == "[StandardScaler()]"
    assert str(base.param_feature_preprocessing_) == "None"
    assert str(base.method_dim_reduction_) == "[PCA()]"
    assert str(base.param_dim_reduction_) == "{'dim_reduction__n_components': [0.8, 0.9], 'dim_reduction__random_state': [0]}"
    assert str(base.method_feature_selection_) == "[RFECV(estimator=SVC(kernel='linear'))]"
    assert str(base.param_feature_selection_) == "{'feature_selection__step': [0.1], 'feature_selection__cv': [5], 'feature_selection__estimator': [SVC(kernel='linear')], 'feature_selection__n_jobs': [-1]}"
    assert str(base.method_unbalance_treatment_) == "RandomOverSampler(random_state=0)"
    assert str(base.param_unbalance_treatment_) == "{'unbalance_treatment__random_state': [0]}"
    assert str(base.method_machine_learning_) == "[LinearSVC()]"
    assert str(base.param_machine_learning_) == "{'estimator__C': [1], 'estimator__multi_class': ['ovr'], 'estimator__random_state': [0]}"
    assert str(base.method_model_evaluation_) == "StratifiedKFold(n_splits=2, random_state=0, shuffle=True)"
    assert str(base.param_model_evaluation_) == "{'n_splits': 2, 'shuffle': 'True', 'random_state': 0}"
    assert str(base.method_statistical_analysis_) == "Permutation test"
    assert str(base.param_statistical_analysis_) == "5"


if __name__ == '__main__':
    test_base()