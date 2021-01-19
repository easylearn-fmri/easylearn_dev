#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
Base class for all modules
"""

from eslearn.base import BaseMachineLearning, DataLoader


def test_dataLoading():
    data_loader = DataLoader(configuration_file='./dataLoadingTest.json')
    data_loader.load_data()


if __name__ == '__main__':
    test_dataLoading()