import pandas as pd
import numpy as np
import pytest

from eslearn.machine_learning.classification.classification import Classification

def test_classification():
    clf = Classification(configuration_file=r'D:\My_Codes\virtualenv_eslearn\Lib\site-packages\eslearn\GUI\tests\szVShc.json', 
        out_dir=r"D:\My_Codes\virtualenv_eslearn\Lib\site-packages\eslearn\GUI\tests") 
    clf.main_run()
    clf.run_statistical_analysis()