import pandas as pd
import numpy as np

from eslearn.machine_learning.classification.classification import Classification
from eslearn.machine_learning.classification._base_classification import StatisticalAnalysis

def test_classification():
    clf = Classification(configuration_file=r"D:\work\workstation_b\林赛湘雅\T1 test1.json", out_dir=r"D:\work\workstation_b\林赛湘雅")
    clf.main_run()
    clf.run_statistical_analysis()

if __name__ == "__main__":
    test_classification()