import pandas as pd
import numpy as np
import pytest

from eslearn.machine_learning.classification.classification import Classification

def test_classification():
    clf = Classification(configuration_file="./clf_configuration.json", out_dir="./") 
    clf.main_run()
    clf.run_statistical_analysis()


if __name__ == "__main__":
    test_classification()