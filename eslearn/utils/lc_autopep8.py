# -*- coding: utf-8 -*-
"""
Created on Tue May  7 21:14:51 2019

@author: lenovo
"""

import subprocess
import os
import numpy as np

def my_autopep8(py_name):
    cmd = "autopep8 --in-place --aggressive --aggressive" + " " + py_name
    print('converting {}...'.format(py_name))
    state = subprocess.call(cmd, shell=True)

    if not state:
        print("Succeed!")
    else:
        print("Failed!")


def my_autopep8_folder(folder):
    """
    autopep8 for all .py file in the folder
    """
    folder = r'F:\黎超\dynamicFC\Code\lc_rsfmri_tools_python\Utils'
    file_name = os.listdir(folder)
    py_name = [filename for filename in file_name if '.py' in filename]

    all_cmd = ["autopep8 --in-place --aggressive --aggressive" +
               " " + pyname for pyname in py_name]

    num_py = np.arange(1, len(py_name) + 1)
    len_py = len(py_name)
    for i, cmd, pyname in zip(num_py, all_cmd, py_name):
        print('converting {} ({}/{})...'.format(pyname, i, len_py))
        state = subprocess.call(cmd, shell=True)

        if not state:
            print("Succeed!\n")
        else:
            print("Failed!\n")
    else:
        print("Done!")

if __name__ == '__main__':
    #    my_autopep8('lc_delect_sensitive_info.py')
    my_autopep8_folder(r'F:\黎超\dynamicFC\Code\lc_rsfmri_tools_python\Utils')
