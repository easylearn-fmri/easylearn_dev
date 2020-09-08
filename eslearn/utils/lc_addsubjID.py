# -*- coding: utf-8 -*-
"""
Created on Sat May 18 15:50:17 2019

@author: lenovo
"""
import os


class Input():
    """
    process input
    """

    def __init__(sel):
#        cwd = os.path.dirname(__file__)
        cwd = os.getcwd()
        input_path = os.path.join(cwd, "input.txt")
        if not os.path.exists(input_path):
            input("No input text file!")

        with open(input_path, 'r', encoding='UTF-8') as f:
            cont = f.readlines()
        allinput = [cont_.split('=')[1] for cont_ in cont]
        allinput = [allin.split('\n')[0] for allin in allinput]

        sel.root_path = eval(allinput[0])
        sel.modality = eval(allinput[1])
        sel.metric = eval(allinput[2])


class AddSubjID(Input):
    """
    add subject ID to files in BIDS
    """
    def __init__(sel, root_path=None, modality=None, metric=None):
        super().__init__()
        if not sel.root_path:
            print("please set root path")
        print("AddsubjID initiated!")

    def addsubjid(sel):
        """
        main function
        """
        sel.__get_all_subj_folder()
        sel.__get_all_metric_path()
        sel.__get_all_files_path()
        sel.__addid()
        input("All Done!\nPress any key to exit...")
        
    def __get_all_subj_folder(sel):
        sel.all_subj_folder_name = os.listdir(sel.root_path)

    def __get_all_metric_path(sel):
        sel.all_metric_path = \
        [os.path.join(sel.root_path, folder, sel.modality, sel.metric) for folder in sel.all_subj_folder_name]
        return sel.all_metric_path
    
    def __get_all_files_path(sel):
        all_files_name = [os.listdir(filepath) for filepath in sel.all_metric_path]
        sel.all_files_path = []
        for metric, file in zip(sel.all_metric_path, all_files_name):
            filepath = [os.path.join(metric, onefile) for onefile in file]
            sel.all_files_path.append(filepath)
        return sel.all_files_path

    def __addid(sel):
        n_subj = len(sel.all_subj_folder_name)
        i = 1
        for folder_name, file_path in zip(sel.all_subj_folder_name, sel.all_files_path):
            print("processing {}/{}".format(i, n_subj))
            old_name = file_path
            old_basename = [os.path.basename(oldname) for oldname in old_name]
            old_dirname = [os.path.dirname(oldname) for oldname in old_name]
            new_name = [folder_name + "_" + basename for basename in old_basename]
            new_name = [os.path.join(olddirname, newname) for olddirname, newname in zip(old_dirname, new_name)]
            # execute!
            [os.rename(oldname, newname) for oldname, newname in zip(old_name, new_name)]
            i += 1


if __name__ == "__main__":
    addid = AddSubjID(root_path=r'F:\黎超\陆衡鹏飞\2d_sample')
    addid.addsubjid()
