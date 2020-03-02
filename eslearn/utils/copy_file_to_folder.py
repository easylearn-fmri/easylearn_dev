# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 20:46:52 2019
将一个文件夹中的ROI文件复制到相应的被试文件夹下面
@author: lenovo
"""
import os
import shutil
import pandas as pd
"""input"""
file_folder_path = r"D:\dms-lymph-nodes\merge_noContrast"
out_folder_path = r"D:\dms-lymph-nodes\finish"


def read_files_name():
    all_files_name = os.listdir(file_folder_path)
    all_files_path = [
        os.path.join(
            file_folder_path,
            file_name) for file_name in all_files_name]
    return all_files_name, all_files_path


def extract_uid(all_files_name):
    uid = [file_name.split('_')[0] for file_name in all_files_name]
    return uid


def copy_file_to_folder(all_files_name, all_files_path, uid):
    """读取被试文件名以及路径"""
    all_folders_name = os.listdir(out_folder_path)
    all_folders_path = [
        os.path.join(
            out_folder_path,
            folder_name) for folder_name in all_folders_name]
    all_folders_path = pd.Series(all_folders_path)
    """copy..."""
    print("copying...")
    for i, one_uid in enumerate(uid):
        print("正在复制第{}/{}个文件".format(i + 1, len(uid)))
        all_folders_name = pd.Series(all_folders_name)
        my_id = all_folders_name.str.contains(one_uid)
        out_path = all_folders_path[my_id]
        in_file = all_files_path[i]
        out_file = os.path.join(out_path.iloc[0], all_files_name[i])
        shutil.copyfile(in_file, out_file)
    else:
        print("finished!\n")


def delete():
    """读取被试文件名以及路径"""
    all_folders_name = os.listdir(out_folder_path)
    all_folders_path = [
        os.path.join(
            out_folder_path,
            folder_name) for folder_name in all_folders_name]
    all_folders_path = pd.Series(all_folders_path)
    print("deleting...")
    for i in range(len(all_folders_path)):
        print("正在删除第{}/{}个文件".format(i, len(all_folders_path)))
        all_files_of_one_subj = pd.Series(os.listdir(all_folders_path.iloc[i]))
        ind = all_files_of_one_subj.str.contains(".nii")
        remove_file = all_files_of_one_subj[ind]
        try:
            remove_file_path = os.path.join(
                all_folders_path[i], remove_file.iloc[0])  # 可能有多个匹配文件
            os.remove(remove_file_path)
        except IndexError:
            print("{}文件夹下没有需要删除的文件".format(all_folders_path[i]))


def main(if_copy=0, if_del=0):

    if if_copy:
        all_files_name, all_files_path = read_files_name()
        uid = extract_uid(all_files_name)
        copy_file_to_folder(all_files_name, all_files_path, uid)

    if if_del:
        delete()


if __name__ == "__main__":
    main(0, 1)
