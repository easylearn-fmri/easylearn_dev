# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 22:37:32 2018
move selected files to selected folder
Note. Code will create folder for each file with the same name
as those of the source file
#复制单个文件
shutil.copy("C:\\a\\1.txt","C:\\b")
#复制并重命名新文件
shutil.copy("C:\\a\\2.txt","C:\\b\\121.txt")
#复制整个目录(备份)
shutil.copytree("C:\\a","C:\\b\\new_a")

#删除文件
os.unlink("C:\\b\\1.txt")
os.unlink("C:\\b\\121.txt")
#删除空文件夹
try:
    os.rmdir("C:\\b\\new_a")
except Exception as ex:
    print("错误信息："+str(ex))#提示：错误信息，目录不是空的
#删除文件夹及内容
shutil.rmtree("C:\\b\\new_a")

#移动文件
shutil.move("C:\\a\\1.txt","C:\\b")
#移动文件夹
shutil.move("C:\\a\\c","C:\\b")

#重命名文件
shutil.move("C:\\a\\2.txt","C:\\a\\new2.txt")
#重命名文件夹
shutil.move("C:\\a\\d","C:\\a\\new_d")
@author: lenovo
"""

# import
from lc_selectFile_ import selectFile
import shutil
import os

# def


def copyFiles_multi(in_files, out_folder):
    [moveFiles_single(in_file, out_folder) for in_file in in_files]
#


def moveFiles_single(file, folder):
    # find the folder contain the file
    dirname = os.path.dirname(file)
    file_folder = os.path.basename(dirname)
    # create folder to contain file in output folder
    try:
        output_folder = os.mkdir(os.path.join(folder, file_folder))
    except FileExistsError:
        output_folder = os.path.join(folder, file_folder)
        print(
            'folder [{}]\nhave already exist'.format(
                os.path.join(
                    folder,
                    file_folder)))
    # move file
    try:
        shutil.copytree(file, output_folder)
        print('{} copy successfully!'.format(file_folder))
    except BaseException:
        print('{} no need to copy'.format(file_folder))
#


def obtainAllFile(folder):
    files = selectFile(folder)
    return files


def main():
    # input
    out_folder = r'I:\Data_Code\insomnia\workstation_MVPA_2018_05\FunImgARWS'
    in_folder = r'I:\Data_Code\insomnia\workstation_MVPA_2018_05\FunImgARW1'
    # all files
    files = obtainAllFile(in_folder)
    copyFiles_multi(files, out_folder)
