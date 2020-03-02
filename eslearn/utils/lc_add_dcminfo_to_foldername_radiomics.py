# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 22:30:29 2019
读取dicom文件的信息,并在文件夹名字中添加dicom信息【ID_name_oldname】
@author: LI Chao
"""

import os
import pydicom


class ChangeFolderName():
    """change subjects' folder name"""

    def __init__(self):
        self.path = r'J:\lymph nodes\356_patients_data'

    def fetch_all_subj_folder_path(self):
        """读取所有被试文件夹的路径"""
        all_subj_folder_path = os.listdir(self.path)
        self.all_subj_folder_path = [
            os.path.join(
                self.path,
                path) for path in all_subj_folder_path]
        print("### 读取了所有被试文件夹路径 ###\n")
        return self

    def fetch_all_subj_dicom_info(self):
        """
        Get all subject's dicom information
        """
        all_sample_dicom = [os.listdir(path)[0]
                            for path in self.all_subj_folder_path]
        all_sample_dicom = [
            os.path.join(
                path, dicom) for path, dicom in zip(
                self.all_subj_folder_path, all_sample_dicom)]

        # read dicom info
        print("### Reading dicom info... ###\n")
        self.dicom_info = [
            pydicom.read_file(
                sample_dicom,
                force=True) for sample_dicom in all_sample_dicom]
        self.patient_ID = [info.PatientID for info in self.dicom_info]
        self.patient_name = [
            info.PatientName.components for info in self.dicom_info]
        print("### dicom info read completed! ###\n")
        return self

    def re_name(self):
        """修改文件夹名字"""
        my_count = 1
        num_subj = len(self.patient_ID)
        for ID, name, folder_path in zip(
                self.patient_ID, self.patient_name, self.all_subj_folder_path):
            print("changing the subject: {}/{}\n".format(my_count, num_subj))
            my_count = my_count + 1

            old_name = folder_path
            new_name = "".join(
                [ID, "_", name[0], "_", os.path.basename(folder_path)])
            new_name = os.path.join(self.path, new_name)
            # 如果old_name 中有ID则不重命名
            if ID in old_name:
                print("{}中以及存在ID号，不需修改".format(old_name))
                continue
            else:
                os.rename(old_name, new_name)
            print("### 修改了所有被试文件夹名字 ###\n")


if __name__ == "__main__":
    read = ChangeFolderName()
    results = read.fetch_all_subj_folder_path()
    results = read.fetch_all_subj_dicom_info()
    read.re_name()
