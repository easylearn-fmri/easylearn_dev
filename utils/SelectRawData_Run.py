# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 15:06:51 2018

@author: lenovo
"""
# import modules

from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog
from PyQt5 import QtWidgets
#from PyQt5 import QtWidgets
from SelectRawData_Window import Ui_CopySelectedData
import pandas as pd
import time
#from sel import Ui_Dialog
# ==============define class and initialization===================


class select(QWidget, Ui_CopySelectedData):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.ScaleFolder.clicked.connect(self.scaleFolder)
        self.RawFolder.clicked.connect(self.rawFolder)
        self.SaveFolder.clicked.connect(self.saveFolder)
        self.RunCopy.clicked.connect(self.runCopy)

#     DIY
    def scaleFolder(self):
        self.fileName, filetype = QFileDialog.getOpenFileName(self,
                                                              "选择参考ID(Folder)文件(待选)",
                                                              "D:\myCodes\LC_MVPA\workstation_20180829_dynamicFC",
                                                              "All Files (*);;Text Files (*.txt)")
        self.folder_scale = pd.read_excel(self.fileName)
#        print (self.folder_scale)
        self.folder_scale = self.folder_scale.iloc[:, 0]
        self.ScaleFolder.setText('选择参考ID(Folder)文件(已选)')
        print('你选择的参考ID(Folder)为:\n[{}]'.format(self.fileName))

    def rawFolder(self):
        self.directory_RawFolder = QFileDialog.getExistingDirectory(
            self, "选取原始数据文件夹(待选)", "D:\myCodes\LC_MVPA\workstation_20180829_dynamicFC")

        self.RawFolder.setText('选取原始数据文件夹(已选)')
        print('你选择的参考原始数据文件夹为:\n[{}]'.format(self.directory_RawFolder))

    def saveFolder(self):
        self.directory_SaveFolder = QFileDialog.getExistingDirectory(
            self, "选取结果保存文件夹(待选)", "D:\myCodes\LC_MVPA\workstation_20180829_dynamicFC")
        self.SaveFolder.setText('选取结果保存文件夹(已选)')
        print('你选择的结果保存文件夹为:\n[{}]'.format(self.directory_SaveFolder))

    def runCopy(self):
        import copySelectedDicomFile as copy
        # 获取当前时间
#        Time=time.asctime(time.localtime(time.time()) )
#        Time=Time.split(' ')
#        Time=Time[5]+'_'+Time[1]+'_'+Time[3]+'_'+Time[4]

        sel = copy.copy_fmri(
            subjID_forSelect=self.folder_scale,
            modalityName_forSelect='resting',
            templates={'path': '*\\*'},
            rootPath=self.directory_RawFolder,
            savePath=self.directory_SaveFolder,
            saveFolderName='resting_' + str(time.time()),
            n_processess=10)
        path_subject_all, folder_mri, path_subject_sel, path_modality_all\
            = sel.main_run()
# =====================close window=======================

    def closeEvent(self, event):
        reply = QtWidgets.QMessageBox.question(
            self,
            '复制程序',
            "是否要退出程序？",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

# ===========================================================


def main():
    import sys
    app = QApplication(sys.argv)
    w = select()
    w.show()
    sys.exit(app.exec_())


# ===================executing==========================
if __name__ == '__main__':
    main()
