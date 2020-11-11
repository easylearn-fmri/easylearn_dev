# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'SelectRawData_Window.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_CopySelectedData(object):
    def setupUi(self, CopySelectedData):
        CopySelectedData.setObjectName("CopySelectedData")
        CopySelectedData.resize(293, 331)
        self.ScaleFolder = QtWidgets.QPushButton(CopySelectedData)
        self.ScaleFolder.setGeometry(QtCore.QRect(30, 20, 221, 41))
        self.ScaleFolder.setObjectName("ScaleFolder")
        self.RawFolder = QtWidgets.QPushButton(CopySelectedData)
        self.RawFolder.setGeometry(QtCore.QRect(30, 110, 221, 41))
        self.RawFolder.setObjectName("RawFolder")
        self.SaveFolder = QtWidgets.QPushButton(CopySelectedData)
        self.SaveFolder.setGeometry(QtCore.QRect(30, 200, 221, 41))
        self.SaveFolder.setObjectName("SaveFolder")
        self.RunCopy = QtWidgets.QPushButton(CopySelectedData)
        self.RunCopy.setGeometry(QtCore.QRect(152, 270, 101, 41))
        self.RunCopy.setObjectName("RunCopy")

        self.retranslateUi(CopySelectedData)
        QtCore.QMetaObject.connectSlotsByName(CopySelectedData)

    def retranslateUi(self, CopySelectedData):
        _translate = QtCore.QCoreApplication.translate
        CopySelectedData.setWindowTitle(
            _translate("CopySelectedData", "Dialog"))
        self.ScaleFolder.setText(_translate("CopySelectedData", "选择参考Folder"))
        self.RawFolder.setText(_translate("CopySelectedData", "选择原始数据"))
        self.SaveFolder.setText(_translate("CopySelectedData", "选择保存路径"))
        self.RunCopy.setText(_translate("CopySelectedData", "Copy"))
