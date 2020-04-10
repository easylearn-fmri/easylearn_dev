# -*- coding: utf-8 -*-
"""The GUI of the feature_engineering module of easylearn

Created on Wed Jul  4 13:57:15 2018
@author: Li Chao
Email:lichao19870617@gmail.com
GitHub account name: lichao312214129
Institution (company): Brain Function Research Section, The First Affiliated Hospital of China Medical University, Shenyang, Liaoning, PR China. 

License: MIT
"""


import sys
sys.path.append('../stylesheets/PyQt5_stylesheets')
import os
import json
import cgitb
from PyQt5.QtWidgets import QApplication,QMainWindow, QFileDialog
from PyQt5.QtWidgets import *
from PyQt5 import *
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication,QWidget,QVBoxLayout,QListView,QMessageBox
from PyQt5.QtCore import*
import PyQt5_stylesheets

from easylearn_feature_engineering_gui import Ui_MainWindow


from PyQt5 import QtCore, QtGui, QtWidgets

class EasylearnFeatureEngineeringRun(QMainWindow, Ui_MainWindow):
    def __init__(self, working_directory=None):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        # connect
        self.pushButton.clicked.connect(self.on_pushButton1_clicked)
        self.pushButton_2.clicked.connect(self.on_pushButton2_clicked)
        self.pushButton_3.clicked.connect(self.on_pushButton3_clicked)
        self.pushButton_4.clicked.connect(self.on_pushButton1_clicked)


    def on_pushButton1_clicked(self):
        self.stackedWidget.setCurrentIndex(0)


    def on_pushButton2_clicked(self):
        self.stackedWidget.setCurrentIndex(1)


    def on_pushButton3_clicked(self):
        self.stackedWidget.setCurrentIndex(2)


if __name__ == "__main__":
    app=QApplication(sys.argv)
    md=EasylearnFeatureEngineeringRun()
    md.show()
    sys.exit(app.exec_())
