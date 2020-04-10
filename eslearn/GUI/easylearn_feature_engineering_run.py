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

        # intial
        
        # connect items signal to slot
        self.pushButton_preprocessing.clicked.connect(self.on_pushButton_preprocessing_clicked)
        self.pushButton_dimreduction.clicked.connect(self.on_pushButton_dimreduction_clicked)
        self.pushButton_selection.clicked.connect(self.on_pushButton_selection_clicked)
        self.pushButton_unbalance_treatment.clicked.connect(self.on_pushButton_unbalance_treatment_clicked)
        # connect preprocessing setting signal to slot
        self.radioButton_scaling.clicked.connect(self.on_radioButton_scaling_clicked)
        self.radioButton_zscore.clicked.connect(self.on_radioButton_not_scaling_clicked)
        self.radioButton_demean.clicked.connect(self.on_radioButton_not_scaling_clicked)
        # connect dimreduction setting signal to slot
        self.dimreduction_radioButton_dict = {"PCA": 0, "ICA": 1, "LDA": 2, "LLE": 3}
        self.radioButton_pca.clicked.connect(self.on_radioButton_dimreduction__clicked)
        self.radioButton_ica.clicked.connect(self.on_radioButton_dimreduction__clicked)
        self.radioButton_lda.clicked.connect(self.on_radioButton_dimreduction__clicked)
        self.radioButton_lle.clicked.connect(self.on_radioButton_dimreduction__clicked)

        # set appearance
        self.set_run_appearance()

    def set_run_appearance(self):
        """Set style_sheets
        """
        qss_special = """QPushButton:hover
        {
            font-weight: bold; font-size: 15px;
        } 

        """
        self.setWindowTitle('Data Loading')
        self.setWindowIcon(QIcon('../logo/logo-upper.jpg'))

        sender = self.sender()
        if sender:
            if (sender.text() in list(self.skins.keys())):
                self.setStyleSheet(PyQt5_stylesheets.load_stylesheet_pyqt5(style=self.skins[sender.text()]))
            else:
                self.setStyleSheet(PyQt5_stylesheets.load_stylesheet_pyqt5(style="style_Dark"))
        else:
            self.setStyleSheet(PyQt5_stylesheets.load_stylesheet_pyqt5(style="style_Dark"))

        # Make the stackedWidg to default at the begining
        self.stackedWidget_items.setCurrentIndex(0)
        self.on_radioButton_not_scaling_clicked()

    def on_pushButton_preprocessing_clicked(self):
        print("preprocessing")
        self.stackedWidget_items.setCurrentIndex(0)


    def on_pushButton_dimreduction_clicked(self):
        print("dimreduction")
        self.stackedWidget_items.setCurrentIndex(1)


    def on_pushButton_selection_clicked(self):
        print("selection")
        self.stackedWidget_items.setCurrentIndex(3)

    def on_pushButton_unbalance_treatment_clicked(self):
        print("deal with unbalance")
        self.stackedWidget_items.setCurrentIndex(4)

    def on_radioButton_scaling_clicked(self):
        self.stackedWidget_preprocessing_methods.setCurrentIndex(0)
        print("scaling")

    def on_radioButton_not_scaling_clicked(self):
        self.stackedWidget_preprocessing_methods.setCurrentIndex(1)

    #%% radioButtons of dimreduction
    def on_radioButton_dimreduction__clicked(self):
        self.stackedWidget_dimreduction.setCurrentIndex(self.dimreduction_radioButton_dict[self.sender().text()])


if __name__ == "__main__":
    app=QApplication(sys.argv)
    md=EasylearnFeatureEngineeringRun()
    md.show()
    sys.exit(app.exec_())
