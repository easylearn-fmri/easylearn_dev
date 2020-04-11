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

        # Debug
        cgitb.enable(display=1, logdir=None)  

        # intial
        
        # connect main items signal to slot: switche to corresponding stackedWidget
        self.items_stackedwedge_dict = {"Preprocessing": 0, "Dimension reduction": 1, "Feature selection": 2, "Unbalance treatment": 3, "None": 4}
        self.pushButton_preprocessing.clicked.connect(self.on_pushButton_items_clicked)
        self.pushButton_dimreduction.clicked.connect(self.on_pushButton_items_clicked)
        self.pushButton_selection.clicked.connect(self.on_pushButton_items_clicked)
        self.pushButton_unbalance_treatment.clicked.connect(self.on_pushButton_items_clicked)
        # connect preprocessing setting signal to slot: switche to corresponding stackedWidget
        self.preprocessing_stackedwedge_dict = {"Z-score normalization": 0, "Scaling": 1, "De-mean": 2, "None": 3}
        self.radioButton_scaling.clicked.connect(self.on_preprocessing_detail_stackedwedge_clicked)
        self.radioButton_zscore.clicked.connect(self.on_preprocessing_detail_stackedwedge_clicked)
        self.radioButton_demean.clicked.connect(self.on_preprocessing_detail_stackedwedge_clicked)
        self.radioButton_none_methods.clicked.connect(self.on_preprocessing_detail_stackedwedge_clicked)
        # connect dimreduction setting signal to slot: switche to corresponding stackedWidget
        self.dimreduction_stackedwedge_dict = {"Principal component analysis": 0, "Independent component analysis": 1, "Latent Dirichlet Allocation": 2, " Non-negative matrix factorization": 3, "None": 4}
        self.radioButton_pca.clicked.connect(self.on_dimreduction_stackedwedge_clicked)
        self.radioButton_ica.clicked.connect(self.on_dimreduction_stackedwedge_clicked)
        self.radioButton_lda.clicked.connect(self.on_dimreduction_stackedwedge_clicked)
        self.radioButton_nmf.clicked.connect(self.on_dimreduction_stackedwedge_clicked)
        self.radioButton_none.clicked.connect(self.on_dimreduction_stackedwedge_clicked)
        # connect feature selection setting signal to slot: switche to corresponding stackedWidget
        self.feature_selection_stackedwedge_dict = {"Variance threshold": 0, "Correlation": 1, "Distance correlation": 2, "F-Score": 3, 
        "Mutual information (classification)": 4, "Mutual information (regression)": 5, "ReliefF": 6, "ANOVA": 7, 
        "RFE": 8, 
        "L1 regularization (Lasso)": 9, "L2 regularization (Ridge regression)": 10, "L1 + L2 regularization (Elastic net regression)": 11}
        self.radioButton_variance_threshold.clicked.connect(self.on_feature_selection_stackedwedge_clicked)
        self.radioButton_correlation.clicked.connect(self.on_feature_selection_stackedwedge_clicked)
        self.radioButton_distancecorrelation.clicked.connect(self.on_feature_selection_stackedwedge_clicked)
        self.radioButton_fscore.clicked.connect(self.on_feature_selection_stackedwedge_clicked)
        self.radioButton_mutualinfo_cls.clicked.connect(self.on_feature_selection_stackedwedge_clicked)
        self.radioButton_mutualinfo_regression.clicked.connect(self.on_feature_selection_stackedwedge_clicked)
        self.radioButton_relieff.clicked.connect(self.on_feature_selection_stackedwedge_clicked)
        self.radioButton_anova.clicked.connect(self.on_feature_selection_stackedwedge_clicked)
        self.radioButton_rfe.clicked.connect(self.on_feature_selection_stackedwedge_clicked)
        self.radioButton_l1.clicked.connect(self.on_feature_selection_stackedwedge_clicked)
        self.radioButton_l2.clicked.connect(self.on_feature_selection_stackedwedge_clicked)
        self.radioButton_elasticnet.clicked.connect(self.on_feature_selection_stackedwedge_clicked)

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
        self.setWindowTitle('Feature Engineering')
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
        self.stackedWidget_items.setCurrentIndex(4)
        self.stackedWidget_preprocessing_methods.setCurrentIndex(-1)

    def on_pushButton_items_clicked(self):
        print(self.sender().text())
        self.stackedWidget_items.setCurrentIndex(self.items_stackedwedge_dict[self.sender().text()])
 
    def on_preprocessing_detail_stackedwedge_clicked(self):
        print(self.sender().text())
        if self.sender().text():
            self.stackedWidget_preprocessing_methods.setCurrentIndex(self.preprocessing_stackedwedge_dict[self.sender().text()])
        else:
            self.stackedWidget_preprocessing_methods.setCurrentIndex(-1)

    def on_radioButton_not_scaling_clicked(self):
        self.stackedWidget_preprocessing_methods.setCurrentIndex(1)

    #%% radioButtons of dimreduction
    def on_dimreduction_stackedwedge_clicked(self):
        self.stackedWidget_dimreduction.setCurrentIndex(self.dimreduction_stackedwedge_dict[self.sender().text()])

    def on_feature_selection_stackedwedge_clicked(self):
        self.groupBox_feature_selection_input.setTitle(self.sender().text())
        self.stackedWidget_feature_selection.setCurrentIndex(self.feature_selection_stackedwedge_dict[self.sender().text()])

    def closeEvent(self, event):
        """This function is called when exit icon of the window is clicked.

        This function make sure the program quit safely.
        """
        # Set qss to make sure the QMessageBox can be seen
        reply = QMessageBox.question(self, 'Quit',"Are you sure to quit?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore() 


if __name__ == "__main__":
    app=QApplication(sys.argv)
    md=EasylearnFeatureEngineeringRun()
    md.show()
    sys.exit(app.exec_())
