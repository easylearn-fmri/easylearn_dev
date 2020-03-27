#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Main GUI of the easylearn
"""

# Author: Chao Li <lichao19870617@gmail.com>
# License: MIT


import sys
import os
import time
import json
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog
from PyQt5.QtWidgets import *
from PyQt5 import *
from PyQt5.QtGui import QIcon, QPixmap, QPalette
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QApplication,QWidget,QVBoxLayout,QListView,QMessageBox
# from PyQt5.QtCore import QStringListModel

from easylearn_main_gui import Ui_MainWindow
from easylearn_data_loading_run import EasylearnDataLoadingRun


class EasylearnMainGUI(QMainWindow, Ui_MainWindow):
    """This class is used to display the main GUI of the easylearn.
    """
    def __init__(self):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.working_directory = ""
        self.textBrowser.setText("Hi~, I'm easylearn. I hope I can help you finish this project successfully\n")

        # Set appearance
        self.set_run_appearance()
        logal_style = '''
            #logal{
                background-color: black;
                border: 5px solid white;
                border-radius: 50px;
                border-image: url('./easylearn1.jpg');
                }
                #logal:hover {
                border-radius: 0px;}
        '''
        self.logal.setStyleSheet(logal_style)
        self.setWindowTitle('EASYLEARN')
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("./easylearn.jpg"))
        self.setWindowIcon(icon)

        # Connecting to functions
        self.choose_working_directory.triggered.connect(self.select_workingdir_fun)
        self.create_file.triggered.connect(self.initialize_configuration_fun)
        self.data_loading.clicked.connect(self.data_loading_fun)
        self.feature_engineering.clicked.connect(self.feature_engineering_fun)
        self.machine_learning.clicked.connect(self.machine_learning_fun)
        self.model_evaluation.clicked.connect(self.model_evaluation_fun)
        self.statistical_analysis.clicked.connect(self.statistical_analysis_fun)
        # self.save_workflow.clicked.connect(self.save_workflow_fun)
        self.run.clicked.connect(self.run_fun)
        self.quit.clicked.connect(self.closeEvent_button)

    
    def set_run_appearance(self):
        """Set dart style appearance.
        """
        qss_string_all_pushbutton = """
        QPushButton{color: rgb(200,200,200); border: 2px solid rgb(100,100,100); border-radius:10}
        QPushButton:hover {background-color: black; color: white; font-size:20px; font-weight: bold}
        #MainWindow{background-color: rgb(50, 50, 50)}                               
        """
        qss_string_run_pushbutton = """
        QPushButton{background-color:rgb(100,200,100); color:white; border: 2px solid rgb(100,100,100); border-radius:15}        
        QPushButton:hover {background-color:green; color:white; border: 2px solid rgb(100,100,100); border-radius:15; font-weight: bold}                   
        """
        qss_string_quit_pushbutton = """
        QPushButton{background-color:rgb(200,100,100); color:white; border: 2px solid rgb(100,100,100); border-radius:15}   
        QPushButton:hover {background-color:red; color:white; border: 2px solid rgb(100,100,100); border-radius:15; font-weight: bold}                        
        """
        qss_string_textbrowser = """
        background-color:rgb(200,200,200); color:black; border: 2px solid rgb(100,100,100); border-radius:15; font-size:20px
        """
        self.setStyleSheet(qss_string_all_pushbutton)
        self.run.setStyleSheet(qss_string_run_pushbutton)
        self.quit.setStyleSheet(qss_string_quit_pushbutton)
        self.textBrowser.setStyleSheet(qss_string_textbrowser)

    def set_quite_appearance(self):
        """Set appearance when quit program.

        This make the quit message can be seen clearly.
        """
        qss_string_qmessage = """
        QPushButton:hover {background-color: white; color: black}

        QPushButton{color:white; border: 2px solid rgb(100,100,100); border-radius:5}

        #formLayoutWidget_2{color:white; border: 2px solid rgb(100,100,100); border-radius:9}

        #MainWindow{background-color: rgb(50, 50, 50)}

        QMessageBox{background-color: gray; color: white}                       
        """
        self.setStyleSheet(qss_string_qmessage)

    def select_workingdir_fun(self):
        """
        This function is used to select the working working_directory
        """
        #  If has selected working working_directory previously, then I set it as initial working working_directory.
        if self.working_directory == "":
            self.working_directory = QFileDialog.getExistingDirectory(self, "Select a working_directory", os.getcwd()) 
            self.textBrowser.setText("Current working directory is " + self.working_directory + "\n")
        else:
            self.working_directory = QFileDialog.getExistingDirectory(self, "Select a working_directory", self.working_directory) 
            self.textBrowser.setText("Current working directory is " + self.working_directory + "\n")

    def initialize_configuration_fun(self):
        """Create file to save settings

        This function will add the configuration_file to self
        """
        configuration_file_name, ok = QInputDialog.getText(self, "Initialize configuration", "Please name the configuration file:", QLineEdit.Normal, "configuration_file.json")
        if self.working_directory != "":
            self.configuration_file = os.path.join(self.working_directory, configuration_file_name)
            with open(self.configuration_file, 'w') as configuration_file:
                config={'groups':{'group_1':['mod1','mod2'],'group_2':['mod1','mod2']}}
                config = json.dumps(config)
                configuration_file.write(config)
                self.textBrowser.setText("Configuration file is " + self.configuration_file)
        else:
            self.set_quite_appearance()
            QMessageBox.warning( self, 'Warning', 'Configuration in configuration file is not valid JSON')
            self.set_run_appearance()

    def data_loading_fun(self):
        """This function is called when data_loading button is clicked.

        Then, this function will process the data loading.
        """
        print('data_loading_fun')
        self.data_loading = EasylearnDataLoadingRun()
        self.data_loading.show()

    def feature_engineering_fun(self):
        """This function is called when feature_engineering button is clicked.

        Then, this function will process the feature_engineering.
        """
        print('feature_engineering_fun')

    def machine_learning_fun(self):
        """This function is called when machine_learning button is clicked.

        Then, this function will process the data loading.
        """
        print('machine_learning_fun')

    def model_evaluation_fun(self):
        """This function is called when model_evaluation button is clicked.

        Then, this function will process the model evaluation.
        """
        print('model_evaluation_fun')

    def statistical_analysis_fun(self):
        """This function is called when data_loading button is clicked.

        Then, this function will process the data loading.
        """
        print('statistical_analysis_fun')

    def save_workflow_fun(self):
        """This function is called when data_loading button is clicked.

        Then, this function will process the data loading.
        """
        print('save_workflow_fun')

    def run_fun(self):
        """This function is called when data_loading button is clicked.

        Then, this function will process the data loading.
        """
        print('run_fun')

    def closeEvent(self, event):
        """This function is called when exit icon of the window is clicked.

        This function make sure the program quit safely.
        """
        # Set qss to make sure the QMessageBox can be seen
        self.set_quite_appearance()
        reply = QMessageBox.question(self, 'Quit',"Are you sure to quit?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
            self.set_run_appearance()

    def closeEvent_button(self, event):
        """This function is called when quit button is clicked.

        This function make sure the program quit safely.
        """
        # Set qss to make sure the QMessageBox can be seen
        self.set_quite_appearance()
        reply = QMessageBox.question(self, 'Quit',"Are you sure to quit?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            QCoreApplication.quit()
        else:
            # Make appearance back
            self.set_run_appearance()


if __name__=='__main__':
    app=QApplication(sys.argv)
    md=EasylearnMainGUI()
    md.show()
    sys.exit(app.exec_())
