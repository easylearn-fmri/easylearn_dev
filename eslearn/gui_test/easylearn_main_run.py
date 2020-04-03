#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Main GUI of the easylearn

# Author: Chao Li <lichao19870617@gmail.com>
# License: MIT
"""


import sys
sys.path.append('E:/easylearn/')
import os
import time
import json
from PyQt5.QtWidgets import *
from PyQt5 import *
from PyQt5.QtGui import QIcon, QPixmap, QPalette
from PyQt5.QtCore import *
import qdarkstyle

from easylearn_main_gui import Ui_MainWindow
from easylearn_data_loading_run import EasylearnDataLoadingRun
from easylearn_logger import easylearn_logger


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

        # Connecting to functions
        self.select_working_directory.triggered.connect(self.select_workingdir_fun)
        self.create_configuration_file.triggered.connect(self.initialize_configuration_fun)
        self.choose_configuration_file.triggered.connect(self.load_configuration_fun)
        self.data_loading.clicked.connect(self.data_loading_fun)
        self.feature_engineering.clicked.connect(self.feature_engineering_fun)
        self.machine_learning.clicked.connect(self.machine_learning_fun)
        self.model_evaluation.clicked.connect(self.model_evaluation_fun)
        self.statistical_analysis.clicked.connect(self.statistical_analysis_fun)
        self.run.clicked.connect(self.run_fun)
        self.quit.clicked.connect(self.closeEvent_button)

        # Skins
        self.skins = ["Dark", "Black", "DarkOrange", "Gray", "Blue", "Navy", "Classic", "Light"]
        self.actionDark.triggered.connect(self.set_run_appearance)
        self.actionBlack.triggered.connect(self.set_run_appearance)
        self.actionDarkOrange.triggered.connect(self.set_run_appearance)
        self.actionGray.triggered.connect(self.set_run_appearance)
        self.actionBlue.triggered.connect(self.set_run_appearance)
        self.actionNavy.triggered.connect(self.set_run_appearance)
        self.actionClassic.triggered.connect(self.set_run_appearance)
        self.actionLight.triggered.connect(self.set_run_appearance)

    
    def set_run_appearance(self):
        """Set a appearance for easylearn (title, logo, skin, etc).
        """
        qss_logo = """#logo{background-color: black;
                border: 2px solid white;
                border-radius: 20px;
                border-image: url('../logo/logo-dms.png');
                }
                #logo:hover {border-radius: 0px;}
        """

        qss_special = """QPushButton:hover
        {
            font-weight: bold; font-size: 20px;
        }     

        QPushButton#run
        {
             border-radius:30px; border: 1px dashed green;
        }     

        QPushButton#run:hover 
        {
            font-weight: bold; border-radius:20px; border: 2px solid green;
        }  

        QPushButton#quit
        { 
            border-radius:30px; border: 1px dashed red;
        }  

        QPushButton#quit:hover 
        {
            font-weight: bold; border-radius:20px; border: 2px solid red;
        }  
        """
        self.logo.setStyleSheet(qss_logo)
        self.setWindowTitle('easylearn')

        # icon = QtGui.QIcon()
        # icon.addPixmap(QtGui.QPixmap("../logo/logo-dms-small.png"))
        self.setWindowIcon(QIcon('../logo/logo-dms1.png'))
        # self.setWindowIcon(icon)

        sender = self.sender()
        if sender:
            if (sender.text() in self.skins):
                with open("../stylesheets/" + sender.text() + ".qss") as f:
                    style_sheets = f.read()
                    style_sheets = style_sheets + qss_special
                    self.setStyleSheet(style_sheets)
            else:
                with open("../stylesheets/Dark.qss") as f:
                    style_sheets = f.read()
                    style_sheets = style_sheets + qss_special
                    self.setStyleSheet(style_sheets)
        else:
            with open("../stylesheets/Dark.qss") as f:
                style_sheets = f.read()
                style_sheets = style_sheets + qss_special
                self.setStyleSheet(style_sheets)

        # Run Icon
        self.run.setIcon(QIcon("../logo/run.png"));
        self.run.setIconSize(QPixmap("../logo/run.png").size());
        self.run.resize(QPixmap("../logo/run.png").size());
        # Close Icon
        self.quit.setIcon(QIcon("../logo/close.png"));
        self.quit.setIconSize(QPixmap("../logo/close.png").size());
        self.quit.resize(QPixmap("../logo/close.png").size());

    def select_workingdir_fun(self):
        """
        This function is used to select the working working_directory, then change directory to this directory.
        """
        #  If has selected working working_directory previously, then I set it as initial working working_directory.
        if self.working_directory == "":
            self.working_directory = QFileDialog.getExistingDirectory(self, "Select a working_directory", os.getcwd()) 
            self.textBrowser.setText("Current working directory is " + self.working_directory + "\n")
        else:
            self.working_directory = QFileDialog.getExistingDirectory(self, "Select a working_directory", self.working_directory) 
            self.textBrowser.setText("Current working directory is " + self.working_directory + "\n")

        # If already choose a working directory, change directory to the working directory
        if self.working_directory != "":
            os.chdir(self.working_directory)

    def initialize_configuration_fun(self):
        """Create file to save settings

        This function will add the configuration_file to self
        """
        configuration_file_name, ok = QInputDialog.getText(self, "Initialize configuration", "Please name the configuration file:", QLineEdit.Normal, "configuration_file.json")
        if self.working_directory != "":
            self.configuration_file = os.path.join(self.working_directory, configuration_file_name)
            with open(self.configuration_file, 'w') as configuration_file:
                config = {"data_loading": {}, "features_engineering": {}, "machine_learning": {}, "model_evaluation": {}, "statistical_analysis": {}}
                config = json.dumps(config)
                configuration_file.write(config)
                config_message = "Configuration file is " + self.configuration_file
                self.textBrowser.setText(config_message)
        else:
            QMessageBox.warning( self, 'Warning', f'Please choose a working directory first! (press button at the top left corner)')

    def load_configuration_fun(self):
        """Load configuration
        """
        self.configuration_file, filetype = QFileDialog.getOpenFileName(self,  
                                "Select configuration file",  
                                os.getcwd(), "Text Files (*.json);;All Files (*);;") 

        # Read configuration_file if already selected
        if self.configuration_file != "": 
        # TODO: 解决中文编码的问题 
            with open(self.configuration_file, 'r') as config:
                self.configuration = config.read()
            # Check the configuration is valid JSON, then transform the configuration to dict
            # If the configuration is not valid JSON, then give configuration and configuration_file to ""
            try:
                self.configuration = json.loads(self.configuration)
                self.textBrowser.setText("Configuration file is " + self.configuration_file)
            except json.decoder.JSONDecodeError:
    
                QMessageBox.warning( self, 'Warning', f'{self.configuration_file} is not valid JSON')
                self.configuration_file = ""
        else:

            QMessageBox.warning( self, 'Warning', 'Configuration file was not selected')

    def data_loading_fun(self):
        """This function is called when data_loading button is clicked.

        Then, this function will process the data loading.
        """
        print('data_loading_fun')
        self.data_loading = EasylearnDataLoadingRun(self.working_directory)
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
        reply = QMessageBox.question(self, 'Quit',"Are you sure to quit?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def closeEvent_button(self, event):
        """This function is called when quit button is clicked.

        This function make sure the program quit safely.
        """
        # Set qss to make sure the QMessageBox can be seen
        reply = QMessageBox.question(self, 'Quit',"Are you sure to quit?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            QCoreApplication.quit()


if __name__=='__main__':
    app=QApplication(sys.argv)
    md=EasylearnMainGUI()
    md.show()
    sys.exit(app.exec_())
