# -*- coding: utf-8 -*-
"""The GUI of the machine_learning module of easylearn

Created on 2020/04/15

@author: Li Chao <lichao19870617@gmail.com; lichao312214129>
Institution (company): Brain Function Research Section, The First Affiliated Hospital of China Medical University, Shenyang, Liaoning, PR China. 

@author: Dong Mengshi <dongmengshi1990@163.com;  dongmengshi>
GitHub account name: dongmengshstitution (company): Department of radiology, The First Affiliated Hospital of China Medical University, Shenyang, Liaoning, PR China. 
License: MIT
"""


import sys
import os
import json
import cgitb
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from eslearn.stylesheets.PyQt5_stylesheets import PyQt5_stylesheets

from easylearn_machine_learning_gui import Ui_MainWindow


class EasylearnMachineLearningRun(QMainWindow, Ui_MainWindow):
    """The GUI of the machine_learning module of easylearn

    All users' input will save to configuration_file for finally run the whole machine learning pipeline.
    Specificity, the self.machine_learning configuration will save to the configuration_file that the user created in 
    the main window.
    """

    def __init__(self, working_directory=None):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        # Initialization
        self.machine_learning = {}
        self.configuration_file = ""

        # Debug
        cgitb.enable(display=1, logdir=None)  

        # Skins
        self.skins = {"Dark": "style_Dark", "Black": "style_black", "DarkOrange": "style_DarkOrange", 
                    "Gray": "style_gray", "Blue": "style_blue", "Navy": "style_navy", "Classic": "style_Classic"}
        self.actionDark.triggered.connect(self.set_run_appearance)
        self.actionBlack.triggered.connect(self.set_run_appearance)
        self.actionDarkOrange.triggered.connect(self.set_run_appearance)
        self.actionGray.triggered.connect(self.set_run_appearance)
        self.actionBlue.triggered.connect(self.set_run_appearance)
        self.actionNavy.triggered.connect(self.set_run_appearance)
        self.actionClassic.triggered.connect(self.set_run_appearance)

        # Connect configuration functions
        self.actionLoad_configuration.triggered.connect(self.load_configuration)
        self.actionSave_configuration.triggered.connect(self.save_configuration)

        # connect to radioButton of machine learning type: switche to corresponding machine learning type window
        self.machine_learning_type_stackedwedge_dict = {
            "Classification": 0, "Regression": 1, "Clustering": 2, "Deep learning": 3,
        }
        self.radioButton_classification.clicked.connect(self.switche_stacked_wedge_for_machine_learning_type)
        self.radioButton_regression.clicked.connect(self.switche_stacked_wedge_for_machine_learning_type)
        self.radioButton_clustering.clicked.connect(self.switche_stacked_wedge_for_machine_learning_type)
        self.radioButton_deeplearning.clicked.connect(self.switche_stacked_wedge_for_machine_learning_type)

        # connect classification setting signal to slot: switche to corresponding classification model
        self.classification_stackedwedge_dict = {
            "Logistic regression": 0, "Support vector machine": 1, "Ridge classification": 2,
            "Gaussian process": 3, "Random forest": 4, "AdaBoost": 5
        }
        self.radioButton_classificaton_lr.clicked.connect(self.switche_stacked_wedge_for_classification)
        self.radioButton_classification_svm.clicked.connect(self.switche_stacked_wedge_for_classification)
        self.radioButton_classification_ridge.clicked.connect(self.switche_stacked_wedge_for_classification)
        self.radioButton_classification_gaussianprocess.clicked.connect(self.switche_stacked_wedge_for_classification)
        self.radioButton_classification_randomforest.clicked.connect(self.switche_stacked_wedge_for_classification)
        self.radioButton_classification_adaboost.clicked.connect(self.switche_stacked_wedge_for_classification)
        
        # Set appearance
        self.set_run_appearance()

    def set_run_appearance(self):
        """Set style_sheets
        """
        qss_special = """QPushButton:hover
        {
            font-weight: bold; font-size: 15px;
        } 

        """
        self.setWindowTitle('Machine learning')
        self.setWindowIcon(QIcon('../logo/logo-upper.jpg'))

        sender = self.sender()
        if sender:
            if (sender.text() in list(self.skins.keys())):
                self.setStyleSheet(PyQt5_stylesheets.load_stylesheet_pyqt5(style=self.skins[sender.text()]))
                if sender.text() == "Classic":
                    self.setStyleSheet("")
            else:
                self.setStyleSheet(PyQt5_stylesheets.load_stylesheet_pyqt5(style="style_Dark"))
        else:
            self.setStyleSheet(PyQt5_stylesheets.load_stylesheet_pyqt5(style="style_Dark"))

        # Make the stackedWidg to default at the begining

    def get_current_inputs(self):
        """Get all current inputs

        Programme will scan the GUI to determine the user's inputs.

        Attrs:
        -----
            self.machine_learning: dictionary
                all machine_learning parameters that the user input.
        """

        # I put all available inputs in a dictionary named all_available_inputs
        self.all_available_inputs = {
            "Classification": {
                self.radioButton_classificaton_lr:{
                    "Logistic regression": {
                        "maxl1ratio": {"value": self.doubleSpinBox_clf_lr_maxl1ratio.text(), "wedget": self.doubleSpinBox_clf_lr_maxl1ratio},
                        "minl1ratio": {"value": self.doubleSpinBox_clf_lr_maxl1ratio.text(), "wedget": self.doubleSpinBox_clf_lr_minl1ratio}, 
                        "numberl1ratio": {"value": self.spinBox__clf_lr_numl1ratio.text(), "wedget": self.spinBox__clf_lr_numl1ratio},
                    },
                }, 

                self.radioButton_classification_svm:{
                    "Support vector machine": {
                        "kernel": {"value": self.comboBox_clf_svm_kernel.currentText(), "wedget": self.comboBox_clf_svm_kernel},
                        "minc": {"value": self.doubleSpinBox_clf_svm_minc.text(), "wedget": self.doubleSpinBox_clf_svm_minc}, 
                        "maxc": {"value": self.doubleSpinBox_clf_svm_maxc.text(), "wedget": self.doubleSpinBox_clf_svm_maxc},
                        "numc": {"value": self.spinBox_clf_svm_numc.text(), "wedget": self.spinBox_clf_svm_numc},
                        "maxgamma": {"value": self.lineEdit_clf_svm_maxgamma.text(), "wedget": self.lineEdit_clf_svm_maxgamma},
                        "mingamma": {"value": self.lineEdit_clf_svm_mingamma.text(), "wedget": self.lineEdit_clf_svm_mingamma},
                        "numgamma": {"value": self.spinBox_clf_svm_numgamma.text(), "wedget": self.spinBox_clf_svm_numgamma},
                    },
                },

                self.radioButton_classification_ridge:{
                    "Ridge classification": {
                        "minalpha": {"value": self.doubleSpinBox_clf_ridgeclf_minalpha.text(), "wedget": self.doubleSpinBox_clf_ridgeclf_minalpha}, 
                        "maxalpha": {"value": self.doubleSpinBox_clf_ridgeclf_maxalpha.text(), "wedget": self.doubleSpinBox_clf_ridgeclf_maxalpha},
                        "numalpha": {"value": self.spinBox_clf_ridgeclf_numalpha.text(), "wedget": self.spinBox_clf_ridgeclf_numalpha},
                    },
                },

                self.radioButton_classification_gaussianprocess:{
                    "Gaussian process": {},
                },

                self.radioButton_classification_randomforest:{
                    "Random forest": {
                        "minestimators": {"value": self.spinBox_clf_randomforest_minestimators.text(), "wedget": self.spinBox_clf_randomforest_minestimators}, 
                        "maxestimators": {"value": self.spinBox_clf_randomforest_maxestimators.text(), "wedget": self.spinBox_clf_randomforest_maxestimators},
                        "maxdepth": {"value": self.spinBox_clf_randomforest_maxdepth.text(), "wedget": self.spinBox_clf_randomforest_maxdepth},
                    },
                },

                self.radioButton_classification_adaboost:{
                    "AdaBoost": {
                        "minestimators": {"value": self.spinBox_clf_adaboost_minestimators.text(), "wedget": self.spinBox_clf_adaboost_minestimators}, 
                        "maxestimators": {"value": self.spinBox_clf_adaboost_maxestimators.text(), "wedget": self.spinBox_clf_adaboost_maxestimators},
                    },
                },
            },

            "Regression": {
                self.radioButton_regression_linearregression:{
                    "Linear regression": {
                        
                    },
                }, 
            },

            "Clustering": {
                self.radioButton_clustering_kmeans:{
                    "K-means clustering": {
                        
                    },
                }, 
            },


            "Deep learning": {
                self.radioButton_regression_linearregression:{
                    "Linear regression": {
                        
                    },
                }, 
            },

        }

        # Get current inputs
        for key_machine_learning in self.all_available_inputs:
            for mltype in self.all_available_inputs[key_machine_learning]:
                if mltype.isChecked():
                    self.machine_learning[key_machine_learning] = self.all_available_inputs[key_machine_learning][mltype]

    def load_configuration(self):
        """Load configuration, and refresh_gui configuration in GUI
        """

        # Get current inputs before load configuration, so we can 
        # compare loaded configuration["machine_learning"] with the current self.machine_learning
        self.get_current_inputs()

        self.configuration_file, filetype = QFileDialog.getOpenFileName(self,  
                                "Select configuration file",  
                                os.getcwd(), "Text Files (*.json);;All Files (*);;") 

        # Read configuration_file if already selected
        if self.configuration_file != "": 
            with open(self.configuration_file, 'r', encoding='utf-8') as config:
                self.configuration = config.read()
            # Check the configuration is valid JSON, then transform the configuration to dict
            # If the configuration is not valid JSON, then give configuration and configuration_file to ""
            try:
                self.configuration = json.loads(self.configuration)
                # If already exists self.machine_learning
                if (self.machine_learning != {}):
                    # If the loaded self.configuration["machine_learning"] is not empty
                    # Then ask if rewrite self.machine_learning with self.configuration["machine_learning"]
                    if (list(self.configuration["machine_learning"].keys()) != []):
                        reply = QMessageBox.question(
                            self, "Data loading configuration already exists", 
                            "The machine_learning configuration is already exists, do you want to rewrite it with the  loaded configuration?",
                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                        )

                        if reply == QMessageBox.Yes:  
                            self.machine_learning = self.configuration["machine_learning"]
                            self.refresh_gui()
                    # If the loaded self.configuration["machine_learning"] is empty
                     # Then assign self.configuration["machine_learning"] with self.machine_learning
                    else:
                        self.configuration["machine_learning"] = self.machine_learning
                else:
                    self.machine_learning = self.configuration["machine_learning"]
                    self.refresh_gui()
            except json.decoder.JSONDecodeError:
                QMessageBox.warning( self, 'Warning', f'{self.configuration_file} is not valid JSON')
                self.configuration_file = ""
   
        else:
            QMessageBox.warning( self, 'Warning', 'Configuration file was not selected')

    def save_configuration(self):
        """Save configuration that users inputed.
        """

        # Get current inputs before saving
        self.get_current_inputs()
    
        # Delete wedgets object from self.machine_learning dict
        # NOTE: This code is only for current configuration structure
        for machine_learning_name in list(self.machine_learning.keys()):
            for method_name in list(self.machine_learning[machine_learning_name].keys()):
                for setting in list(self.machine_learning[machine_learning_name][method_name].keys()):
                    for content in list(self.machine_learning[machine_learning_name][method_name][setting].keys()):
                        if "wedget" in list(self.machine_learning[machine_learning_name][method_name][setting].keys()):
                            self.machine_learning[machine_learning_name][method_name][setting].pop("wedget")

        # TODO: only keep one machine learning type
        print(self.machine_learning)
        
        # If already identified the configuration file, then excude saving logic.      
        if self.configuration_file != "":
            try:
                # self.configuration = json.dumps(self.configuration, ensure_ascii=False)
                self.configuration["machine_learning"] = self.machine_learning
                self.configuration = json.dumps(self.configuration)
                with open(self.configuration_file, 'w', encoding="utf-8") as config:    
                    config.write(self.configuration)
            except json.decoder.JSONDecodeError:
                QMessageBox.warning( self, 'Warning', f'{self.configuration}'+ ' is not a valid JSON!')

        else:
            QMessageBox.warning( self, 'Warning', 'Please choose a configuration file first (press button at top left corner)!')

    #%% Update GUI: including refresh_gui and switche_stacked_wedge_for_*
    def refresh_gui(self):
        """ Refresh gui the display the loaded configuration in the GUI
        """

        # Generate a dict for switch stacked wedgets
        switch_dict = {
            "Classification": self.switche_stacked_wedge_for_classification,
            "Regression": self.switche_stacked_wedge_for_regression,
            "Clustering": self.switche_stacked_wedge_for_clustering,
            "Deep learning": self.switche_stacked_wedge_for_deep_learning,
        }

        for mltype in self.all_available_inputs:  # each machine learning type
            for wedget in self.all_available_inputs[mltype].keys():  # each model wedget of one machine learning type
                for method in self.all_available_inputs[mltype][wedget].keys():
                    # print(method)
                    if mltype in self.machine_learning.keys():
                        # self.switche_stacked_wedge_for_machine_learning_type(True, mltype)
                        # print(mltype)
                        if method in list(self.machine_learning[mltype].keys()):
                            # Make the wedget checked according loaded param
                            wedget.setChecked(True)   
                            # Make setting to loaded text
                            for key_setting in self.machine_learning[mltype][method]:
                                if "wedget" in list(self.all_available_inputs[mltype][wedget][method][key_setting].keys()):
                                    loaded_text = self.machine_learning[mltype][method][key_setting]["value"]
                                    # Identity wedget type, then using different methods to "setText"
                                    # NOTE. 所有控件在设计时，尽量保留原控件的名字在命名的前部分，这样下面才好确定时哪一种类型的控件，从而用不同的赋值方式！
                                    if "lineEdit" in self.all_available_inputs[mltype][wedget][method][key_setting]["wedget"].objectName():
                                        self.all_available_inputs[mltype][wedget][method][key_setting]["wedget"].setText(loaded_text)
                                    elif "doubleSpinBox" in self.all_available_inputs[mltype][wedget][method][key_setting]["wedget"].objectName():
                                        self.all_available_inputs[mltype][wedget][method][key_setting]["wedget"].setValue(float(loaded_text))
                                    elif "spinBox" in self.all_available_inputs[mltype][wedget][method][key_setting]["wedget"].objectName():
                                        self.all_available_inputs[mltype][wedget][method][key_setting]["wedget"].setValue(int(loaded_text))
                                    elif "comboBox" in self.all_available_inputs[mltype][wedget][method][key_setting]["wedget"].objectName():
                                        self.all_available_inputs[mltype][wedget][method][key_setting]["wedget"].setCurrentText(loaded_text)

                                    # Switch stacked wedget
                                    switch_dict[mltype](True, method)


    def switche_stacked_wedge_for_machine_learning_type(self, signal_bool, mltype=None):
        """ Switch to corresponding machine learning type window
        """

        if self.sender():
            if not mltype:
                self.stackedWidget_type.setCurrentIndex(self.machine_learning_type_stackedwedge_dict[self.sender().text()])
            else:
                self.stackedWidget_type.setCurrentIndex(self.machine_learning_type_stackedwedge_dict[mltype])
        else:
            self.stackedWidget_type.setCurrentIndex(-1)

    def switche_stacked_wedge_for_classification(self, signal_bool, method=None):
        """ Switch to corresponding classification model window
        """

        self.radioButton_classification.setChecked(True)

        if self.sender():
            if not method:
                self.stackedWidget_setting.setCurrentIndex(self.classification_stackedwedge_dict[self.sender().text()])
            else:
                self.stackedWidget_setting.setCurrentIndex(self.classification_stackedwedge_dict[method])
        else:
            self.stackedWidget_setting.setCurrentIndex(-1)

    def switche_stacked_wedge_for_regression(self, signal_bool, method=None):
        """ Switch to corresponding regression model window
        """
        self.radioButton_regression.setChecked(True)

        if self.sender():
            if not method:
                self.stackedWidget_setting.setCurrentIndex(self.regression_stackedwedge_dict[self.sender().text()])
            else:
                self.stackedWidget_setting.setCurrentIndex(self.regression_stackedwedge_dict[method])
        else:
            self.stackedWidget_setting.setCurrentIndex(-1)

    def switche_stacked_wedge_for_clustering(self, signal_bool, method=None):
        """ Switch to corresponding clustering model window
        """

        self.radioButton_clustering.setChecked(True)

        if self.sender():
            if not method:
                self.stackedWidget_setting.setCurrentIndex(self.clustering_stackedwedge_dict[self.sender().text()])
            else:
                self.stackedWidget_setting.setCurrentIndex(self.clustering_stackedwedge_dict[method])
        else:
            self.stackedWidget_setting.setCurrentIndex(-1)

    def switche_stacked_wedge_for_deep_learning(self, signal_bool, method=None):
        """ Switch to corresponding deep learning model window
        """
        self.radioButton_deeplearning.setChecked(True)

        if self.sender():
            if not method:
                self.stackedWidget_setting.setCurrentIndex(self.deep_learning_stackedwedge_dict[self.sender().text()])
            else:
                self.stackedWidget_setting.setCurrentIndex(self.deep_learning_stackedwedge_dict[method])
        else:
            self.stackedWidget_setting.setCurrentIndex(-1)

    # def closeEvent(self, event):
    #     """This function is called when exit icon of the window is clicked.

    #     This function make sure the program quit safely.
    #     """
    #     # Set qss to make sure the QMessageBox can be seen
    #     reply = QMessageBox.question(self, 'Quit',"Are you sure to quit?",
    #                                  QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

    #     if reply == QMessageBox.Yes:
    #         event.accept()
    #     else:
    #         event.ignore() 


if __name__ == "__main__":
    app=QApplication(sys.argv)
    md=EasylearnMachineLearningRun()
    md.show()
    sys.exit(app.exec_())
