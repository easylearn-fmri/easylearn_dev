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
import numpy as np
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
        self.all_available_inputs()

        # Debug
        # Set working_directory
        self.working_directory = working_directory
        if self.working_directory:
            cgitb.enable(format="text", display=1, logdir=os.path.join(self.working_directory, "log_machine_learning"))
        else:
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
        self.actionGet_all_available_configuration.triggered.connect(self._get_all_available_inputs)

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
        self.radioButton_classification_lr.clicked.connect(self.switche_stacked_wedge_for_classification)
        self.radioButton_classification_svm.clicked.connect(self.switche_stacked_wedge_for_classification)
        self.radioButton_classification_ridge.clicked.connect(self.switche_stacked_wedge_for_classification)
        self.radioButton_classification_gaussianprocess.clicked.connect(self.switche_stacked_wedge_for_classification)
        self.radioButton_classification_randomforest.clicked.connect(self.switche_stacked_wedge_for_classification)
        self.radioButton_classification_adaboost.clicked.connect(self.switche_stacked_wedge_for_classification)
        
        # connect regression setting signal to slot: switche to corresponding regression method
        self.regression_stackedwedge_dict = {
            "Linear regression": 0, "Lasso regression": 1, "Ridge regression": 2,
            "Elastic-Net regression": 3, "Support vector machine": 4, "Gaussian process": 5,
            "Random forest": 6
        }
        self.radioButton_regression_linearregression.clicked.connect(self.switche_stacked_wedge_for_regression)
        self.radioButton_regression_lasso.clicked.connect(self.switche_stacked_wedge_for_regression)
        self.radioButton_regression_ridge.clicked.connect(self.switche_stacked_wedge_for_regression)
        self.radioButton_regression_elasticnet.clicked.connect(self.switche_stacked_wedge_for_regression)
        self.radioButton_regression_svm.clicked.connect(self.switche_stacked_wedge_for_regression)
        self.radioButton_regression_gaussianprocess.clicked.connect(self.switche_stacked_wedge_for_regression)

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

    def all_available_inputs(self):
        """I put all available inputs in a dictionary named all_available_inputs
        """

        # This dictionary is used to keep track of machine learning types
        self.machine_learning_type_dict = {
            "Classification": self.radioButton_classification, "Regression": self.radioButton_regression,
            "Clustering":self.radioButton_clustering, "Deep learning": self.radioButton_deeplearning,
        }

        # All available inputs
        self.all_available_inputs = {
            "Classification": {
                self.radioButton_classification_lr:{
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
                self.radioButton_regression_lasso: {
                    "Lasso regression":{
                        "minalpha": {"value": self.doubleSpinBox_regression_lasso_minalpha.text(), "wedget": self.doubleSpinBox_regression_lasso_minalpha}, 
                        "maxalpha": {"value": self.doubleSpinBox_regression_lasso_maxalpha.text(), "wedget": self.doubleSpinBox_regression_lasso_maxalpha},
                        "numalpha": {"value": self.spinBox_regression_lasso_numalpha.text(), "wedget": self.spinBox_regression_lasso_numalpha},
                    }

                },
                self.radioButton_regression_ridge: {
                    "Ridge regression":{
                        "minalpha": {"value": self.doubleSpinBox_regression_ridge_minalpha.text(), "wedget": self.doubleSpinBox_regression_ridge_minalpha}, 
                        "maxalpha": {"value": self.doubleSpinBox_regression_ridge_maxalpha.text(), "wedget": self.doubleSpinBox_regression_ridge_maxalpha},
                        "numalpha": {"value": self.spinBox_regression_ridge_numalpha.text(), "wedget": self.spinBox_regression_ridge_numalpha},
                    }

                }
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

    def _get_all_available_inputs(self):
        """ This method used to get all available inputs for users
        
        Delete wedgets object from all available inputs dict
        NOTE: This code is only for current configuration structure
        """

        all_available_inputs_for_user_tmp = self.all_available_inputs
        for machine_learning_name in list(all_available_inputs_for_user_tmp.keys()):
            for method in list(all_available_inputs_for_user_tmp[machine_learning_name].keys()):
                for method_name in list(all_available_inputs_for_user_tmp[machine_learning_name][method].keys()):
                    for setting in list(all_available_inputs_for_user_tmp[machine_learning_name][method][method_name].keys()):
                        if "wedget" in list(all_available_inputs_for_user_tmp[machine_learning_name][method][method_name][setting].keys()):
                            all_available_inputs_for_user_tmp[machine_learning_name][method][method_name][setting].pop("wedget")
        
        all_available_inputs_for_user = {}
        for machine_learning_name in list(all_available_inputs_for_user_tmp.keys()):
            all_available_inputs_for_user[machine_learning_name] = {}
            for method in list(all_available_inputs_for_user_tmp[machine_learning_name].keys()):
                all_available_inputs_for_user[machine_learning_name].update(all_available_inputs_for_user_tmp[machine_learning_name][method])
        del all_available_inputs_for_user_tmp

        # Save to folder that contains configuration file
        if self.configuration_file != "":
            outname = os.path.join(os.path.dirname(self.configuration_file), 'all_available_machine_learning_inputs.json')
            with open(outname, 'w', encoding="utf-8") as config:    
                config.write(json.dumps(all_available_inputs_for_user, indent=4))
        else:
            QMessageBox.warning( self, 'Warning', "configuration file is not selected!")

    def get_current_inputs(self):
        """Get all current inputs

        Programme will scan the GUI to determine the user's inputs.

        Attrs:
        -----
            self.machine_learning: dictionary
                all machine_learning parameters that the user input.
        """

        # Get current inputs
        for machine_learning_type in self.all_available_inputs:
            for method in self.all_available_inputs[machine_learning_type]:
                # Only both machine_learning_type and method are clicked, I save configuration to self.machine_learning dictionary 
                if self.machine_learning_type_dict[machine_learning_type].isChecked() and method.isChecked():
                    self.machine_learning[machine_learning_type] = self.all_available_inputs[machine_learning_type][method]

    def load_configuration(self):
        """Load configuration, and refresh_gui configuration in GUI
        """

        # Get current inputs before load configuration, so we can 
        # compare loaded configuration["machine_learning"] with the current self.machine_learning
        self.get_current_inputs()

        if not self.working_directory:
            self.configuration_file, filetype = QFileDialog.getOpenFileName(
                self,  
                "Select configuration file",  
                os.getcwd(), "Text Files (*.json);;All Files (*);;"
            ) 
        else:
            self.configuration_file, filetype = QFileDialog.getOpenFileName(
                self,  
                "Select configuration file",  
                self.working_directory, "Text Files (*.json);;All Files (*);;"
            ) 

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
        
        # If already identified the configuration file, then excude saving logic.      
        if self.configuration_file != "":
            try:
                # self.configuration = json.dumps(self.configuration, ensure_ascii=False)
                self.configuration["machine_learning"] = self.machine_learning
                # self.configuration = json.dumps(self.configuration, indent=4)
                with open(self.configuration_file, 'w', encoding="utf-8") as config:   
                    config.write( json.dumps(self.configuration, ensure_ascii=False, indent=4) )
            except json.decoder.JSONDecodeError:
                QMessageBox.warning( self, 'Warning', f'{self.configuration}'+ ' is not a valid JSON!')

        else:
            QMessageBox.warning( self, 'Warning', 'Please choose a configuration file first (press button at top left corner)!')


    #%% Update GUI: including refresh_gui and switche_stacked_wedge_for_*
    def refresh_gui(self):
        """ Refresh gui the display the loaded configuration in the GUI
        """

        # Generate a dict for switch stacked wedgets
        method_switch_dict = {
            "Classification": self.switche_stacked_wedge_for_classification,
            "Regression": self.switche_stacked_wedge_for_regression,
            "Clustering": self.switche_stacked_wedge_for_clustering,
            "Deep learning": self.switche_stacked_wedge_for_deep_learning,
        }

        for machine_learning_type in list(self.all_available_inputs.keys()):
            for method in self.all_available_inputs[machine_learning_type].keys():  
                if machine_learning_type in self.machine_learning.keys():
                    # Click the input machine learning type wedget
                    self.machine_learning_type_dict[machine_learning_type].setChecked(True)
                    
                    if method.text() in list(self.machine_learning[machine_learning_type].keys()):
                        # Click the input method wedget
                        method.setChecked(True) 
                        
                        # Click the input setting wedget
                        for key_setting in self.machine_learning[machine_learning_type][method.text()]:
                            if "wedget" in list(self.all_available_inputs[machine_learning_type][method][method.text()][key_setting].keys()):
                                loaded_text = self.machine_learning[machine_learning_type][method.text()][key_setting]["value"]
                                # Identity wedget type, then using different methods to "setText"
                                # NOTE. 所有控件在设计时，尽量保留原控件的名字在命名的前部分，这样下面才好确定时哪一种类型的控件，从而用不同的赋值方式！
                                if "lineEdit" in self.all_available_inputs[machine_learning_type][method][method.text()][key_setting]["wedget"].objectName():
                                    self.all_available_inputs[machine_learning_type][method][method.text()][key_setting]["wedget"].setText(loaded_text)
                                elif "doubleSpinBox" in self.all_available_inputs[machine_learning_type][method][method.text()][key_setting]["wedget"].objectName():
                                    self.all_available_inputs[machine_learning_type][method][method.text()][key_setting]["wedget"].setValue(float(loaded_text))
                                elif "spinBox" in self.all_available_inputs[machine_learning_type][method][method.text()][key_setting]["wedget"].objectName():
                                    self.all_available_inputs[machine_learning_type][method][method.text()][key_setting]["wedget"].setValue(int(loaded_text))
                                elif "comboBox" in self.all_available_inputs[machine_learning_type][method][method.text()][key_setting]["wedget"].objectName():
                                    self.all_available_inputs[machine_learning_type][method][method.text()][key_setting]["wedget"].setCurrentText(loaded_text)

                                # Switch stacked wedget
                                self.switche_stacked_wedge_for_machine_learning_type(True, machine_learning_type)
                                method_switch_dict[machine_learning_type](True, method.text())


    def switche_stacked_wedge_for_machine_learning_type(self, signal_bool, machine_learning_type=None):
        """ Switch to corresponding machine learning type window
        """

        if self.sender():
            if not machine_learning_type:
                self.stackedWidget_machine_learning_type.setCurrentIndex(self.machine_learning_type_stackedwedge_dict[self.sender().text()])
            else:
                self.stackedWidget_machine_learning_type.setCurrentIndex(self.machine_learning_type_stackedwedge_dict[machine_learning_type])
        else:
            self.stackedWidget_machine_learning_type.setCurrentIndex(-1)

    def switche_stacked_wedge_for_classification(self, signal_bool, method=None):
        """ Switch to corresponding classification model window
        """

        self.radioButton_classification.setChecked(True)

        if self.sender():
            if not method:
                self.stackedWidget_classification_setting.setCurrentIndex(self.classification_stackedwedge_dict[self.sender().text()])
            else:
                self.stackedWidget_classification_setting.setCurrentIndex(self.classification_stackedwedge_dict[method])
        else:
            self.stackedWidget_classification_setting.setCurrentIndex(-1)

    def switche_stacked_wedge_for_regression(self, signal_bool, method=None):
        """ Switch to corresponding regression model window
        """
        self.radioButton_regression.setChecked(True)

        if self.sender():
            if not method:
                self.stackedWidget_regression.setCurrentIndex(self.regression_stackedwedge_dict[self.sender().text()])
            else:
                self.stackedWidget_regression.setCurrentIndex(self.regression_stackedwedge_dict[method])
        else:
            self.stackedWidget_regression.setCurrentIndex(-1)

    def switche_stacked_wedge_for_clustering(self, signal_bool, method=None):
        """ Switch to corresponding clustering model window
        """

        self.radioButton_clustering.setChecked(True)

        if self.sender():
            if not method:
                self.stackedWidget_classification_setting.setCurrentIndex(self.clustering_stackedwedge_dict[self.sender().text()])
            else:
                self.stackedWidget_classification_setting.setCurrentIndex(self.clustering_stackedwedge_dict[method])
        else:
            self.stackedWidget_classification_setting.setCurrentIndex(-1)

    def switche_stacked_wedge_for_deep_learning(self, signal_bool, method=None):
        """ Switch to corresponding deep learning model window
        """
        self.radioButton_deeplearning.setChecked(True)

        if self.sender():
            if not method:
                self.stackedWidget_classification_setting.setCurrentIndex(self.deep_learning_stackedwedge_dict[self.sender().text()])
            else:
                self.stackedWidget_classification_setting.setCurrentIndex(self.deep_learning_stackedwedge_dict[method])
        else:
            self.stackedWidget_classification_setting.setCurrentIndex(-1)

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
