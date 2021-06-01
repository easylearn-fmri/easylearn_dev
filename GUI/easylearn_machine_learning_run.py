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
from eslearn.stylesheets.PyQt5_stylesheets import pyqt5_loader

import eslearn
from eslearn.GUI.easylearn_machine_learning_gui import Ui_MainWindow
from eslearn.machine_learning.neural_network.eeg.run import EEGClassifier


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
        self.root_dir = os.path.dirname(eslearn.__file__)

        # Initialization
        self.machine_learning = {}
        self.working_directory = working_directory
        self.configuration_file = ""
        self.configuration = {}
        self.all_inputs_fun()

        # Debug
        # Set working_directory
        self.working_directory = working_directory
        if self.working_directory:
            cgitb.enable(format="text", display=1, logdir=os.path.join(self.working_directory, "log_machine_learning"))
        else:
            cgitb.enable(display=1, logdir=None) 

        # Connect configuration functions
        self.actionLoad_configuration.triggered.connect(self.load_configuration)
        self.actionSave_configuration.triggered.connect(self.save_configuration)
        self.actionGet_all_available_configuration.triggered.connect(self._get_all_available_inputs)

        # Connect to radioButton of machine learning type: switche to corresponding machine learning type window
        self.machine_learning_type_stackedwedge_dict = {
            "Classification": 0, "Regression": 1, "Clustering": 2, "Deep learning": 3,
        }
        self.radioButton_classification.clicked.connect(self.switche_stacked_wedge_for_machine_learning_type)
        self.radioButton_regression.clicked.connect(self.switche_stacked_wedge_for_machine_learning_type)
        self.radioButton_clustering.clicked.connect(self.switche_stacked_wedge_for_machine_learning_type)
        self.radioButton_deeplearning.clicked.connect(self.switche_stacked_wedge_for_machine_learning_type)

        # Connect classification setting signal to slot: switche to corresponding classification model
        self.classification_stackedwedge_dict = {
            "LogisticRegression(solver='saga')": 0, "LinearSVC()":1, "SVC()": 2, "RidgeClassifier()": 3,
            "GaussianProcessClassifier()": 4, "RandomForestClassifier()": 5, "AdaBoostClassifier()": 6
        }
        self.radioButton_classification_lr.clicked.connect(self.switche_stacked_wedge_for_classification)
        self.radioButton_classification_linearsvc.clicked.connect(self.switche_stacked_wedge_for_classification)
        self.radioButton_classification_svm.clicked.connect(self.switche_stacked_wedge_for_classification)
        self.radioButton_classification_ridge.clicked.connect(self.switche_stacked_wedge_for_classification)
        self.radioButton_classification_gaussianprocess.clicked.connect(self.switche_stacked_wedge_for_classification)
        self.radioButton_classification_randomforest.clicked.connect(self.switche_stacked_wedge_for_classification)
        self.radioButton_classification_adaboost.clicked.connect(self.switche_stacked_wedge_for_classification)
        
        # Connect regression setting signal to slot: switche to corresponding regression method
        self.regression_stackedwedge_dict = {
            "LinearRegression()": 0, "LassoCV()": 1, "RidgeCV()": 2,
            "ElasticNetCV()": 3, "SVR()": 4, "GaussianProcessRegressor()": 5,
            "RandomForestRegressor()": 6
        }
        self.radioButton_regression_linearregression.clicked.connect(self.switche_stacked_wedge_for_regression)
        self.radioButton_regression_lasso.clicked.connect(self.switche_stacked_wedge_for_regression)
        self.radioButton_regression_ridge.clicked.connect(self.switche_stacked_wedge_for_regression)
        self.radioButton_regression_elasticnet.clicked.connect(self.switche_stacked_wedge_for_regression)
        self.radioButton_regression_svm.clicked.connect(self.switche_stacked_wedge_for_regression)
        self.radioButton_regression_gaussianprocess.clicked.connect(self.switche_stacked_wedge_for_regression)
        self.radioButton_regression_randomforest.clicked.connect(self.switche_stacked_wedge_for_regression)

        # Connect clustering setting signal to slot: switche to corresponding clustering method
        self.clustering_stackedwedge_dict = {
            "KMeans()": 0, "SpectralClustering()": 1, 
            "AgglomerativeClustering()": 2,"DBSCAN()": 3
        }
        self.radioButton_clustering_kmeans.clicked.connect(self.switche_stacked_wedge_for_clustering)
        self.radioButton_spectral_clustering.clicked.connect(self.switche_stacked_wedge_for_clustering)
        self.radioButton_hierarchical_clustering.clicked.connect(self.switche_stacked_wedge_for_clustering)
        self.radioButton_DBSCAN.clicked.connect(self.switche_stacked_wedge_for_clustering)

        # Connect clustering setting signal to slot: switche to corresponding deep learning method
        self.deep_learning_stackedwedge_dict = {
            "EEGClassifier": 0, "CNN": 1, "GCN": 2,"RNN": 3
        }
        self.radioButton_EEGClassifier.clicked.connect(self.switche_stacked_wedge_for_deep_learning)
        self.radioButton_CNN.clicked.connect(self.switche_stacked_wedge_for_deep_learning)
        self.radioButton_GCN.clicked.connect(self.switche_stacked_wedge_for_deep_learning)
        self.radioButton_RNN.clicked.connect(self.switche_stacked_wedge_for_deep_learning)

        self.pushButton_eegclf_prepare_data.clicked.connect(self.eegclf_prepare_data)
        self.eegclf_train.clicked.connect(self.eegclf_train_fun)
        self.pushButton_eegclf_eval.clicked.connect(self.eegclf_eval_fun)
        self.pushButton_eegclf_save.clicked.connect(self.eegclf_save_fun)
        self.eegclf_test.clicked.connect(self.eegclf_test_fun)
        self.eegclf_train_with_pretrained_model.clicked.connect(self.eegclf_train_with_pretrained_model_fun)


        # Skin
        self.skins = {"Dark": "style_Dark", "Black": "style_black", "DarkOrange": "style_DarkOrange", 
                    "Gray": "style_gray", "Blue": "style_blue", "Navy": "style_navy", "Classic": "style_Classic"}
        self.actionDark.triggered.connect(self.change_skin)
        self.actionBlack.triggered.connect(self.change_skin)
        self.actionDarkOrange.triggered.connect(self.change_skin)
        self.actionGray.triggered.connect(self.change_skin)
        self.actionBlue.triggered.connect(self.change_skin)
        self.actionNavy.triggered.connect(self.change_skin)
        self.actionClassic.triggered.connect(self.change_skin)

        # Set appearance
        self.set_run_appearance()

        # Set initial skin
        self.setStyleSheet(pyqt5_loader.load_stylesheet_pyqt5(style="style_Dark"))

    def set_run_appearance(self):
        """Set style_sheets
        """

        winsep = "\\"
        linuxsep = "/"
        root_dir = os.path.dirname(eslearn.__file__)
        root_dir = root_dir.replace(winsep, linuxsep)
        logo_upper = os.path.join(root_dir, "logo/logo-upper.ico")

        qss_special = """QPushButton:hover
        {
            font-weight: bold; font-size: 15px;
        } 

        """
        self.setWindowTitle('Machine learning')
        self.setWindowIcon(QIcon(logo_upper))

    def change_skin(self):
        """Set skins"""

        sender = self.sender()
        if sender:
            if (sender.text() in list(self.skins.keys())):
                self.setStyleSheet(pyqt5_loader.load_stylesheet_pyqt5(style=self.skins[sender.text()]))
            else:
                self.setStyleSheet(pyqt5_loader.load_stylesheet_pyqt5(style="style_Dark"))
        else:
            self.setStyleSheet(pyqt5_loader.load_stylesheet_pyqt5(style="style_Dark"))

    def all_inputs_fun(self):
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
                    "LogisticRegression(solver='saga')": {
                        "penalty": {"value": self.comboBox_clf_lr_penalty.currentText(), "wedget": self.comboBox_clf_lr_penalty},
                        "l1_ratio": {"value": self.lineEdit_clf_lr_l1ratio.text(), "wedget": self.lineEdit_clf_lr_l1ratio},
                        "C": {"value": self.lineEdit_clf_lr_C.text(), "wedget": self.lineEdit_clf_lr_C},
                    },
                }, 

                self.radioButton_classification_linearsvc:{
                    "LinearSVC()": {
                        "C": {"value": self.lineEdit_clf_linearsvc_c.text(), "wedget": self.lineEdit_clf_linearsvc_c},
                        "multi_class": {"value": self.comboBox_clf_linearsvc_multiclass.currentText(), "wedget": self.comboBox_clf_linearsvc_multiclass},                    
                    },
                },

                self.radioButton_classification_svm:{
                    "SVC()": {
                        "kernel": {"value": self.comboBox_clf_svm_kernel.currentText(), "wedget": self.comboBox_clf_svm_kernel},
                        "C": {"value": self.lineEdit_clf_svm_c.text(), "wedget": self.lineEdit_clf_svm_c}, 
                        "gamma": {"value": self.lineEdit_clf_svm_gamma.text(), "wedget": self.lineEdit_clf_svm_gamma},
                    },
                },

                self.radioButton_classification_ridge:{
                    "RidgeClassifier()": {
                        "alpha": {"value": self.lineEdit_clf_ridgeclf_alpha.text(), "wedget": self.lineEdit_clf_ridgeclf_alpha}, 
                    },
                },

                self.radioButton_classification_gaussianprocess:{
                    "GaussianProcessClassifier()": {
                        "kernel": {"value": self.lineEdit_clf_gaussianprocess_kernel.text(), "wedget": self.lineEdit_clf_gaussianprocess_kernel}, 
                        "multi_class": {"value": self.comboBox_clf_gaussianprocess_multiclass.currentText(), "wedget": self.comboBox_clf_gaussianprocess_multiclass}, 
                    },
                },

                self.radioButton_classification_randomforest:{
                    "RandomForestClassifier()": {
                        "criterion": {"value": self.comboBox_clf_randomforest_criterion.currentText(), "wedget": self.comboBox_clf_randomforest_criterion},
                        "max_depth": {"value": self.lineEdit_clf_randomforest_maxdepth.text(), "wedget": self.lineEdit_clf_randomforest_maxdepth},
                        "n_estimators": {"value": self.lineEdit_clf_randomforest_estimators.text(), "wedget": self.lineEdit_clf_randomforest_estimators}, 
                    },
                },

                self.radioButton_classification_adaboost:{
                    "AdaBoostClassifier()": {
                        "n_estimators": {"value": self.lineEdit_clf_adaboost_estimators.text(), "wedget": self.lineEdit_clf_adaboost_estimators}, 
                        "algorithm": {"value": self.comboBox_clf_adaboost_algoritm.currentText(), "wedget": self.comboBox_clf_adaboost_algoritm}, 
                        "base_estimator": {"value": self.comboBox_clf_adaboost_baseesitmator.currentText(), "wedget": self.comboBox_clf_adaboost_baseesitmator},                         
                    },
                },
            },

            "Regression": {
                self.radioButton_regression_linearregression:{
                    "LinearRegression()": {
                        
                    },
                }, 
                self.radioButton_regression_lasso: {
                    "LassoCV()":{
                        "alphas": {"value": self.lineEdit_regression_lasso_alpha.text(), "wedget": self.lineEdit_regression_lasso_alpha}, 
                    }

                },
                self.radioButton_regression_ridge: {
                    "RidgeCV()":{
                    }

                },

                self.radioButton_regression_elasticnet: {
                    "ElasticNetCV()":{
                        "l1_ratio": {"value": self.lineEdit_regression_elasticnet_l1ratio.text(), "wedget": self.lineEdit_regression_elasticnet_l1ratio}, 
                        "alphas": {"value": self.lineEdit_regression_elasticnet_alpha.text(), "wedget": self.lineEdit_regression_elasticnet_alpha}, 
                    }

                },

                self.radioButton_regression_svm: {
                    "SVR()":{
                        "kernel": {"value": self.comboBox_regression_svm_kernel.currentText(), "wedget": self.comboBox_regression_svm_kernel}, 
                        "C": {"value": self.lineEdit_regression_svm_c.text(), "wedget": self.lineEdit_regression_svm_c}, 
                        "gamma": {"value": self.lineEdit_regression_svm_gamma.text(), "wedget": self.lineEdit_regression_svm_gamma},
                    }

                },

                self.radioButton_regression_gaussianprocess: {
                    "GaussianProcessRegressor()":{
                        "kernel": {"value": self.lineEdit_regression_gaussianprocess_kernel.text(), "wedget": self.lineEdit_regression_gaussianprocess_kernel}, 
                        "alpha": {"value": self.lineEdit_regression_gaussianprocess_alpha.text(), "wedget": self.lineEdit_regression_gaussianprocess_alpha}, 
                    }

                },

                self.radioButton_regression_randomforest: {
                    "RandomForestRegressor()":{
                        "criterion": {"value": self.comboBox_regression_randomforest_criterion.currentText(), "wedget": self.comboBox_regression_randomforest_criterion}, 
                        "n_estimators": {"value": self.lineEdit_regression_randomforest_estimators.text(), "wedget": self.lineEdit_regression_randomforest_estimators}, 
                        "max_depth": {"value": self.lineEdit_regression_randomforest_maxdepth.text(), "wedget": self.lineEdit_regression_randomforest_maxdepth}, 
                    }

                },

            },

            "Clustering": {
                self.radioButton_clustering_kmeans:{
                    "KMeans()": {
                        
                    },
                }, 

                self.radioButton_spectral_clustering:{
                    "SpectralClustering()": {
                        
                    },
                },

                self.radioButton_hierarchical_clustering:{
                    "AgglomerativeClustering()": {
                        
                    },
                },

                self.radioButton_DBSCAN:{
                    "DBSCAN()": {
                        
                    },
                },
            },


            "Deep learning": {
                self.radioButton_EEGClassifier:{
                    "EEGClassifier": {
                        "coordinate": {"value": self.lineEdit_eegclf_coordinate.text(), "wedget": self.lineEdit_eegclf_coordinate},
                        "frequency": {"value": self.lineEdit_eegclf_frequency.text(), "wedget": self.lineEdit_eegclf_frequency},
                        "image_size": {"value": self.lineEdit_eegclf_image_size.text(), "wedget": self.lineEdit_eegclf_image_size},
                        "frame_duration": {"value": self.lineEdit_eegclf_frame_duration.text(), "wedget": self.lineEdit_eegclf_frame_duration},
                        "overlap": {"value": self.lineEdit_eegclf_overlap.text(), "wedget": self.lineEdit_eegclf_overlap},
                        "num_classes": {"value": self.lineEdit_eegclf_num_classes.text(), "wedget": self.lineEdit_eegclf_num_classes},
                        "batch_size": {"value": self.lineEdit_eegclf_batch_size.text(), "wedget": self.lineEdit_eegclf_batch_size},
                        "epochs": {"value": self.lineEdit_eegclf_epochs.text(), "wedget": self.lineEdit_eegclf_epochs},
                        "learning_rate": {"value": self.lineEdit_eegclf_learning_rate.text(), "wedget": self.lineEdit_eegclf_learning_rate},
                        "decay": {"value": self.lineEdit_eegclf_decay.text(), "wedget": self.lineEdit_eegclf_decay},
                    },
                },  
            },

        }

    def _get_all_available_inputs(self):
        """ This method used to get all available inputs for users
        
        Delete wedgets object from all available inputs dict
        NOTE: This code is only for current configuration structure
        """

        # Scan the current inputs
        self.all_inputs_fun()

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

        # Scan the current inputs
        self.all_inputs_fun()
        
        # Get current inputs
        for machine_learning_type in self.all_available_inputs:
            for method in self.all_available_inputs[machine_learning_type]:
                # Only both machine_learning_type and method are clicked, I save configuration to self.machine_learning dictionary 
                # due to users may clicked multiple methods across different machine_learning_type
                # given that machine_learning_type is stackedWidgetPage
                if self.machine_learning_type_dict[machine_learning_type].isChecked() and method.isChecked():
                    self.machine_learning = {}
                    self.machine_learning[machine_learning_type] = self.all_available_inputs[machine_learning_type][method]

    def load_configuration(self):
        """Load configuration, and display_loaded_inputs_in_gui configuration in GUI (removed to get_current_inputs method)
        """

        # Get current inputs before load configuration, so we can 
        # compare loaded configuration["machine_learning"] with the current self.machine_learning

        # Scan the current GUI first and get current inputs, so that to compare with loaded configuration
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
                            self.display_loaded_inputs_in_gui()
                    # If the loaded self.configuration["machine_learning"] is empty
                     # Then assign self.configuration["machine_learning"] with self.machine_learning
                    else:
                        self.configuration["machine_learning"] = self.machine_learning
                else:
                    self.machine_learning = self.configuration["machine_learning"]
                    self.display_loaded_inputs_in_gui()
            except json.decoder.JSONDecodeError:
                QMessageBox.warning( self, 'Warning', f'{self.configuration_file} is not valid JSON')
                self.configuration_file = ""
   
        else:
            QMessageBox.warning( self, 'Warning', 'Configuration file was not selected')

    def save_configuration(self):
        """Save configuration that users inputed.
        """

        # Refresh the current GUI first and get current inputs
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
                self.configuration["machine_learning"] = self.machine_learning
                with open(self.configuration_file, 'w', encoding="utf-8") as config:   
                    config.write( json.dumps(self.configuration, ensure_ascii=False, indent=4) )
            except json.decoder.JSONDecodeError:
                QMessageBox.warning( self, 'Warning', f'{self.configuration}'+ ' is not a valid JSON!')

        else:
            QMessageBox.warning( self, 'Warning', 'Please choose a configuration file first (press button at top left corner)!')


    #%% Update GUI: including display_loaded_inputs_in_gui and switche_stacked_wedge_for_*
    def display_loaded_inputs_in_gui(self):
        """ Display the loaded configuration in the GUI
        """

        # Generate a dict for switch stacked wedgets
        method_switch_dict = {
            "Classification": self.switche_stacked_wedge_for_classification,
            "Regression": self.switche_stacked_wedge_for_regression,
            "Clustering": self.switche_stacked_wedge_for_clustering,
            "Deep learning": self.switche_stacked_wedge_for_deep_learning,
        }

        for machine_learning_type in list(self.all_available_inputs.keys()):
            if machine_learning_type in self.machine_learning.keys():  # Avoiding duplicate machine_learning_type selection, because users may clicked multiple methods in different machine_learning_type
                # Click the input machine_learning_type wedget
                self.switche_stacked_wedge_for_machine_learning_type(True, machine_learning_type)
                self.machine_learning_type_dict[machine_learning_type].setChecked(True)  
                
                for method in self.all_available_inputs[machine_learning_type].keys():  
                    
                    if method.text() in list(self.machine_learning[machine_learning_type].keys()):  # TODO: Is it necessary to use "if"
                        # Click the input method wedget
                        # Switch stacked wedget
                        method_switch_dict[machine_learning_type](True, method.text())
                        method.setChecked(True) 
                        
                        # Click the input setting wedget
                        for key_setting in self.machine_learning[machine_learning_type][method.text()]:
                            if "wedget" in list(self.all_available_inputs[machine_learning_type][method][method.text()][key_setting].keys()):
                                loaded_text = self.machine_learning[machine_learning_type][method.text()][key_setting]["value"]
                                # Identify wedget type, then using different methods to "setText"

                                # In the design of all wedgets, try to keep the name of the original control in the front part of the name, 
                                # so that it is easy to determine which type of control to use different assignment methods!
                                # TODO: Update point
                                if "lineEdit" in self.all_available_inputs[machine_learning_type][method][method.text()][key_setting]["wedget"].objectName():
                                    self.all_available_inputs[machine_learning_type][method][method.text()][key_setting]["wedget"].setText(loaded_text)
                                elif "doubleSpinBox" in self.all_available_inputs[machine_learning_type][method][method.text()][key_setting]["wedget"].objectName():
                                    self.all_available_inputs[machine_learning_type][method][method.text()][key_setting]["wedget"].setValue(float(loaded_text))
                                elif "spinBox" in self.all_available_inputs[machine_learning_type][method][method.text()][key_setting]["wedget"].objectName():
                                    self.all_available_inputs[machine_learning_type][method][method.text()][key_setting]["wedget"].setValue(int(loaded_text))
                                elif "comboBox" in self.all_available_inputs[machine_learning_type][method][method.text()][key_setting]["wedget"].objectName():
                                    self.all_available_inputs[machine_learning_type][method][method.text()][key_setting]["wedget"].setCurrentText(loaded_text)
                break
                


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
                self.stackedWidget_clustering.setCurrentIndex(self.clustering_stackedwedge_dict[self.sender().text()])
            else:
                self.stackedWidget_clustering.setCurrentIndex(self.clustering_stackedwedge_dict[method])
        else:
            self.stackedWidget_clustering.setCurrentIndex(-1)

    def switche_stacked_wedge_for_deep_learning(self, signal_bool, method=None):
        """ Switch to corresponding deep learning model window
        """
        self.radioButton_deeplearning.setChecked(True)

        if self.sender():
            if not method:
                self.stackedWidget_deeplearning_setting.setCurrentIndex(self.deep_learning_stackedwedge_dict[self.sender().text()])
            else:
                self.stackedWidget_classification_setting.setCurrentIndex(self.deep_learning_stackedwedge_dict[method])
        else:
            self.stackedWidget_classification_setting.setCurrentIndex(-1)

    def eegclf_prepare_data(self):
        self.eegclf = EEGClassifier(configuration_file=self.configuration_file)
        self.eegclf.prepare_data()
        return self

    def eegclf_train_fun(self):
        self.eegclf.train()
        return self

    def eegclf_eval_fun(self):
        self.eegclf.eval()
        return self

    def eegclf_save_fun(self):
        self.eegclf.save_model_and_loss()
        return self

    def eegclf_train_with_pretrained_model_fun(self):
        self.eegclf.train_with_pretrained_model()
        return self

    def eegclf_test_fun(self):
        self.eegclf.test()
        return self

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
    md=EasylearnMachineLearningRun()
    md.show()
    sys.exit(app.exec_())
