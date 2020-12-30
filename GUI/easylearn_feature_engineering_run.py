 # -*- coding: utf-8 -*-
"""The GUI of the feature_engineering module of easylearn

Created on 2020/04/12

@author: Li Chao
Email:lichao19870617@gmail.com
GitHub account name: lichao312214129
Institution (company): Brain Function Research Section, The First Affiliated Hospital of China Medical University, Shenyang, Liaoning, PR China. 

@author: Dong Mengshi
Email:dongmengshi1990@163.com
GitHub account name: dongmengshi
Institution (company): Department of radiology, The First Affiliated Hospital of China Medical University, Shenyang, Liaoning, PR China. 
License: MIT
"""


import sys
import os
import json
import cgitb
# from PyQt5 import *
# from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from eslearn.stylesheets.PyQt5_stylesheets import pyqt5_loader

import eslearn
from eslearn.GUI.easylearn_feature_engineering_gui import Ui_MainWindow


class EasylearnFeatureEngineeringRun(QMainWindow, Ui_MainWindow):
    def __init__(self, working_directory=None):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.root_dir = os.path.dirname(eslearn.__file__)

        # Initialization
        self.feature_engineering = {}
        self.working_directory = working_directory
        self.configuration_file = ""
        self.configuration = {}
        # self.all_available_inputs_fun()

        # Debug
        # Set working_directory
        if self.working_directory:
            cgitb.enable(format="text", display=1, logdir=os.path.join(self.working_directory, "log_feature_engineering"))
        else:
            cgitb.enable(display=1, logdir=None)  

        # Connect configuration functions
        self.actionLoad_configuration.triggered.connect(self.load_configuration)
        self.actionSave_configuration.triggered.connect(self.save_configuration)
        self.actionGet_all_available_configuraton.triggered.connect(self._get_all_available_inputs)

        # connect preprocessing setting signal to slot: switche to corresponding stackedWidget
        self.preprocessing_stackedwedge_dict = {"StandardScaler()": 0, "MinMaxScaler()": 1, "None": 2}
        self.radioButton_zscore.clicked.connect(self.switche_stacked_wedge_for_preprocessing)
        self.radioButton_scaling.clicked.connect(self.switche_stacked_wedge_for_preprocessing)
        self.radioButton_none_methods.clicked.connect(self.switche_stacked_wedge_for_preprocessing)
        
        # connect dimreduction setting signal to slot: switche to corresponding stackedWidget
        self.dimreduction_stackedwedge_dict = {"PCA()": 0, "NMF()": 1, "None": 2}
        self.radioButton_pca.clicked.connect(self.switche_stacked_wedge_for_dimreduction)
        self.radioButton_nmf.clicked.connect(self.switche_stacked_wedge_for_dimreduction)
        self.radioButton_none.clicked.connect(self.switche_stacked_wedge_for_dimreduction)
        
        # connect feature selection setting signal to slot: switche to corresponding stackedWidget

        self.feature_selection_stackedwedge_dict = {
            "VarianceThreshold()": 0, "SelectPercentile(f_classif)": 1, "SelectPercentile(f_regression)": 2, 
            "SelectPercentile(mutual_info_classif)": 3, "SelectPercentile(mutual_info_regression)": 4,  
            "RFE()": 5, 
            "SelectFromModel(LassoCV())": 6, "SelectFromModel(ElasticNetCV())": 7, 
            "None": 8
        }
        self.radioButton_variance_threshold.clicked.connect(self.switche_stacked_wedge_for_feature_selection)
        self.radioButton_correlation.clicked.connect(self.switche_stacked_wedge_for_feature_selection)
        self.radioButton_mutualinfo_cls.clicked.connect(self.switche_stacked_wedge_for_feature_selection)
        self.radioButton_mutualinfo_regression.clicked.connect(self.switche_stacked_wedge_for_feature_selection)
        self.radioButton_anova.clicked.connect(self.switche_stacked_wedge_for_feature_selection)
        self.radioButton_rfe.clicked.connect(self.switche_stacked_wedge_for_feature_selection)
        self.radioButton_l1.clicked.connect(self.switche_stacked_wedge_for_feature_selection)
        self.radioButton_elasticnet.clicked.connect(self.switche_stacked_wedge_for_feature_selection)
        self.radioButton_featureselection_none.clicked.connect(self.switche_stacked_wedge_for_feature_selection)

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
        self.setWindowTitle('Feature Engineering')
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

        # Make the stackedWidg to default at the begining
        self.tabWidget_items.setCurrentIndex(0)
        self.stackedWidget_preprocessing_methods.setCurrentIndex(-1)
        self.stackedWidget_dimreduction.setCurrentIndex(-1)
        self.stackedWidget_feature_selection.setCurrentIndex(-1)

    def all_available_inputs_fun(self):
        """I put all available inputs in a dictionary named all_available_inputs

        All potential wedget are also in the dictionary for reloading parameters.
        """

        self.all_available_inputs = {
            "feature_preprocessing": {
                self.radioButton_zscore : {"StandardScaler()": {}}, 
                self.radioButton_scaling: {
                    "MinMaxScaler()": {
                        "feature_range": {"value": self.lineEdit_scaling_feature_range.text(), "wedget": self.lineEdit_scaling_feature_range}, 
                    }
                }, 

                self.radioButton_none_methods: {"None": {}}, 
                # self.radioButton_grouplevel: {"grouplevel": {}}, 
                # self.radioButton_subjectlevel: {"subjectlevel": {}}
            },

            "dimreduction": {
                self.radioButton_pca: {
                    "PCA()": {
                        "n_components": {"value": self.lineEdit_pca_components.text(), "wedget": self.lineEdit_pca_components}, 
                    }, 
                },

                self.radioButton_nmf: {
                    "NMF()": {
                        "n_components": {"value": self.lineEdit_nmf_components.text(), "wedget": self.lineEdit_nmf_components}, 
                    }
                },

                self.radioButton_none: {
                    "None": {
                        
                    }
                }
            },

            "feature_selection": {
                self.radioButton_variance_threshold: {
                    "VarianceThreshold()": {
                        "threshold": {"value": self.lineEdit_variancethreshold_threshold.text(), "wedget": self.lineEdit_variancethreshold_threshold}, 
                    }
                },

                self.radioButton_correlation: {
                    "SelectPercentile(f_regression)": {
                        "percentile": {"value": self.lineEdit_correlation_percentile.text(), "wedget": self.lineEdit_correlation_percentile}, 
                    }
                }, 

                self.radioButton_mutualinfo_cls: {
                    "SelectPercentile(mutual_info_classif)": {
                        "percentile": {"value": self.lineEdit_mutualinfocls_topnum.text(), "wedget": self.lineEdit_mutualinfocls_topnum}, 
                    }
                }, 

                self.radioButton_mutualinfo_regression: {
                    "SelectPercentile(mutual_info_regression)": {
                        "percentile": {"value": self.lineEdit_mutualinforeg_topnum.text(), "wedget": self.lineEdit_mutualinforeg_topnum}, 
                    }
                }, 

                self.radioButton_anova: {
                    "SelectPercentile(f_classif)": {
                        "percentile": {"value": self.lineEdit_anova_topnum.text(), "wedget": self.lineEdit_anova_topnum}, 
                    }
                }, 

                self.radioButton_rfe: {
                    "RFE()": {
                        "step": {"value": self.doubleSpinBox_rfe_step.text(), "wedget": self.doubleSpinBox_rfe_step}, 
                        # "cv": {"value": self.spinBox_rfe_nfold.text(), "wedget":  self.spinBox_rfe_nfold}, 
                        "estimator": {"value": self.comboBox_rfe_estimator.currentText(), "wedget": self.comboBox_rfe_estimator}, 
                        # "n_jobs": {"value": self.spinBox_rfe_njobs.text(), "wedget": self.spinBox_rfe_njobs}
                    }
                },

                self.radioButton_l1: {
                    "SelectFromModel(LassoCV())": {
                    }
                }, 

                self.radioButton_elasticnet: {
                    "SelectFromModel(ElasticNetCV())": {
                        "l1_ratio": {"value": self.lineEdit_elasticnet_l1ratio.text(), "wedget": self.lineEdit_elasticnet_l1ratio}, 
                    }
                },

                self.radioButton_featureselection_none: {
                    "None": {
                    }
                }
            },

            "unbalance_treatment": {
                self.radioButton_randover: {"RandomOverSampler()": {}},
                self.radioButton_smoteover: {"SMOTE()": {}},
                # self.radioButton_smotencover: {"SMOTENC()": {}}, 
                self.radioButton_bsmoteover: {"BorderlineSMOTE()": {}},
                self.radioButton_randunder: {"RandomUnderSampler()": {}}, 
                self.radioButton_cludterunder: {"ClusterCentroids()": {}}, 
                self.radioButton_nearmissunder: {"NearMiss()": {}},
            }
        }

    def _get_all_available_inputs(self):
        """ This method used to get all available inputs for users
        
        Delete wedgets object from all available inputs dict
        NOTE: This code is only for current configuration structure
        """

        all_available_inputs_for_user_tmp = self.all_available_inputs
        for feature_engineering_name in list(all_available_inputs_for_user_tmp.keys()):
            for method in list(all_available_inputs_for_user_tmp[feature_engineering_name].keys()):
                for method_name in list(all_available_inputs_for_user_tmp[feature_engineering_name][method].keys()):
                    for setting in list(all_available_inputs_for_user_tmp[feature_engineering_name][method][method_name].keys()):
                        if "wedget" in list(all_available_inputs_for_user_tmp[feature_engineering_name][method][method_name][setting].keys()):
                            all_available_inputs_for_user_tmp[feature_engineering_name][method][method_name][setting].pop("wedget")

        all_available_inputs_for_user = {}
        for feature_engineering_name in list(all_available_inputs_for_user_tmp.keys()):
            all_available_inputs_for_user[feature_engineering_name] = {}
            for method in list(all_available_inputs_for_user_tmp[feature_engineering_name].keys()):
                all_available_inputs_for_user[feature_engineering_name].update(all_available_inputs_for_user_tmp[feature_engineering_name][method])
        del all_available_inputs_for_user_tmp

        # Save to folder that contains configuration file
        if self.configuration_file != "":
            outname = os.path.join(os.path.dirname(self.configuration_file), 'all_available_feature_engineering_inputs.json')
            with open(outname, 'w', encoding="utf-8") as config:    
                config.write(json.dumps(all_available_inputs_for_user, indent=4))
        else:
            QMessageBox.warning( self, 'Warning', "configuration file is not selected!")


    def get_current_inputs(self):
        """Get all current inputs

        Programme will scan the GUI to determine the user's inputs.

        Attrs:
        -----
            self.feature_engineering: dictionary
                all feature_engineering parameters that the user input.
        """
        
        # Refresh self.all_availble_inputs
        self.all_available_inputs_fun()
        
        # Get current inputs
        self.feature_engineering = {}  # Remember clear self.feature_engineering before give values to it.
        for key_feature_engineering in self.all_available_inputs:
            for keys_one_feature_engineering in self.all_available_inputs[key_feature_engineering]:
                if keys_one_feature_engineering.isChecked():
                    self.feature_engineering[key_feature_engineering] = self.all_available_inputs[key_feature_engineering][keys_one_feature_engineering]
    
        # print(self.feature_engineering)
        
    def load_configuration(self):
        """Load configuration, and refresh_gui configuration in GUI
        """

        # Get current inputs before load configuration, so we can 
        # compare loaded configuration["feature_engineering"] with the current self.feature_engineering
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
                # If already exists self.feature_engineering
                if (self.feature_engineering != {}):
                    # If the loaded self.configuration["feature_engineering"] is not empty
                    # Then ask if rewrite self.feature_engineering with self.configuration["feature_engineering"]
                    if (list(self.configuration["feature_engineering"].keys()) != []):
                        reply = QMessageBox.question(self, "Data loading configuration already exists", 
                                                    "The feature_engineering configuration is already exists, do you want to rewrite it with the loaded configuration?",
                                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
           
                        if reply == QMessageBox.Yes:  
                            self.feature_engineering = self.configuration["feature_engineering"]
                            self.refresh_gui()
                        # else:
                        #     self.configuration["feature_engineering"] = self.feature_engineering
                    # If the loaded self.configuration["feature_engineering"] is empty
                    # Then assign self.configuration["feature_engineering"] with self.feature_engineering
                    else:
                        self.configuration["feature_engineering"] = self.feature_engineering
                else:
                    self.feature_engineering = self.configuration["feature_engineering"]
                    self.refresh_gui()

            except json.decoder.JSONDecodeError:
                QMessageBox.warning( self, 'Warning', f'{self.configuration_file} is not valid JSON')
                self.configuration_file = ""
   
        else:

            QMessageBox.warning( self, 'Warning', 'Configuration file was not selected')

    def refresh_gui(self):
        """ Refresh gui to display the loaded configuration in the GUI
        """

        # Generate a dict for switch stacked wedgets
        switch_dict = {
            "feature_preprocessing": self.switche_stacked_wedge_for_preprocessing,
            "dimreduction": self.switche_stacked_wedge_for_dimreduction,
            "feature_selection": self.switche_stacked_wedge_for_feature_selection,
        }

        for keys_one_feature_engineering in self.all_available_inputs:  # 4 feature eng module loop
            if keys_one_feature_engineering in self.feature_engineering.keys():
                for wedget in self.all_available_inputs[keys_one_feature_engineering].keys():  # all wedgets in one feature eng loop
                    for method in self.all_available_inputs[keys_one_feature_engineering][wedget].keys():
                        if method in list(self.feature_engineering[keys_one_feature_engineering].keys()):
                            # Make the radiobutton wedget checked according loaded param
                            wedget.setChecked(True) 
                            if self.feature_engineering[keys_one_feature_engineering][method] != {}:
                                switch_dict[keys_one_feature_engineering](True, method)
                                
                            # Make setting to loaded text
                            for key_setting in self.feature_engineering[keys_one_feature_engineering][method]:
                                if "wedget" in list(self.all_available_inputs[keys_one_feature_engineering][wedget][method][key_setting].keys()):
                                    loaded_text = self.feature_engineering[keys_one_feature_engineering][method][key_setting]["value"]
                                    # Identity wedget type, then using different methods to "setText"
                                    # NOTE. In the design of all wedgets (in pyqt5 and disigner), 
                                    # make sure that keeping the name of the original wedgets in the first part of the name 
                                    # (e.g., "lineEdit_scaling_min" is a wedget name of a lineEdit wedget), 
                                    # so that the following can determine which type of wedgets, so as to use different assignment methods!
                                    if "lineEdit" in self.all_available_inputs[keys_one_feature_engineering][wedget][method][key_setting]["wedget"].objectName():
                                        self.all_available_inputs[keys_one_feature_engineering][wedget][method][key_setting]["wedget"].setText(loaded_text)
                                    elif "doubleSpinBox" in self.all_available_inputs[keys_one_feature_engineering][wedget][method][key_setting]["wedget"].objectName():
                                        self.all_available_inputs[keys_one_feature_engineering][wedget][method][key_setting]["wedget"].setValue(float(loaded_text))
                                    elif "spinBox" in self.all_available_inputs[keys_one_feature_engineering][wedget][method][key_setting]["wedget"].objectName():
                                        self.all_available_inputs[keys_one_feature_engineering][wedget][method][key_setting]["wedget"].setValue(int(loaded_text))
                                    elif "comboBox" in self.all_available_inputs[keys_one_feature_engineering][wedget][method][key_setting]["wedget"].objectName():
                                        self.all_available_inputs[keys_one_feature_engineering][wedget][method][key_setting]["wedget"].setCurrentText(loaded_text)

    def save_configuration(self):
        """Save configuration
        """

        # Get current inputs before saving feature_engineering parameters
        self.get_current_inputs()
    
        # Delete wedgets object from self.feature_engineering dict
        for feature_engineering_name in list(self.feature_engineering.keys()):
            for method_name in list(self.feature_engineering[feature_engineering_name].keys()):
                for setting in self.feature_engineering[feature_engineering_name][method_name]:
                    for content in list(self.feature_engineering[feature_engineering_name][method_name][setting].keys()):
                        if "wedget" in list(self.feature_engineering[feature_engineering_name][method_name][setting].keys()):
                            self.feature_engineering[feature_engineering_name][method_name][setting].pop("wedget")
        
        if self.configuration_file != "":
            try:
                self.configuration["feature_engineering"] = self.feature_engineering
                with open(self.configuration_file, 'w', encoding="utf-8") as config:    
                    config.write(json.dumps(self.configuration, ensure_ascii=False, indent=4))
            except json.decoder.JSONDecodeError:
                QMessageBox.warning( self, 'Warning', f'{self.configuration}'+ ' is not a valid JSON!')

        else:
            QMessageBox.warning( self, 'Warning', 'Please choose a configuration file first (press button at top left corner)!')

    def switche_stacked_wedge_for_preprocessing(self, signal_bool, method=None):
        self.groupBox_preprocessing_setting.setTitle(self.sender().text())
        if self.sender().text():
            if not method:
                self.stackedWidget_preprocessing_methods.setCurrentIndex(self.preprocessing_stackedwedge_dict[self.sender().text()])
            else:
                self.stackedWidget_preprocessing_methods.setCurrentIndex(self.preprocessing_stackedwedge_dict[method])
        else:
            self.stackedWidget_preprocessing_methods.setCurrentIndex(-1)

    def switche_stacked_wedge_for_dimreduction(self, signal_bool, method=None):
        self.groupBox_dimreduction_setting.setTitle(self.sender().text())
        if self.sender():
            if not method:
                self.stackedWidget_dimreduction.setCurrentIndex(self.dimreduction_stackedwedge_dict[self.sender().text()])
            else:
                self.stackedWidget_dimreduction.setCurrentIndex(self.dimreduction_stackedwedge_dict[method])
        else:
            self.stackedWidget_dimreduction.setCurrentIndex(-1)

    def switche_stacked_wedge_for_feature_selection(self, signal_bool, method=None):
        self.groupBox_feature_selection_setting.setTitle(self.sender().text())
        if self.sender().text():
            if not method:
                self.stackedWidget_feature_selection.setCurrentIndex(self.feature_selection_stackedwedge_dict[self.sender().text()])
            else:
                self.stackedWidget_feature_selection.setCurrentIndex(self.feature_selection_stackedwedge_dict[method])
        else:
            self.stackedWidget_feature_selection.setCurrentIndex(-1)

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
