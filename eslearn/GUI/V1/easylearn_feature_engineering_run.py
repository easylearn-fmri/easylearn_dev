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
# from PyQt5.QtCore import *
from eslearn.stylesheets.PyQt5_stylesheets import PyQt5_stylesheets

from easylearn_feature_engineering_gui import Ui_MainWindow


class EasylearnFeatureEngineeringRun(QMainWindow, Ui_MainWindow):
    def __init__(self, working_directory=None):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        # Initialization
        self.feature_engineering = {}
        self.configuration_file = ""
        self.all_available_inputs()

        # Debug
        # Set working_directory
        self.working_directory = working_directory
        if self.working_directory:
            cgitb.enable(format="text", display=1, logdir=os.path.join(self.working_directory, "log_feature_engineering"))
        else:
            cgitb.enable(display=1, logdir=None)  

        # Connect configuration functions
        self.actionLoad_configuration.triggered.connect(self.load_configuration)
        self.actionSave_configuration.triggered.connect(self.save_configuration)
        self.actionGet_all_available_configuraton.triggered.connect(self._get_all_available_inputs)

        # connect preprocessing setting signal to slot: switche to corresponding stackedWidget
        self.preprocessing_stackedwedge_dict = {"Z-score normalization": 0, "Scaling": 1, "De-mean": 2, "None": 3}
        self.radioButton_zscore.clicked.connect(self.switche_stacked_wedge_for_preprocessing)
        self.radioButton_scaling.clicked.connect(self.switche_stacked_wedge_for_preprocessing)
        self.radioButton_demean.clicked.connect(self.switche_stacked_wedge_for_preprocessing)
        self.radioButton_none_methods.clicked.connect(self.switche_stacked_wedge_for_preprocessing)
        
        # connect dimreduction setting signal to slot: switche to corresponding stackedWidget
        self.dimreduction_stackedwedge_dict = {
            "Principal component analysis": 0, "Independent component analysis": 1, 
            "Latent Dirichlet Allocation": 2, "Non-negative matrix factorization": 3, "None": 4
        }
        self.radioButton_pca.clicked.connect(self.switche_stacked_wedge_for_dimreduction)
        self.radioButton_ica.clicked.connect(self.switche_stacked_wedge_for_dimreduction)
        self.radioButton_lda.clicked.connect(self.switche_stacked_wedge_for_dimreduction)
        self.radioButton_nmf.clicked.connect(self.switche_stacked_wedge_for_dimreduction)
        self.radioButton_none.clicked.connect(self.switche_stacked_wedge_for_dimreduction)
        
        # connect feature selection setting signal to slot: switche to corresponding stackedWidget
        self.feature_selection_stackedwedge_dict = {
            "Variance threshold": 0, "Correlation": 1, "Distance correlation": 2, "F-Score (classification)": 3, 
            "Mutual information (classification)": 4, "Mutual information (regression)": 5, "ReliefF": 6, "ANOVA/Ttest2 (classification)": 7, 
            "RFE": 8, 
            "L1 regularization (Lasso)": 9, "L1 + L2 regularization (Elastic net regression)": 10, 
            "None": 11
        }
        self.radioButton_variance_threshold.clicked.connect(self.switche_stacked_wedge_for_feature_selection)
        self.radioButton_correlation.clicked.connect(self.switche_stacked_wedge_for_feature_selection)
        self.radioButton_distancecorrelation.clicked.connect(self.switche_stacked_wedge_for_feature_selection)
        self.radioButton_fscore.clicked.connect(self.switche_stacked_wedge_for_feature_selection)
        self.radioButton_mutualinfo_cls.clicked.connect(self.switche_stacked_wedge_for_feature_selection)
        self.radioButton_mutualinfo_regression.clicked.connect(self.switche_stacked_wedge_for_feature_selection)
        self.radioButton_relieff.clicked.connect(self.switche_stacked_wedge_for_feature_selection)
        self.radioButton_anova.clicked.connect(self.switche_stacked_wedge_for_feature_selection)
        self.radioButton_rfe.clicked.connect(self.switche_stacked_wedge_for_feature_selection)
        self.radioButton_l1.clicked.connect(self.switche_stacked_wedge_for_feature_selection)
        self.radioButton_elasticnet.clicked.connect(self.switche_stacked_wedge_for_feature_selection)
        self.radioButton_featureselection_none.clicked.connect(self.switche_stacked_wedge_for_feature_selection)

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
        self.setWindowTitle('Feature Engineering')
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
        self.tabWidget_items.setCurrentIndex(0)
        self.stackedWidget_preprocessing_methods.setCurrentIndex(-1)
        self.stackedWidget_dimreduction.setCurrentIndex(-1)
        self.stackedWidget_feature_selection.setCurrentIndex(-1)

    def all_available_inputs(self):
        """I put all available inputs in a dictionary named all_available_inputs
        """

        self.all_available_inputs = {
            "feature_preprocessing": {
                self.radioButton_zscore : {"Z-score normalization": {}}, 
                self.radioButton_scaling: {
                    "Scaling": {
                        "min": {"value": self.lineEdit_scaling_min.text(), "wedget": self.lineEdit_scaling_min}, 
                        "max": {"value": self.lineEdit_scaling_max.text(), "wedget": self.lineEdit_scaling_max},
                    }
                }, 

                self.radioButton_demean: {"demean": {}}, 
                self.radioButton_none_methods: {"none": {}}, 
                self.radioButton_grouplevel: {"grouplevel": {}}, 
                self.radioButton_subjectlevel: {"subjectlevel": {}}
            },

            "dimreduction": {
                self.radioButton_pca: {
                    "Principal component analysis": {
                        "min": {"value": self.doubleSpinBox_pca_maxcomponents.text(), "wedget": self.doubleSpinBox_pca_maxcomponents}, 
                        "max": {"value": self.doubleSpinBox_pca_mincomponents.text(), "wedget": self.doubleSpinBox_pca_mincomponents}, 
                        "number": {"value": self.spinBox_pcanum.text(), "wedget": self.spinBox_pcanum}
                    }, 
                },

                self.radioButton_ica: {
                    "Independent component analysis": { 
                            "min": {"value": self.doubleSpinBox_ica_minics.text(), "wedget": self.doubleSpinBox_ica_minics}, 
                            "max": {"value": self.doubleSpinBox_ica_maxics.text(), "wedget": self.doubleSpinBox_ica_maxics}, 
                            "number": {"value": self.spinBox_icnum.text(), "wedget": self.spinBox_icnum},
                    }
                },

                self.radioButton_lda: {"lda": {}},

                self.radioButton_nmf: {
                    "Non-negative matrix factorization": {
                        "min": {"value": self.doubleSpinBox_nmf_mincompnents.text(), "wedget": self.doubleSpinBox_nmf_mincompnents}, 
                        "max": {"value": self.doubleSpinBox_nmf_maxcomponents.text(), "wedget": self.doubleSpinBox_nmf_maxcomponents}, 
                        "number": {"value": self.spinBox_icnum.text(), "wedget": self.spinBox_icnum},
                    }
                },

                self.radioButton_none: {"none": {}}
            },

            "feature_selection": {
                self.radioButton_variance_threshold: {
                    "Variance threshold": {
                        "min": {"value": self.doubleSpinBox_variancethreshold_min.text(), "wedget": self.doubleSpinBox_variancethreshold_min}, 
                        "max": {"value": self.doubleSpinBox_variancethreshold_max.text(), "wedget": self.doubleSpinBox_variancethreshold_max}, 
                        "number": {"value": self.spinBox_variancethreshold_num.text(), "wedget": self.spinBox_variancethreshold_num}
                    }
                },

                self.radioButton_correlation: {
                    "Correlation": {
                        "min": {"value": self.doubleSpinBox_correlation_minabscoef.text(), "wedget": self.doubleSpinBox_correlation_minabscoef}, 
                        "max": {"value": self.doubleSpinBox_correlation_maxabscoef.text(), "wedget": self.doubleSpinBox_correlation_maxabscoef}, 
                        "number": {"value": self.spinBox_correlation_num.text(), "wedget": self.spinBox_correlation_num},
                    }
                }, 

                self.radioButton_distancecorrelation: {
                    "Distance correlation": {
                        "min": {"value": self.doubleSpinBox_distancecorrelation_minabscoef.text(), "wedget": self.doubleSpinBox_distancecorrelation_minabscoef}, 
                        "max": {"value": self.doubleSpinBox_distancecorrelation_maxabscoef.text(), "wedget": self.doubleSpinBox_distancecorrelation_maxabscoef}, 
                        "number": {"value": self.spinBox_distancecorrelation_num.text(), "wedget": self.spinBox_distancecorrelation_num},
                    }
                },

                self.radioButton_fscore: {
                    "F-Score (classification)": {
                        "max":{"value": self.doubleSpinBox_fscore_maxnum.text(), "wedget": self.doubleSpinBox_fscore_maxnum}, 
                        "min": {"value":self.doubleSpinBox_fscore_minnum.text(), "wedget": self.doubleSpinBox_fscore_minnum}, 
                        "number": {"value":self.spinBox_fscore_num.text(), "wedget": self.spinBox_fscore_num},
                    }
                }, 

                self.radioButton_mutualinfo_cls: {
                    "Mutual information (classification)": {
                        "max": {"value": self.doubleSpinBox_mutualinfocls_maxnum.text(), "wedget": self.doubleSpinBox_mutualinfocls_maxnum}, 
                        "min": {"value": self.doubleSpinBox_mutualinfocls_minnum.text(), "wedget": self.doubleSpinBox_mutualinfocls_minnum},
                        "number": {"value": self.spinBox_mutualinfocls_num.text(), "wedget":  self.spinBox_mutualinfocls_num},
                        "n_neighbors": {"value": self.spinBox_mutualinfocls_neighbors.text(), "wedget": self.spinBox_mutualinfocls_neighbors},
                    }
                }, 

                self.radioButton_mutualinfo_regression: {
                    "Mutual information (regression)": {
                        "max": {"value": self.doubleSpinBox_mutualinforeg_maxnum.text(), "wedget": self.doubleSpinBox_mutualinforeg_maxnum}, 
                        "min": {"value": self.doubleSpinBox_mutualinforeg_minnum.text(), "wedget": self.doubleSpinBox_mutualinforeg_minnum},
                        "number": {"value": self.spinBox_mutualinforeg_num.text(), "wedget":  self.spinBox_mutualinforeg_num},
                        "n_neighbors": {"value": self.spinBox_mutualinforeg_neighbors.text(), "wedget": self.spinBox_mutualinforeg_neighbors},
                    }
                }, 

                self.radioButton_relieff: {
                    "ReliefF": {
                        "max": {"value": self.doubleSpinBox_relieff_max.text(), "wedget": self.doubleSpinBox_relieff_max}, 
                        "min": {"value": self.doubleSpinBox_relieff_min.text(), "wedget": self.doubleSpinBox_relieff_min}, 
                        "number": {"value": self.spinBox_relief_num.text(), "wedget": self.spinBox_relief_num},
                    }
                }, 

                self.radioButton_anova: {
                    "ANOVA": {
                        "max": {"value": self.doubleSpinBox_anova_alpha_max.text(), "wedget": self.doubleSpinBox_anova_alpha_max}, 
                        "min": {"value": self.doubleSpinBox_anova_alpha_min.text(), "wedget": self.doubleSpinBox_anova_alpha_min}, 
                        "number": {"value": self.spinBox_anova_num.text(), "wedget": self.spinBox_anova_num}, 
                        "multiple_correction": {"value": self.comboBox_anova_multicorrect.currentText(), "wedget": self.comboBox_anova_multicorrect},
                    }
                }, 

                self.radioButton_rfe: {
                    "RFE": {
                        "step": {"value": self.doubleSpinBox_rfe_step.text(), "wedget": self.doubleSpinBox_rfe_step}, 
                        "n_folds": {"value": self.spinBox_rfe_nfold.text(), "wedget":  self.spinBox_rfe_nfold}, 
                        "estimator": {"value": self.comboBox_rfe_estimator.currentText(), "wedget": self.comboBox_rfe_estimator}, 
                        "n_jobs": {"value": self.spinBox_rfe_njobs.text(), "wedget": self.spinBox_rfe_njobs}
                    }
                },

                self.radioButton_l1: {
                    "L1 regularization (Lasso)": {
                        "max": {"va1ue": self.doubleSpinBox_l1_alpha_max.text(), "wedget": self.doubleSpinBox_l1_alpha_max}, 
                        "min": {"va1ue": self.doubleSpinBox_l1_alpha_min.text(), "wedget": self.doubleSpinBox_l1_alpha_min}, 
                        "number": {"va1ue": self.spinBox_l1_num.text(), "wedget": self.spinBox_l1_num}
                    }
                }, 

                self.radioButton_elasticnet: {
                    "L1 + L2 regularization (Elastic net regression)": {
                        "max_alpha": {"value": self.doubleSpinBox_elasticnet_alpha_max.text(), "wedget": self.doubleSpinBox_elasticnet_alpha_max}, 
                        "min_alpha": {"value": self.doubleSpinBox_elasticnet_alpha_min.text(), "wedget": self.doubleSpinBox_elasticnet_alpha_min}, 
                        "number_alpha": {"value": self.spinBox_elasticnet_num.text(), "wedget": self.spinBox_elasticnet_num}, 
                        "max_l1ratio": {"value": self.doubleSpinBox_elasticnet_l1ratio_max.text(), "wedget": self.doubleSpinBox_elasticnet_l1ratio_max}, 
                        "min_l1ratio": {"value": self.doubleSpinBox_elasticnet_l1ratio_min.text(), "wedget":  self.doubleSpinBox_elasticnet_l1ratio_min}, 
                        "Number_l1ratio": {"value": self.spinBox_l1ratio_num.text(), "wedget":  self.spinBox_l1ratio_num},
                    }
                }
            },

            "unbalance_treatment": {
                self.radioButton_randover: {"randover": {}}, 
                self.radioButton_smoteover: {"somteover": {}},
                self.radioButton_smotencover: {"somtencover": {}}, 
                self.radioButton_bsmoteover: {"bsmoteover": {}},
                self.radioButton_randunder: {"randunder": {}}, 
                self.radioButton_extractionunder: {"extractionunder": {}},
                self.radioButton_cludterunder: {"clusterunder": {}}, 
                self.radioButton_nearmissunder: {"nearmissunder": {}},
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

        # Get current inputs
        for key_feature_engineering in self.all_available_inputs:
            for keys_one_feature_engineering in self.all_available_inputs[key_feature_engineering]:
                if keys_one_feature_engineering.isChecked():
                    self.feature_engineering[key_feature_engineering] = self.all_available_inputs[key_feature_engineering][keys_one_feature_engineering]
    
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
                                                    "The feature_engineering configuration is already exists, do you want to rewrite it with the  loaded configuration?",
                                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
           
                        if reply == QMessageBox.Yes:  
                            self.feature_engineering = self.configuration["feature_engineering"]
                            self.refresh_gui()
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
        """ Refresh gui the display the loaded configuration in the GUI
        """

        # Generate a dict for switch stacked wedgets
        switch_dict = {
            "feature_preprocessing": self.switche_stacked_wedge_for_preprocessing,
            "dimreduction": self.switche_stacked_wedge_for_dimreduction,
            "feature_selection": self.switche_stacked_wedge_for_feature_selection,
        }

        for keys_one_feature_engineering in self.all_available_inputs:  # 4 feature eng module loop
            for wedget in self.all_available_inputs[keys_one_feature_engineering].keys():  # all wedgets in one feature eng loop
                for method in self.all_available_inputs[keys_one_feature_engineering][wedget].keys():
                    if keys_one_feature_engineering in self.feature_engineering.keys():
                        if method in list(self.feature_engineering[keys_one_feature_engineering].keys()):
                            # Make the wedget checked according loaded param
                            wedget.setChecked(True)   
                            # Make setting to loaded text
                            for key_setting in self.feature_engineering[keys_one_feature_engineering][method]:
                                if "wedget" in list(self.all_available_inputs[keys_one_feature_engineering][wedget][method][key_setting].keys()):
                                    loaded_text = self.feature_engineering[keys_one_feature_engineering][method][key_setting]["value"]
                                    # Identity wedget type, then using different methods to "setText"
                                    # NOTE. 所有控件在设计时，尽量保留原控件的名字在命名的前部分，这样下面才好确定时哪一种类型的控件，从而用不同的赋值方式！
                                    if "lineEdit" in self.all_available_inputs[keys_one_feature_engineering][wedget][method][key_setting]["wedget"].objectName():
                                        self.all_available_inputs[keys_one_feature_engineering][wedget][method][key_setting]["wedget"].setText(loaded_text)
                                    elif "doubleSpinBox" in self.all_available_inputs[keys_one_feature_engineering][wedget][method][key_setting]["wedget"].objectName():
                                        self.all_available_inputs[keys_one_feature_engineering][wedget][method][key_setting]["wedget"].setValue(float(loaded_text))
                                    elif "spinBox" in self.all_available_inputs[keys_one_feature_engineering][wedget][method][key_setting]["wedget"].objectName():
                                        self.all_available_inputs[keys_one_feature_engineering][wedget][method][key_setting]["wedget"].setValue(int(loaded_text))
                                    elif "comboBox" in self.all_available_inputs[keys_one_feature_engineering][wedget][method][key_setting]["wedget"].objectName():
                                        self.all_available_inputs[keys_one_feature_engineering][wedget][method][key_setting]["wedget"].setCurrentText(loaded_text)
                                    
                                # Switch stacked wedget
                                switch_dict[keys_one_feature_engineering](True, method)

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
                # self.configuration = json.dumps(self.configuration, ensure_ascii=False)
                # print(self.feature_engineering)
                with open(self.configuration_file, 'w', encoding="utf-8") as config:    
                    config.write(json.dumps(self.configuration, ensure_ascii=False, indent=4))
            except json.decoder.JSONDecodeError:
                QMessageBox.warning( self, 'Warning', f'{self.configuration}'+ ' is not a valid JSON!')

        else:
            QMessageBox.warning( self, 'Warning', 'Please choose a configuration file first (press button at top left corner)!')

    def switche_stacked_wedge_for_preprocessing(self, signal_bool, method=None):
        if self.sender().text():
            if not method:
                self.stackedWidget_preprocessing_methods.setCurrentIndex(self.preprocessing_stackedwedge_dict[self.sender().text()])
            else:
                self.stackedWidget_preprocessing_methods.setCurrentIndex(self.preprocessing_stackedwedge_dict[method])
        else:
            self.stackedWidget_preprocessing_methods.setCurrentIndex(-1)

    def switche_stacked_wedge_for_dimreduction(self, signal_bool, method=None):
        if self.sender():
            if not method:
                self.stackedWidget_dimreduction.setCurrentIndex(self.dimreduction_stackedwedge_dict[self.sender().text()])
            else:
                self.stackedWidget_dimreduction.setCurrentIndex(self.dimreduction_stackedwedge_dict[method])
        else:
            self.stackedWidget_dimreduction.setCurrentIndex(-1)

    def switche_stacked_wedge_for_feature_selection(self, signal_bool, method=None):
        self.groupBox_feature_selection_input.setTitle(self.sender().text())
        if self.sender().text():
            if not method:
                self.stackedWidget_feature_selection.setCurrentIndex(self.feature_selection_stackedwedge_dict[self.sender().text()])
            else:
                self.stackedWidget_feature_selection.setCurrentIndex(self.feature_selection_stackedwedge_dict[method])
        else:
            self.stackedWidget_feature_selection.setCurrentIndex(-1)

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
    md=EasylearnFeatureEngineeringRun()
    md.show()
    sys.exit(app.exec_())
