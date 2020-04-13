# -*- coding: utf-8 -*-
"""The GUI of the feature_engineering module of easylearn

Created on 2020/04/12
@author: Li Chao 黎超
Email:lichao19870617@gmail.com
GitHub account name: lichao312214129
Institution (company): Brain Function Research Section, The First Affiliated Hospital of China Medical University, Shenyang, Liaoning, PR China. 

License: MIT
"""


import sys
sys.path.append('../stylesheets/PyQt5_stylesheets')
import os
import numpy as np
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

        # Initialization
        self.feature_engineering = {}
        self.configuration_file = ""

        # Set appearance
        self.set_run_appearance()

        # Debug
        cgitb.enable(display=1, logdir=None)  

        # Connect configuration functions
        self.actionLoad_configuration.triggered.connect(self.load_configuration)
        self.actionSave_configuration.triggered.connect(self.save_configuration)

        # connect preprocessing setting signal to slot: switche to corresponding stackedWidget
        self.preprocessing_stackedwedge_dict = {"Z-score normalization": 0, "Scaling": 1, "De-mean": 2, "None": 3}
        self.radioButton_zscore.clicked.connect(self.on_preprocessing_detail_stackedwedge_clicked)
        self.radioButton_scaling.clicked.connect(self.on_preprocessing_detail_stackedwedge_clicked)
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
        self.feature_selection_stackedwedge_dict = {
            "Variance threshold": 0, "Correlation": 1, "Distance correlation": 2, "F-Score (classification)": 3, 
            "Mutual information (classification)": 4, "Mutual information (regression)": 5, "ReliefF": 6, "ANOVA/Ttest2 (classification)": 7, 
            "RFE": 8, 
            "L1 regularization (Lasso)": 9, "L1 + L2 regularization (Elastic net regression)": 10, 
            "None": 11
        }
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
        self.radioButton_elasticnet.clicked.connect(self.on_feature_selection_stackedwedge_clicked)
        self.radioButton_featureselection_none.clicked.connect(self.on_feature_selection_stackedwedge_clicked)

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
        self.tabWidget_items.setCurrentIndex(0)
        self.stackedWidget_preprocessing_methods.setCurrentIndex(-1)
        self.stackedWidget_dimreduction.setCurrentIndex(-1)
        self.stackedWidget_feature_selection.setCurrentIndex(-1)

    def get_current_inputs(self):
        """Get all current inputs

        Attrs:
        -----
            self.feature_engineering: dictionary
                all feature_engineering parameters that the user input.
        """

        self.all_inputs = {
            "feature_preprocessing": {
                self.radioButton_zscore : {"zscore": {}}, 
                self.radioButton_scaling: {
                    "scaling": {
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
                    "pca": {
                        "min": {"value": self.doubleSpinBox_pca_maxcomponents.text(), "wedget": self.doubleSpinBox_pca_maxcomponents}, 
                        "max": {"value": self.doubleSpinBox_pca_mincomponents.text(), "wedget": self.doubleSpinBox_pca_mincomponents}, 
                        "number": {"value": self.spinBox_pcanum.text(), "wedget": self.spinBox_pcanum}
                    }, 
                },

                self.radioButton_ica: {
                    "ica": {
                            "min": self.doubleSpinBox_ica_minics.text(), 
                            "max": self.doubleSpinBox_ica_maxics.text(), 
                            "number": self.spinBox_icnum.text()
                    }
                },

                self.radioButton_lda: {"lda": {}},

                self.radioButton_nmf: {
                    "nmf": {
                        "min": self.doubleSpinBox_nmf_mincompnents.text(), 
                        "max": self.doubleSpinBox_nmf_maxcomponents.text(), 
                        "number": self.spinBox_icnum.text()
                    }
                },

                self.radioButton_none: {"none": {}}
            },

            "feature_selection": {
                self.radioButton_variance_threshold: {
                    "variance_threshold": {
                        "min": self.doubleSpinBox_variancethreshold_min.text(), 
                        "max": self.doubleSpinBox_variancethreshold_max.text(), 
                        "number": self.spinBox_variancethreshold_num.text()
                    }
                },

                self.radioButton_correlation: {
                    "correlation": {
                        "min": self.doubleSpinBox_correlation_minabscoef.text(), 
                        "max": self.doubleSpinBox_correlation_maxabscoef.text(), 
                        "number": self.spinBox_correlation_num.text()
                    }
                }, 

                self.radioButton_distancecorrelation: {
                    "distancecorrelation": {
                        "min": self.doubleSpinBox_distancecorrelation_minabscoef.text(), 
                        "max": self.doubleSpinBox_distancecorrelation_maxabscoef.text(), 
                        "number": self.spinBox_distancecorrelation_num.text()
                    }
                },

                self.radioButton_fscore: {
                    "fscore": {
                        "max": self.doubleSpinBox_fscore_maxnum.text(), 
                        "min": self.doubleSpinBox_fscore_minnum.text(), 
                        "number": self.spinBox_fscore_num.text()
                    }
                }, 

                self.radioButton_mutualinfo_cls: {
                    "mutualinfocls": {
                        "max": self.doubleSpinBox_mutualinfocls_maxnum.text(), 
                        "min": self.doubleSpinBox_mutualinfocls_minnum.text(),
                        "number": self.spinBox_mutualinfocls_neighbors.text()
                    }
                }, 

                self.radioButton_mutualinfo_regression: {
                    "mutualinforeg": {
                        "max": self.doubleSpinBox_mutualinforeg_maxnum.text(), 
                        "min": self.doubleSpinBox_mutualinforeg_minnum.text(), 
                        "number": self.spinBox_mutualinforeg_num.text(),
                        "n_neighbors": self.spinBox_mutualinforeg_neighbors.text()
                    }
                }, 

                self.radioButton_relieff: {
                    "reliff": {
                        "max": self.doubleSpinBox_relieff_max.text(), 
                        "min": self.doubleSpinBox_relieff_min.text(), 
                        "number": self.spinBox_relief_num.text()
                    }
                }, 

                self.radioButton_anova: {
                    "anova": {
                        "max": self.doubleSpinBox_anova_alpha_max.text(), 
                        "min": self.doubleSpinBox_anova_alpha_min.text(), 
                        "number": self.spinBox_anova_num.text(), 
                        "multiple_correction": self.comboBox_anova_multicorrect.currentText()
                    }
                }, 

                self.radioButton_rfe: {
                    "rfe": {
                            "step": self.doubleSpinBox_rfe_step.text(), 
                        "n_folds": self.spinBox_rfe_nfold.text(), 
                        "estimator": self.comboBox_rfe_estimator.currentText(), 
                        "n_jobs": self.spinBox_rfe_njobs.text()
                    }
                },

                self.radioButton_l1: {
                    "l1": {
                        "max": self.doubleSpinBox_l1_alpha_max.text(), 
                        "min": self.doubleSpinBox_l1_alpha_min.text(), 
                        "number": self.spinBox_l1_num.text()
                    }
                }, 

                self.radioButton_elasticnet: {
                    "elasticnet": {
                        "max_alpha": self.doubleSpinBox_elasticnet_alpha_max.text(), 
                        "min_alpha": self.doubleSpinBox_elasticnet_alpha_min.text(), 
                        "number_alpha": self.spinBox_elasticnet_num.text(), 
                        "max_l1ratio": self.doubleSpinBox_elasticnet_l1ratio_max.text(), 
                        "min_l1ratio": self.doubleSpinBox_elasticnet_l1ratio_min.text(), 
                        "Number_l1ratio": self.spinBox_l1ratio_num.text(),
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

        #%% ------------------------------------------------------------------------------------
        for key_feature_engineering in self.all_inputs:
            for keys_one_feature_engineering in self.all_inputs[key_feature_engineering]:
                if keys_one_feature_engineering.isChecked():
                    self.feature_engineering[key_feature_engineering] = self.all_inputs[key_feature_engineering][keys_one_feature_engineering]

    def load_configuration(self):
        """Load configuration, and display configuration in GUI
        """

        # Get current inputs before load configuration, so we can 
        # compare loaded configuration["feature_engineering"] with the current self.feature_engineering
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
                            self.display()
                    # If the loaded self.configuration["feature_engineering"] is empty
                     # Then assign self.configuration["feature_engineering"] with self.feature_engineering
                    else:
                        self.configuration["feature_engineering"] = self.feature_engineering
                else:
                    self.feature_engineering = self.configuration["feature_engineering"]
                    self.display()

            except json.decoder.JSONDecodeError:
                QMessageBox.warning( self, 'Warning', f'{self.configuration_file} is not valid JSON')
                self.configuration_file = ""
   
        else:

            QMessageBox.warning( self, 'Warning', 'Configuration file was not selected')

    def display(self):
        """ Display the loaded configuration in the GUI
        """
        for keys_one_feature_engineering in self.all_inputs:  # 4 feature eng module loop
            for wedget in self.all_inputs[keys_one_feature_engineering].keys():  # all wedgets in one feature eng loop
                for method in self.all_inputs[keys_one_feature_engineering][wedget].keys():
                    if keys_one_feature_engineering in self.feature_engineering.keys():
                        if method in list(self.feature_engineering[keys_one_feature_engineering].keys()):
                            wedget.setChecked(True)   # make the wedget checked ***

                        # self.lineEdit_scaling_min.setText(self.feature_engineering[keys_one_feature_engineering])
                        # self.lineEdit_scaling_max.setText(self.feature_engineering[keys_one_feature_engineering])

                        # old = {"scaling": {"min": "-1", "max": "1"}}}
                        # display = "scaling": {
                        #     "min": self.lineEdit_scaling_min.text(), 
                        #     "max": self.lineEdit_scaling_max.text(),
                        # }
                        # continue

    def save_configuration(self):
        """Save configuration
        """

        # Get current inputs before saving feature_engineering parameters
        self.get_current_inputs()
        
        if self.configuration_file != "":
            self.configuration["feature_engineering"] = self.feature_engineering
            with open(self.configuration_file, 'w', encoding="utf-8") as config:    
                # Set ensure_ascii=False to save Chinese correctly.
                config.write(json.dumps(self.configuration, ensure_ascii=False))
        else:
            QMessageBox.warning( self, 'Warning', 'Please choose a configuration file first (press button at top left corner)!')

    def on_preprocessing_detail_stackedwedge_clicked(self):
        # self.stackedWidget_preprocessing_methods.setCurrentIndex(0)
        if self.sender().text():
            self.stackedWidget_preprocessing_methods.setCurrentIndex(self.preprocessing_stackedwedge_dict[self.sender().text()])
        else:
            self.stackedWidget_preprocessing_methods.setCurrentIndex(-1)

    def on_dimreduction_stackedwedge_clicked(self):
        if self.sender():
            self.stackedWidget_dimreduction.setCurrentIndex(self.dimreduction_stackedwedge_dict[self.sender().text()])
        else:
            self.stackedWidget_dimreduction.setCurrentIndex(-1)

    def on_feature_selection_stackedwedge_clicked(self):
        self.groupBox_feature_selection_input.setTitle(self.sender().text())
        print(self.sender().text())
        if self.sender().text():
            self.stackedWidget_feature_selection.setCurrentIndex(self.feature_selection_stackedwedge_dict[self.sender().text()])
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
