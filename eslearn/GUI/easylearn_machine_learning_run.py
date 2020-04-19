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

        # connect classification setting signal to slot: switche to corresponding stackedWidget
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

    def get_current_inputs(self):
        """Get all current inputs

        Attrs:
        -----
            self.machine_learning: dictionary
                all machine_learning parameters that the user input.
        """

        self.all_backup_inputs = {
            "classification": {
                self.radioButton_classificaton_lr:{
                    "Logistic regression": {
                        "maxl1ratio": {"value": self.doubleSpinBox_clf_lr_maxl1ratio.text(), "wedget": self.doubleSpinBox_clf_lr_maxl1ratio},
                        "minl1ratio": {"value": self.doubleSpinBox_clf_lr_maxl1ratio.text(), "wedget": self.doubleSpinBox_clf_lr_minl1ration}, 
                        "numberl1ratio": {"value": self.spinBox__clf_lr_numl1ratio.text(), "wedget": self.spinBox__clf_lr_numl1ratio},
                    },
                }, 

                self.radioButton_classification_svm:{
                    "Support vector machine": {
                        "minl1ratio": {"value": self.doubleSpinBox_clf_svm_minc.text(), "wedget": self.doubleSpinBox_clf_svm_minc}, 
                        "maxl1ratio": {"value": self.doubleSpinBox_clf_svm_maxc.text(), "wedget": self.doubleSpinBox_clf_svm_maxc},
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

            "regression": {
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

        #%% ----------------------------------get current inputs---------------------------------------
        for key_feature_engineering in self.all_backup_inputs:
            for keys_one_feature_engineering in self.all_backup_inputs[key_feature_engineering]:
                if keys_one_feature_engineering.isChecked():
                    self.machine_learning[key_feature_engineering] = self.all_backup_inputs[key_feature_engineering][keys_one_feature_engineering]

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
                        reply = QMessageBox.question(self, "Data loading configuration already exists", 
                                                    "The machine_learning configuration is already exists, do you want to rewrite it with the  loaded configuration?",
                                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
           
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

    def refresh_gui(self):
        """ Refresh gui the display the loaded configuration in the GUI
        """

        print("refresh_gui")
        # Generate a dict for switch stacked wedgets
        switch_dict = {
            "feature_preprocessing": self.switche_stacked_wedge_for_preprocessing,
            "dimreduction": self.switche_stacked_wedge_for_classification,
            "feature_selection": self.switche_stacked_wedge_for_feature_selection,
        }

        for keys_one_feature_engineering in self.all_backup_inputs:  # 4 feature eng module loop
            for wedget in self.all_backup_inputs[keys_one_feature_engineering].keys():  # all wedgets in one feature eng loop
                for method in self.all_backup_inputs[keys_one_feature_engineering][wedget].keys():
                    if keys_one_feature_engineering in self.machine_learning.keys():
                        if method in list(self.machine_learning[keys_one_feature_engineering].keys()):
                            # Make the wedget checked according loaded param
                            wedget.setChecked(True)   
                            # Make setting to loaded text
                            for key_setting in self.machine_learning[keys_one_feature_engineering][method]:

                                print(keys_one_feature_engineering)
                                print(wedget)
                                print(key_setting)
                                print(self.all_backup_inputs[keys_one_feature_engineering][wedget][method][key_setting].keys())

                                if "wedget" in list(self.all_backup_inputs[keys_one_feature_engineering][wedget][method][key_setting].keys()):
                                    loaded_text = self.machine_learning[keys_one_feature_engineering][method][key_setting]["value"]
                                    print(f"method = {method}, setting = {key_setting}, loaded_text={loaded_text}") 

                                    # Identity wedget type, then using different methods to "setText"
                                    # NOTE. 所有控件在设计时，尽量保留原控件的名字在命名的前部分，这样下面才好确定时哪一种类型的控件，从而用不同的赋值方式！
                                    if "lineEdit" in self.all_backup_inputs[keys_one_feature_engineering][wedget][method][key_setting]["wedget"].objectName():
                                        self.all_backup_inputs[keys_one_feature_engineering][wedget][method][key_setting]["wedget"].setText(loaded_text)
                                    elif "doubleSpinBox" in self.all_backup_inputs[keys_one_feature_engineering][wedget][method][key_setting]["wedget"].objectName():
                                        self.all_backup_inputs[keys_one_feature_engineering][wedget][method][key_setting]["wedget"].setValue(float(loaded_text))
                                    elif "spinBox" in self.all_backup_inputs[keys_one_feature_engineering][wedget][method][key_setting]["wedget"].objectName():
                                        self.all_backup_inputs[keys_one_feature_engineering][wedget][method][key_setting]["wedget"].setValue(int(loaded_text))
                                    elif "comboBox" in self.all_backup_inputs[keys_one_feature_engineering][wedget][method][key_setting]["wedget"].objectName():
                                        self.all_backup_inputs[keys_one_feature_engineering][wedget][method][key_setting]["wedget"].setCurrentText(loaded_text)
                                    
                                # Switch stacked wedget
                                switch_dict[keys_one_feature_engineering](True, method)

    def save_configuration(self):
        """Save configuration
        """

        # Get current inputs before saving machine_learning parameters
        self.get_current_inputs()
    
        # Delete wedgets object from self.machine_learning dict
        for feature_engineering_name in list(self.machine_learning.keys()):
            for method_name in list(self.machine_learning[feature_engineering_name].keys()):
                for setting in self.machine_learning[feature_engineering_name][method_name]:
                    for content in list(self.machine_learning[feature_engineering_name][method_name][setting].keys()):
                        if "wedget" in list(self.machine_learning[feature_engineering_name][method_name][setting].keys()):
                            self.machine_learning[feature_engineering_name][method_name][setting].pop("wedget")
        
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

    def switche_stacked_wedge_for_classification(self, signal_bool, method=None):
        if self.sender():
            if not method:
                self.stackedWidget_setting.setCurrentIndex(self.classification_stackedwedge_dict[self.sender().text()])
            else:
                self.stackedWidget_setting.setCurrentIndex(self.classification_stackedwedge_dict[method])
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
