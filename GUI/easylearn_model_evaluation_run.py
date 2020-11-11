# -*- coding: utf-8 -*-
"""The GUI of the model_evaluation module of easylearn

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
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from eslearn.stylesheets.PyQt5_stylesheets import pyqt5_loader

import eslearn
from eslearn.GUI.easylearn_model_evaluation_gui import Ui_MainWindow


class EasylearnModelEvaluationRun(QMainWindow, Ui_MainWindow):
    """The GUI of the model_evaluation module of easylearn

    All users' input will save to configuration_file for finally run the whole machine learning pipeline.
    Specificity, the self.model_evaluation configuration will save to the configuration_file that the user created in 
    the main window.
    """

    def __init__(self, working_directory=None):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.root_dir = os.path.dirname(eslearn.__file__)

        # Initialization
        self.working_directory = working_directory
        self.configuration_file = ""
        self.configuration = {}
        self.model_evaluation = {}
        self.all_inputs_fun()

        # Debug
        # Set working_directory
        if self.working_directory:
            cgitb.enable(format="text", display=1, logdir=os.path.join(self.working_directory, "log_model_evaluation"))
        else:
            cgitb.enable(display=1, logdir=None) 

        # Connect configuration functions
        self.actionLoad_configuration.triggered.connect(self.load_configuration)
        self.actionSave_configuration.triggered.connect(self.save_configuration)

        # # Connect to radioButton of model evaluation type: switche to corresponding model evaluation type window
        self.model_evaluation_type_stackedwedge_dict = {
            "KFold()": 0, "StratifiedKFold()": 1, "ShuffleSplit()": 2, "User-defined CV": 3,
        }
      
        # Connect to remove selected datasets 
        # self.listWidget_selected_datasets.doubleClicked.connect(self.remove_selected_datasets)
        # self.listWidget_selected_datasets.customContextMenuRequested.connect(self.remove_selected_datasets)
        # self.listWidget_selected_datasets.itemChanged.connect(self.del_repeated_items)

        # connect to statistical_analysis 
        self.statistical_analysis_method_stackedwedge_dict = {"Binomial test": 0, "Permutation test": 1}
        self.radioButton_binomialtest.clicked.connect(self.statistical_analysis_setting)
        self.radioButton_permutationtest.clicked.connect(self.statistical_analysis_setting)

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
        self.setWindowTitle('Model evaluation')
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

        # # This dictionary is used to keep track of model_evaluation types
        self.model_evaluation_type_dict = {
            0: "KFold()", 1: "StratifiedKFold()",
            2: "ShuffleSplit()", 3: "User-defined CV",
        }

        # Get current selected datasets
        # count = self.listWidget_selected_datasets.count()
        # selected_datasets = []
        # for i in range(count):
        #     selected_datasets.append(self.listWidget_selected_datasets.item(i).text())


        # All available inputs
        self.all_available_inputs = {
            "KFold()": {
                        "n_splits": {"value": self.lineEdit_kfold_n_splits.text(), "wedget": self.lineEdit_kfold_n_splits},
                        "shuffle": {"value": self.comboBox_kfold_shuffle.currentText(), "wedget": self.comboBox_kfold_shuffle},
                        "random_state": {"value": self.spinBox_kfold_randomstate.text(), "wedget": self.spinBox_kfold_randomstate},
            },

            "StratifiedKFold()": {
                                    "n_splits": {"value": self.lineEdit_stratifiedkfold_n_splits.text(), "wedget": self.lineEdit_stratifiedkfold_n_splits},
                                    "shuffle": {"value": self.comboBox_stratifiedkfold_shuffle.currentText(), "wedget": self.comboBox_stratifiedkfold_shuffle},
                                    "random_state": {"value": self.spinBox_stratifiedkfold_randomstate.text(), "wedget": self.spinBox_stratifiedkfold_randomstate},
            },

            "ShuffleSplit()": {
                                "n_splits": {"value": self.lineEdit_randomsplits_n_splits.text(), "wedget": self.lineEdit_randomsplits_n_splits},
                                "train_size": {"value": self.doubleSpinBox_randomsplits_trainsize.text(), "wedget": self.doubleSpinBox_randomsplits_trainsize},
                                "random_state": {"value": self.spinBox_randomsplits_randomstate.text(), "wedget": self.spinBox_randomsplits_randomstate},
            },

            "User-defined CV": {
                               
            },

            self.radioButton_binomialtest: {
                "Binomial test":{}
            },

            self.radioButton_permutationtest: {
                "Permutation test":{
                    "N":{"value":self.spinBox_permutaiontest_n.text(), "wedget":self.spinBox_permutaiontest_n}
                }
            },

        }

    def get_current_inputs(self):
        """Get all current inputs

        Programme will scan the GUI to determine the user's inputs.

        Attrs:
        -----
            self.model_evaluation: dictionary
                all model_evaluation parameters that the user input.
        """

        # Scan the current inputs
        self.all_inputs_fun()
        
        # Get current model evaluation
        self.model_evaluation = {}
        model_evaluation_type = self.model_evaluation_type_dict[self.tabWidget_CV.currentIndex()]
        self.model_evaluation[model_evaluation_type] = self.all_available_inputs[model_evaluation_type]
        
        # Get current statistical analysis
        stat_list = [self.radioButton_permutationtest, self.radioButton_binomialtest]
        for stat in stat_list:
            if stat.isChecked():
                self.model_evaluation["Statistical_analysis"] = self.all_available_inputs[stat]

    def load_configuration(self):
        """Load configuration, and display_loaded_inputs_in_gui configuration in GUI (removed to get_current_inputs method)
        """

        # Get current inputs before load configuration, so we can 
        # compare loaded configuration["model_evaluation"] with the current self.model_evaluation

        # Scan the current GUI first and get current inputs, so that to compare with loaded configuration
        self.get_current_inputs()

        # if self.configuration_file == "":
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
                # self.display_datasets()  # Display datasets in configuration no matter what if rewrite current inputs.
                # If already exists self.model_evaluation
                if (self.model_evaluation != {}):
                    # If the loaded self.configuration["model_evaluation"] is not empty
                    # Then ask if rewrite self.model_evaluation with self.configuration["model_evaluation"]
                    if (list(self.configuration["model_evaluation"].keys()) != []):
                        reply = QMessageBox.question(
                            self, "Data loading configuration already exists", 
                            "The model_evaluation configuration is already exists, do you want to rewrite it with the  loaded configuration?",
                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                        )

                        if reply == QMessageBox.Yes:  
                            self.model_evaluation = self.configuration["model_evaluation"]
                            self.display_loaded_inputs_in_gui()
                    # If the loaded self.configuration["model_evaluation"] is empty
                     # Then assign self.configuration["model_evaluation"] with self.model_evaluation
                    else:
                        self.configuration["model_evaluation"] = self.model_evaluation
                else:
                    self.model_evaluation = self.configuration["model_evaluation"]
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
    
        # Delete wedgets object from self.model_evaluation dict
        # NOTE: This code is only for current configuration structure
        for model_evaluation_name in list(self.model_evaluation.keys()):
            for setting in list(self.model_evaluation[model_evaluation_name].keys()):
                for content in list(self.model_evaluation[model_evaluation_name][setting].keys()):
                    if "wedget" in list(self.model_evaluation[model_evaluation_name][setting].keys()):
                        self.model_evaluation[model_evaluation_name][setting].pop("wedget")
            
            # Get statistical analysis
            if "Statistical_analysis" == model_evaluation_name:
                for setting in list(self.model_evaluation[model_evaluation_name].keys()):
                    for content in list(self.model_evaluation[model_evaluation_name][setting].keys()):
                        if "wedget" in list(self.model_evaluation[model_evaluation_name][setting].keys()):
                            self.model_evaluation[model_evaluation_name][setting].pop("wedget")
                        if "N" in list(self.model_evaluation[model_evaluation_name][setting].keys()):
                            if "wedget" in list(self.model_evaluation[model_evaluation_name][setting][content].keys()):
                                self.model_evaluation[model_evaluation_name][setting][content].pop("wedget")
        
        # If already identified the configuration file, then excude saving logic.      
        if self.configuration_file != "":
            try:
                self.configuration["model_evaluation"] = self.model_evaluation
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

        # Display the cross-validation configuration
        for model_evaluation_type in list(self.all_available_inputs.keys()):
            if model_evaluation_type in self.model_evaluation.keys():
                # Switch to model_evaluation_type tabwedget
                self.tabWidget_CV.setCurrentIndex(self.model_evaluation_type_stackedwedge_dict[model_evaluation_type])
                for setting in list(self.all_available_inputs[model_evaluation_type].keys()):
                    if "wedget" in list(self.all_available_inputs[model_evaluation_type][setting].keys()):
                        loaded_text = self.model_evaluation[model_evaluation_type][setting]["value"]
                        # Identify wedget type, then using different methods to "setText"
                        if "lineEdit" in self.all_available_inputs[model_evaluation_type][setting]["wedget"].objectName():
                            self.all_available_inputs[model_evaluation_type][setting]["wedget"].setText(loaded_text)
                        elif "doubleSpinBox" in self.all_available_inputs[model_evaluation_type][setting]["wedget"].objectName():
                            self.all_available_inputs[model_evaluation_type][setting]["wedget"].setValue(float(loaded_text))
                        elif "spinBox" in self.all_available_inputs[model_evaluation_type][setting]["wedget"].objectName():
                            self.all_available_inputs[model_evaluation_type][setting]["wedget"].setValue(int(loaded_text))
                        elif "comboBox" in self.all_available_inputs[model_evaluation_type][setting]["wedget"].objectName():
                            self.all_available_inputs[model_evaluation_type][setting]["wedget"].setCurrentText(loaded_text)
                        elif "listWidget" in self.all_available_inputs[model_evaluation_type][setting]["wedget"].objectName():
                            self.all_available_inputs[model_evaluation_type][setting]["wedget"].clear()  # To avoid repeated items
                            self.all_available_inputs[model_evaluation_type][setting]["wedget"].addItems(loaded_text)
                        else:
                            # TODO: EXTENSION
                            print("Input wedget is not support now!\n")
                
            if not isinstance(model_evaluation_type, str):
                if model_evaluation_type.text() in self.model_evaluation.get("Statistical_analysis", {}).keys():
                    # Set checked
                    model_evaluation_type.setChecked(True)
                    self.statistical_analysis_setting(True, model_evaluation_type.text())
                    # Switch to stat radiobutton
                    for setting in list(self.all_available_inputs[model_evaluation_type].keys()):
                        for value_wedget in list(self.all_available_inputs[model_evaluation_type][setting].keys()):
                            if "wedget" in list(self.all_available_inputs[model_evaluation_type][setting][value_wedget].keys()):
                                loaded_text = self.model_evaluation["Statistical_analysis"][setting][value_wedget]["value"]
                                # Identify wedget type, then using different methods to "setText"
                                if "lineEdit" in self.all_available_inputs[model_evaluation_type][setting][value_wedget]["wedget"].objectName():
                                    self.all_available_inputs[model_evaluation_type][setting][value_wedget]["wedget"].setText(loaded_text)
                                elif "doubleSpinBox" in self.all_available_inputs[model_evaluation_type][setting][value_wedget]["wedget"].objectName():
                                    self.all_available_inputs[model_evaluation_type][setting][value_wedget]["wedget"].setValue(float(loaded_text))
                                elif "spinBox" in self.all_available_inputs[model_evaluation_type][setting][value_wedget]["wedget"].objectName():
                                    self.all_available_inputs[model_evaluation_type][setting][value_wedget]["wedget"].setValue(int(loaded_text))
                                elif "comboBox" in self.all_available_inputs[model_evaluation_type][setting][value_wedget]["wedget"].objectName():
                                    self.all_available_inputs[model_evaluation_type][setting][value_wedget]["wedget"].setCurrentText(loaded_text)
                                elif "listWidget" in self.all_available_inputs[model_evaluation_type][setting][value_wedget]["wedget"].objectName():
                                    self.all_available_inputs[model_evaluation_type][setting][value_wedget]["wedget"].clear()  # To avoid repeated items
                                    self.all_available_inputs[model_evaluation_type][setting][value_wedget]["wedget"].addItems(loaded_text)
                                else:
                                    # TODO: EXTENSION
                                    print("Input wedget is not support now!\n")

    def statistical_analysis_setting(self, signal_bool, statistical_analysis_method=None):
        """ Switch to corresponding statistical_analysis_setting wedget
        """

        if self.sender():
            if not statistical_analysis_method:
                self.stackedWidget_statisticalanalysissetting.setCurrentIndex(self.statistical_analysis_method_stackedwedge_dict[self.sender().text()])
            else:
                self.stackedWidget_statisticalanalysissetting.setCurrentIndex(self.statistical_analysis_method_stackedwedge_dict[statistical_analysis_method])
        else:
            self.stackedWidget_statisticalanalysissetting.setCurrentIndex(-1)

    def display_datasets(self):
        """Display the datasets"""

        if self.configuration["data_loading"]:
            self.listWidget_candidate_datasets.clear()
            for candidate_dataset_group in self.configuration["data_loading"]:
                for candidate_dataset_modality in self.configuration["data_loading"][candidate_dataset_group]:
                    display_datasets = candidate_dataset_group + ":" + candidate_dataset_modality
                    self.listWidget_candidate_datasets.addItem(display_datasets)
    
    def del_repeated_items(self):
        """
        Delete repeated items in selected_datasets
        """

        nitem = self.listWidget_selected_datasets.count()
        selected_datasets = [self.listWidget_selected_datasets.item(i).text() for i in range(nitem)]
        selected_datasets_new = list(set(selected_datasets))  
        selected_datasets_new.sort(key=selected_datasets.index)

        # Delete old selected datasets
        self.listWidget_selected_datasets.clear()
        
        # Update none-repeated datasets
        self.listWidget_selected_datasets.addItems(selected_datasets_new)


    def remove_selected_datasets(self):
        """
        This function is used to remove selected datasets
        
        If exist selected self.selected_datasets and self.selected_datasets is in list(self.data_loading.keys),
        then remove.
        """

        reply = QMessageBox.question(self, "Delete selected datasets", "Remove this datasets: " + self.listWidget_selected_datasets.currentItem().text() + "?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:  
            # Remove selected datasets
                nitem = self.listWidget_selected_datasets.count()
                selected_datasets = [self.listWidget_selected_datasets.item(i).text() for i in range(nitem)]
                button = self.sender()
                row = self.listWidget_selected_datasets.indexAt(button.pos()).row()
                print(row)
                # delete selected item
                if row != -1:
                    self.listWidget_selected_datasets.takeItem(row)  
                else:          
                    self.listWidget_selected_datasets.takeItem(0)  

    def closeEvent(self, event):
        """This function is called when exit icon of the window is clicked.

        This function make sure the program quit safely.
        """

        reply = QMessageBox.question(self, 'Quit',"Are you sure to quit?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore() 


if __name__ == "__main__":
    app=QApplication(sys.argv)
    md=EasylearnModelEvaluationRun()
    md.show()
    sys.exit(app.exec_())
