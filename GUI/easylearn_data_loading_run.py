# -*- coding: utf-8 -*-
"""The GUI of the data loading module of easylearn

Created on 2020/04
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
from PyQt5.QtWidgets import QApplication,QMainWindow, QFileDialog
from PyQt5.QtWidgets import *
from PyQt5 import *
from PyQt5.QtGui import QIcon
import sys
from PyQt5.QtWidgets import QApplication,QWidget,QVBoxLayout,QListView,QMessageBox
from PyQt5.QtCore import*
from eslearn.stylesheets.PyQt5_stylesheets import pyqt5_loader

import eslearn
from eslearn.GUI.easylearn_data_loading_gui import Ui_MainWindow


class EasylearnDataLoadingRun(QMainWindow, Ui_MainWindow):
    def __init__(self, working_directory=None):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.root_dir = os.path.dirname(eslearn.__file__)

        # Set working_directory and debug
        self.working_directory = working_directory
        self.configuration_file = ""
        if self.working_directory:
            cgitb.enable(format="text", display=1, logdir=os.path.join(self.working_directory, "log_data_loading"))
        else:
            cgitb.enable(display=1, logdir=None)   
        
        # initiating
        self.data_loading = {}
        self.selected_group = None
        self.selected_modality = None
        self.selected_file = None
        self.loaded_targets_and_covariates = None
        self.loaded_mask = None
        self.loaded_files = None
        self.group_keys_exclude_modality = ["targets","covariates"]


        # initialize list_view for groups, modalities and files
        self.slm_group = QStringListModel()
        self.slm_modality = QStringListModel()
        self.slm_file = QStringListModel()

        # connections
        self.actionChoose_configuration_file.triggered.connect(self.load_configuration)
        self.actionSave_configuration.triggered.connect(self.save_configuration)

        self.listView_groups.clicked.connect(self.identify_selected_group)
        self.pushButton_addgroups.clicked.connect(self.add_group)
        self.listView_groups.doubleClicked.connect(self.remove_selected_group)
        self.pushButton_removegroups.clicked.connect(self.remove_selected_group)
        self.pushButton_cleargroups.clicked.connect(self.clear_all_group)

        self.listView_modalities.clicked.connect(self.identify_selected_modality)
        self.pushButton_addmodalities.clicked.connect(self.add_modality)
        self.listView_modalities.doubleClicked.connect(self.remove_selected_modality)
        self.pushButton_removemodalites.clicked.connect(self.remove_selected_modality)
        self.pushButton_clearmodalities.clicked.connect(self.clear_all_modality)

        self.listView_files.clicked.connect(self.identify_selected_file)
        self.pushButton_addfiles.clicked.connect(self.add_files)
        self.listView_files.doubleClicked.connect(self.remove_selected_file)
        self.pushButton_removefiles.clicked.connect(self.remove_selected_file)
        self.pushButton_clearfiles.clicked.connect(self.clear_all_file)

        # mask_target_covariates
        self.target_covariate_mask_dict = {"Select mask": [self.lineEdit_mask, "mask"], "Clear mask": [self.lineEdit_mask, "mask"], 
                            "Select targets": [self.lineEdit_target, "targets"], "Clear targets": [self.lineEdit_target, "targets"], 
                            "Select covariates": [self.lineEdit_covariates, "covariates"], "Clear covariates": [self.lineEdit_covariates, "covariates"]}
        self.pushButton_selectMask.clicked.connect(self.input_mask)
        self.pushButton_selectTarget.clicked.connect(self.input_target_covariate)
        self.pushButton_selectCovariance.clicked.connect(self.input_target_covariate)
        self.pushButton_clearMask.clicked.connect(self.clear_mask_target_covariates)
        self.pushButton_clearTarget.clicked.connect(self.clear_mask_target_covariates)
        self.pushButton_clearCovriance.clicked.connect(self.clear_mask_target_covariates)
        self.pushButton_mask.clicked.connect(self.confirm_box_mask)
        self.pushButton_target.clicked.connect(self.confirm_box_target)
        self.pushButton_covariate.clicked.connect(self.confirm_box_covariates)

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
        self.setWindowTitle('Data Loading')
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

    def load_configuration(self):
        """Load configuration, and display groups
        """
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
        # TODO: considering the Chinese word problem
            with open(self.configuration_file, 'r', encoding='utf-8') as config:
                self.configuration = config.read()
            # Check the configuration is valid JSON, then transform the configuration to dict
            # If the configuration is not valid JSON, then give configuration and configuration_file to ""
            try:
                self.configuration = json.loads(self.configuration)
                # If already exists self.data_loading
                if (self.data_loading != {}):
                    # If the loaded self.configuration["data_loading"] is not empty
                    # Then ask if rewrite self.data_loading with self.configuration["data_loading"]
                    if (list(self.configuration["data_loading"].keys()) != []):
            
                        reply = QMessageBox.question(self, "Data loading configuration already exists", 
                                                    "The data_loading configuration is already exists, do you want to rewrite it with the  loaded configuration?",
                                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
           
                        if reply == QMessageBox.Yes:  
                            self.data_loading = self.configuration["data_loading"]
                            # Because rewrite self.data_loading, we need to re-initialize the follows.
                            self.selected_group = None
                            self.selected_modality = None
                            self.selected_file = None
                    # If the loaded self.configuration["data_loading"] is empty
                     # Then assign self.configuration["data_loading"] with self.data_loading
                    elif (list(self.configuration["data_loading"].keys()) == []):
                        self.configuration["data_loading"] = self.data_loading
                else:
                    self.data_loading = self.configuration["data_loading"]
                self.display_groups()
                self.display_modality()
                self.display_target_covariate()
                self.display_files()

            except json.decoder.JSONDecodeError:
                QMessageBox.warning( self, 'Warning', f'{self.configuration_file} is not valid JSON')
                self.configuration_file = ""
   
        else:
            QMessageBox.warning( self, 'Warning', 'Configuration file was not selected')

    def save_configuration(self):
        """Save configuration
        """

        if self.configuration_file != "":
            self.configuration["data_loading"] = self.data_loading
            with open(self.configuration_file, 'w', encoding="utf-8") as config:    
                # Set ensure_ascii=False to save Chinese correctly.
                config.write(json.dumps(self.configuration, ensure_ascii=False, indent=4))
        else:

            QMessageBox.warning( self, 'Warning', 'Please choose a configuration file first (press button at top left corner)!')

    def select_workingdir(self):
        """
        This function is used to select the working directory
        """

        #  If has selected working directory previously, then I set it as initial working directory.
        try:
            self.directory
        except AttributeError:
            self.directory = 0

        if not self.directory:
            self.directory = QFileDialog.getExistingDirectory(self, "Select a directory", os.getcwd()) 
        else:
            self.directory = QFileDialog.getExistingDirectory(self, "Select a directory", self.directory) 

        self.lineEdit_workingdir.setText(self.directory)

        try:
            self.selected_file = os.listdir(self.directory)
            self.current_list_file = self.selected_file  # Every time selecting directory, the current list will be initiated once.
            self.slm.setStringList(self.selected_file)  
            self.listView_groups.setModel(self.slm) 
        except FileNotFoundError:
            self.lineEdit_workingdir.setText("You need to choose a working directory")
    #%% -----------------------------------------------------------------

    def add_group(self):
        """Add a group
        """

        group_name, ok = QInputDialog.getText(self, "Add group", "Please name the group:", QLineEdit.Normal, "group_") 
        if (group_name != "") and (group_name not in self.data_loading.keys()):
            self.data_loading[group_name] = {"modalities":{},"targets":[], "covariates":""}
        self.display_groups()
        self.selected_group = None

    def identify_selected_group(self, index):
        """Identify the selected file in the list_view_groups and display_files the files
        """

        self.selected_group = list(self.data_loading.keys())[index.row()]
        self.display_modality()
        self.display_target_covariate()
        self.display_files()
        self.display_mask()

        self.selected_modality = None
        self.selected_file = None
   
    def remove_selected_group(self):
        """
        This function is used to remove selected group
        
        If exist selected self.selected_group and self.selected_group is in list(self.data_loading.keys),
        then remove.
        """

        if bool(self.selected_group):
            reply = QMessageBox.question(self, "QListView", "Remove this group: " + self.selected_group + "?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:  
                # Remove selected group
                del self.data_loading[self.selected_group]
                self.selected_group = None
                self.display_groups()
                self.display_modality()
                self.display_target_covariate()
                self.display_files()
        else:

            QMessageBox.warning( self, 'Warning', 'No group selected!')

    def clear_all_group(self):
        """
        Remove all selections
        """
        self.data_loading = {}
        self.selected_group = None
        self.display_groups() 
        self.display_modality()
        self.display_target_covariate()
        self.display_files()
    #%% -----------------------------------------------------------------

    def add_modality(self):
        """Add a modality for a selected group
        """

        if self.selected_group:
            mod_name, ok = QInputDialog.getText(self, "Add modality", "Please name the modality:", QLineEdit.Normal, "modality_")
            # Avoid clear exist modalites
            if (mod_name != "") and (mod_name not in self.data_loading[self.selected_group].keys()): 
                self.data_loading[self.selected_group]["modalities"][mod_name] = {"file":[], "mask": ""}  #  Must copy the dict, otherwise all modalities are the same.
            
            self.display_modality()
            self.selected_modality = None
        else:

            QMessageBox.warning( self, 'Warning', 'Please choose group first!')

    def identify_selected_modality(self, index):
        """Identify the selected modality
        """

        current_modality_list = list(self.data_loading[self.selected_group]["modalities"].keys())
        self.selected_modality = current_modality_list[index.row()]
        self.display_files()
        self.display_mask()
        self.selected_file = None

    def remove_selected_modality(self):
        """This function is used to remove selected modality
        
        If selected self.selected_group and self.selected_modality, and
        the selected modality is in the selected group (some cases, you may selected a group that not contains the selected modality
        namely the selected modality is selected from other groups).
        """  

        if (bool(self.selected_modality) & bool(self.selected_group)):
            if (self.selected_modality in list(self.data_loading[self.selected_group]["modalities"])):
                message = "Remove this modality: " + self.selected_modality + " of " + self.selected_group + "?"
                reply = QMessageBox.question(self, "QListView", message,
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.Yes:     
                    # Remove selected modality for selected group
                    del self.data_loading[self.selected_group]["modalities"][self.selected_modality]
                    self.selected_modality = None
                    self.display_modality()
                    self.display_target_covariate()
                    self.display_files()
            else:
                QMessageBox.warning( self, 'Warning', f'{list(self.data_loading[self.selected_group]["modalities"])} has no {self.selected_modality}')
        else:
            QMessageBox.warning( self, 'Warning', 'No group or modality selected!')

    def clear_all_modality(self):
        """
        Remove all modalities for selected group
        """
        if self.selected_group:
            self.data_loading[self.selected_group] = {}
            self.selected_modality = None
            self.display_modality() 
            self.display_target_covariate() 
            self.display_files()
        else:
            QMessageBox.warning( self, 'Warning', 'No group selected!')
    #%% -----------------------------------------------------------------

    def add_files(self):
        """Add files for selected modality of selected group.
        """

        if self.loaded_files:
            old_dir = os.path.dirname(self.loaded_files[0])
        else:
            if self.working_directory:
                old_dir = self.working_directory
            else:
                old_dir = os.getcwd()
                
        if (bool(self.selected_group) & bool(self.selected_modality)):
            self.loaded_files, filetype = QFileDialog.getOpenFileNames(self,  
                                    "Select files",  old_dir, 
                                    "All Files (*);;Nifti Files (*.nii);;Matlab Files (*.mat);;Excel Files (*.xlsx);;Excel Files (*.xls);;Text Files (*.txt)"
            )
            
            self.data_loading[self.selected_group]["modalities"][self.selected_modality]["file"].extend(self.loaded_files)
            self.display_files()
            self.selected_file = None
        else:

            QMessageBox.warning( self, 'Warning', 'Please select group and modality first!')       
    
    def identify_selected_file(self, index):
        """Identify the selected file in the list_view_files
        """
        self.selected_file = self.data_loading[self.selected_group]["modalities"][self.selected_modality]["file"][index.row()]
    
    def remove_selected_file(self):
        """
        This function is used to remove selected file for selected modality of selected group
        """

        if (bool(self.selected_group) & bool(self.selected_modality) & bool(self.selected_file)):  

            reply = QMessageBox.question(self, "QListView", "Remove this file: " + self.selected_file + "?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                old_files = self.data_loading[self.selected_group]["modalities"][self.selected_modality]["file"]     
                self.data_loading[self.selected_group]["modalities"][self.selected_modality]["file"] = list(set(self.data_loading[self.selected_group]["modalities"][self.selected_modality]["file"]) - set([self.selected_file]))
                self.data_loading[self.selected_group]["modalities"][self.selected_modality]["file"].sort(key=old_files.index)
                self.selected_file = None
                self.display_files()
        else:

            QMessageBox.warning( self, 'Warning', 'No group or modality or file selected!') 

    def clear_all_file(self):
        """
        Remove all files of the selected modality of selected group
        """  
        if (bool(self.selected_group) & bool(self.selected_modality)):  
            self.data_loading[self.selected_group]["modalities"][self.selected_modality]["file"] = []
            self.selected_file = None
            self.display_files() 
    #%% -----------------------------------------------------------------
    
    def display_groups(self):
        """Display groups' name in the list view
        """

        self.slm_group.setStringList(self.data_loading.keys())  
        self.listView_groups.setModel(self.slm_group)
    
    def display_modality(self):
        """
        Display modalities' name in the list view
        """

        if bool(self.selected_group):
            # Get current modalities list
            current_modality_list = list(self.data_loading[self.selected_group]["modalities"].keys()) 
            # Display
            self.slm_modality.setStringList(current_modality_list)
            self.listView_modalities.setModel(self.slm_modality)
        else:
            self.slm_modality.setStringList([])  
            self.listView_modalities.setModel(self.slm_modality)
        
    def display_files(self):
        """
        Display files of the selected modality of the selected group
        """

        if (bool(self.selected_group) & bool(self.selected_modality)):
            if ((self.selected_modality in list(self.data_loading[self.selected_group]["modalities"].keys()))):
                # Add numpy to file name
                file_to_display = [str(i+1) + ":  "+ file_name for (i, file_name) in enumerate(self.data_loading[self.selected_group]["modalities"][self.selected_modality]["file"])]
                self.slm_file.setStringList(file_to_display)  
                self.listView_files.setModel(self.slm_file)
            else:
                self.slm_file.setStringList([])  
                self.listView_files.setModel(self.slm_file)
        else:
            self.slm_file.setStringList([])  
            self.listView_files.setModel(self.slm_file)
    
    def input_target_covariate(self):
        """Can input or select file
        """

        # Get previous directory
        if self.loaded_targets_and_covariates:
            old_dir = os.path.dirname(self.loaded_targets_and_covariates[0])
        else:
            if self.working_directory:
                old_dir = self.working_directory
            else:
                old_dir = os.getcwd()

        if bool(self.selected_group):
            self.loaded_targets_and_covariates, filetype = QFileDialog.getOpenFileName(self,  
                                    "Select file",  old_dir, 
                                    "Nifti Files (*.nii);Matlab Files (*.mat);Text File (*.txt);All Files (*)"
            )

            if self.loaded_targets_and_covariates != "":
                targets_or_covariate = self.target_covariate_mask_dict[self.sender().text()][1]
                self.target_covariate_mask_dict[self.sender().text()][0].setText(self.loaded_targets_and_covariates)
                self.data_loading[self.selected_group][targets_or_covariate] = self.loaded_targets_and_covariates
        else:
            QMessageBox.warning( self, 'Warning', 'Please select group and modality first!') 

    def input_mask(self):
        """Can input or select file
        """

        # Get previous directory
        if self.loaded_mask:
            old_dir = os.path.dirname(self.loaded_mask[0])
        elif self.working_directory:
            old_dir = os.path.dirname(self.working_directory)
        else:
            old_dir = os.getcwd()
        
        if (bool(self.selected_group) & bool(self.selected_modality)):
            self.loaded_mask, filetype = QFileDialog.getOpenFileName(self,  
                                    "Select file", old_dir, 
                                    "Nifti Files (*.nii);Matlab Files (*.mat);Text File (*.txt);All Files (*)"
            )

            if self.loaded_mask != "":
                targets_or_covariate = self.target_covariate_mask_dict[self.sender().text()][1]
                self.target_covariate_mask_dict[self.sender().text()][0].setText(self.loaded_mask)
                self.data_loading[self.selected_group]["modalities"][self.selected_modality][targets_or_covariate] = self.loaded_mask
        else:
            QMessageBox.warning( self, 'Warning', 'Please select group and modality first!')   
    
    def display_target_covariate(self):
        if bool(self.selected_group):
            if self.data_loading[self.selected_group]["targets"] != []:
                self.lineEdit_target.setText(self.data_loading[self.selected_group]["targets"])
            else:
                self.lineEdit_target.setText("")

            if self.data_loading[self.selected_group]["covariates"] != []:
                self.lineEdit_covariates.setText(self.data_loading[self.selected_group]["covariates"])
            else:
                self.lineEdit_covariates.setText("")
        else:
            self.lineEdit_target.setText("")
            self.lineEdit_covariates.setText("")

    def display_mask(self):
        if (bool(self.selected_group) & bool(self.selected_modality)):
            if (self.selected_modality in list(self.data_loading[self.selected_group]["modalities"].keys())):
                if self.data_loading[self.selected_group]["modalities"][self.selected_modality]["mask"] != "":
                    self.lineEdit_mask.setText(self.data_loading[self.selected_group]["modalities"][self.selected_modality]["mask"])
                else:
                    self.lineEdit_mask.setText("")
        else:
            self.lineEdit_mask.setText("")

    def confirm_box_target(self, state):
        """When users input mask by their hands, they should click this pushbutton to confirm the inputs.
        """

        if bool(self.selected_group):
                self.data_loading[self.selected_group]["targets"] = self.lineEdit_target.text()
        else:
            QMessageBox.warning( self, 'Warning', 'Please select group and modality first!') 

    def confirm_box_covariates(self, state):
        """When users input mask by their hands, they should click this pushbutton to confirm the inputs.
        """

        if bool(self.selected_group):
                self.data_loading[self.selected_group]["covariates"] = self.lineEdit_covariates.text()
        else:
            QMessageBox.warning( self, 'Warning', 'Please select group and modality first!') 

    def confirm_box_mask(self, state):
        """When users input mask by their hands, they should click this pushbutton to confirm the inputs.
        """

        if (bool(self.selected_group) & bool(self.selected_modality)):
            self.data_loading[self.selected_group]["modalities"][self.selected_modality]["mask"] = self.lineEdit_mask.text()
        else:
            QMessageBox.warning( self, 'Warning', 'Please select group and modality first!') 

    def clear_mask_target_covariates(self):
        self.target_covariate_mask_dict[self.sender().text()][0].setText("")

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


if __name__=='__main__':
    app=QApplication(sys.argv)
    md=EasylearnDataLoadingRun()
    md.show()
    sys.exit(app.exec_())
