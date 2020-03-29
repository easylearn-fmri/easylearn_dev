"""The GUI of the data loading module of easylearn
@author: lichao19870617@gmail.com
MIT License
"""
import sys
import os
import json
from PyQt5.QtWidgets import QApplication,QMainWindow, QFileDialog
from PyQt5.QtWidgets import *
from PyQt5 import *
from PyQt5.QtGui import QIcon
import sys
from PyQt5.QtWidgets import QApplication,QWidget,QVBoxLayout,QListView,QMessageBox
from PyQt5.QtCore import QStringListModel

from easylearn_data_loading_gui import Ui_MainWindow


class EasylearnDataLoadingRun(QMainWindow, Ui_MainWindow):
    def __init__(self, working_directory=None):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        # Set working_directory
        self.working_directory = working_directory
        if self.working_directory:
            os.chdir(self.working_directory)
        # set appearance
        self.set_run_appearance()

        # initiating
        self.data_loading = {}
        self.selected_group = None
        self.selected_modality = None
        self.selected_file = None
        self.configuration_file = ""
        self.configuration = ""

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

        self.setWindowTitle('Data Loading')
        self.setWindowIcon(QIcon('../logo/easylearn.jpg')) 
    
    def set_run_appearance(self):
        """Set dart style appearance
        """
        qss_string_all = """
        QPushButton{color: rgb(200,200,200); border: 2px solid rgb(100,100,100); border-radius:0}
        QPushButton:hover {background-color: black; color: white; font-size:20px; font-weight: bold}
        #MainWindow{background-color: rgb(50, 50, 50)}    
        QListView{background-color:rgb(200,200,200); color:black; font-size:15px; border: 0px solid rgb(100,100,100); border-radius:0} 
        QListView::item:selected {font-weight:bold; font-size:15; color:black; border: 1px solid black}
        QListView::item:selected:!active {background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,stop: 0 #ABAFE5, stop: 1 #8588B2)}
        QListView::item:selected:active {background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,stop: 0 #E0FFFF, stop: 1 #888dd9)}
        QLineEdit{background-color:rgb(200,200,200); color:black; font-size:15px; border: 0px solid rgb(100,100,100); border-radius:0}
        """
        self.setStyleSheet(qss_string_all)
        self.label_group.setStyleSheet("color:white; font-weight: bold")   
        self.label_modalities.setStyleSheet("color:white; font-weight: bold")   
        self.label_file.setStyleSheet("color:white; font-weight: bold")  
        self.label_mask.setStyleSheet("color:white; font-weight: bold")   
        self.label_target.setStyleSheet("color:white; font-weight: bold")   
        self.label_covariance.setStyleSheet("color:white; font-weight: bold")  
    
    def set_quite_appearance(self):
        """This function is set appearance when quit program.

        This function make sure quit message can be see clearly.
        """
        qss_string_message = """
        QPushButton{color: black; border: 2px solid rgb(100,100,100); border-radius:0}
        QPushButton:hover {background-color: black; color: white; font-size:20px; font-weight: bold}
        #MainWindow{background-color: rgb(50, 50, 50)}    
        QListView{background-color:rgb(200,200,200); color:black; font-size:15px; border: 2px solid rgb(100,100,100); border-radius:0}                   
        """  
        # """
        self.setStyleSheet(qss_string_message)
        self.label_group.setStyleSheet("color:white; font-weight: bold")   
        self.label_modalities.setStyleSheet("color:white; font-weight: bold")   
        self.label_file.setStyleSheet("color:white; font-weight: bold") 

    def load_configuration(self):
        """Load configuration, and display groups
        """
        self.configuration_file, filetype = QFileDialog.getOpenFileName(self,  
                                "Select configuration file",  
                                os.getcwd(), "Text Files (*.json);;All Files (*);;") 

        # Read configuration_file if already selected
        if self.configuration_file != "": 
        # TODO: 解决中文编码的问题 
            with open(self.configuration_file, 'r', encoding='utf-8') as config:
                self.configuration = config.read()
            # Check the configuration is valid JSON, then transform the configuration to dict
            # If the configuration is not valid JSON, then give configuration and configuration_file to ""
            try:
                self.configuration = json.loads(self.configuration)
                # If already exists data_loading configuration
                if self.data_loading != {}:
                    self.set_quite_appearance()
                    reply = QMessageBox.question(self, "Data loading configuration already exists", 
                                                "The data_loading configuration is already exists, do you want to rewrite it with the  loaded configuration?",
                                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                    self.set_run_appearance()
                    if reply == QMessageBox.Yes:  
                        self.data_loading = self.configuration["data_loading"]
                else:
                    self.data_loading = self.configuration["data_loading"]

                self.display_groups()
                self.display_modalities()
                self.display_files()
            except json.decoder.JSONDecodeError:
                self.set_quite_appearance()
                QMessageBox.warning( self, 'Warning', f'{self.configuration_file} is not valid JSON')
                self.configuration_file = ""
                self.set_run_appearance()
        else:
            self.set_quite_appearance()
            QMessageBox.warning( self, 'Warning', 'Configuration file was not selected')
            self.set_run_appearance()

    def save_configuration(self):
        """Save configuration
        """
        if self.configuration_file != "":
            self.configuration["data_loading"] = self.data_loading
            with open(self.configuration_file, 'w', encoding="utf-8") as config:    
                # Set ensure_ascii=False to save Chinese correctly.
                config.write(json.dumps(self.configuration, ensure_ascii=False))
        else:
            self.set_quite_appearance()
            QMessageBox.warning( self, 'Warning', 'Please choose a configuration file first (press button at top left corner)!')
            self.set_run_appearance()

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
        self.set_quite_appearance()
        self.group_name, ok = QInputDialog.getText(self, "Add group", "Please name the group:", QLineEdit.Normal, "group_")  
        self.set_run_appearance()
        if self.group_name not in self.data_loading.keys():
            self.data_loading[self.group_name] = {}
        self.display_groups()

    def identify_selected_group(self, index):
        """Identify the selected file in the list_view_groups and display_files the files
        """
        self.selected_group = list(self.data_loading.keys())[index.row()]
        self.display_modalities()
        self.display_files()
   
    def remove_selected_group(self):
        """
        This function is used to remove selected group
        
        If exist selected self.selected_group and self.selected_group is in list(self.data_loading.keys),
        then remove.
        """

        if bool(self.selected_group):
            self.set_quite_appearance()
            reply = QMessageBox.question(self, "QListView", "Remove this group: " + self.selected_group + "?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            self.set_run_appearance()
            if reply == QMessageBox.Yes:  
                # Remove selected group
                del self.data_loading[self.selected_group]
                self.selected_group = None
                self.display_groups()
                self.display_modalities()
                self.display_files()
        else:
            self.set_quite_appearance()
            QMessageBox.warning( self, 'Warning', 'No group selected!')
            self.set_run_appearance()

    def clear_all_group(self):
        """
        Remove all selections
        """
        self.data_loading = {}
        self.selected_group = False
        self.display_groups() 
        self.display_modalities()
        self.display_files()
    #%% -----------------------------------------------------------------

    def add_modality(self):
        """Add a modality for a selected group
        """
        if self.selected_group:
            self.set_quite_appearance()
            mod_name, ok = QInputDialog.getText(self, "Add modality", "Please name the modality:", QLineEdit.Normal, "modality_")
            self.set_run_appearance()
            if not (mod_name in self.data_loading[self.selected_group].keys()):  # avoid clear exist modalites
                self.data_loading[self.selected_group][mod_name] = {"file":[], "mask": [], "target":[], "covariances": []}  #  must copy the dict, otherwise all modalities are the same.
            self.display_modalities()
        else:
            self.set_quite_appearance()
            QMessageBox.warning( self, 'Warning', 'Please choose group first!')
            self.set_run_appearance()

    def identify_selected_modality(self, index):
        """Identify the selected modality
        """
        self.selected_modality = list(self.data_loading[self.selected_group].keys())[index.row()]
        self.display_files()

    def remove_selected_modality(self):
        """This function is used to remove selected modality
        
        If selected self.selected_group and self.selected_modality, and
        the selected modality is in the selected group (some cases, you may selected a group that not contains the selected modality
        namely the selected modality is selected from other groups).
        """    
        if (bool(self.selected_modality) & bool(self.selected_group) & 
            (self.selected_modality in list(self.data_loading[self.selected_group]))):
            self.set_quite_appearance()
            message = "Remove this modality: " + self.selected_modality + " of " + self.selected_group + "?"
            self.set_run_appearance()
            reply = QMessageBox.question(self, "QListView", message,
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:     
                # Remove selected modality for selected group
                del self.data_loading[self.selected_group][self.selected_modality]
                self.selected_modality = None
                self.display_modalities()
                self.display_files()
        else:
            self.set_quite_appearance()
            QMessageBox.warning( self, 'Warning', 'No group or modality selected!')
            self.set_run_appearance()

    def clear_all_modality(self):
        """
        Remove all modalities for selected group
        """
        if self.selected_group:
            self.data_loading[self.selected_group] = {}
            self.selected_modality = None
            self.display_modalities()  
            self.display_files()
        else:
            self.set_quite_appearance()
            QMessageBox.warning( self, 'Warning', 'No group selected!')
            self.set_run_appearance()
    #%% -----------------------------------------------------------------

    def add_files(self):
        """Add files for selected modality of selected group.
        """
        if (bool(self.selected_group) & bool(self.selected_modality)):
            self.loaded_files, filetype = QFileDialog.getOpenFileNames(self,  
                                    "Select files",  os.getcwd(), 
                                    "Nifti Files (*.nii);;Matlab Files (*.mat);;Excel Files (*.xlsx);;Excel Files (*.xls);;Text Files (*.txt);;All Files (*)"
        )
            
            self.data_loading[self.selected_group][self.selected_modality]["file"].extend(self.loaded_files)
            self.display_files()
        else:
            self.set_quite_appearance()
            QMessageBox.warning( self, 'Warning', 'Please select group and modality first!')
            self.set_run_appearance()       
    
    def identify_selected_file(self, index):
        """Identify the selected file in the list_view_files
        """
        self.selected_file = self.data_loading[self.selected_group][self.selected_modality]["file"][index.row()]
    
    def remove_selected_file(self):
        """
        This function is used to remove selected file for selected modality of selected group
        """
        if (bool(self.selected_group) & bool(self.selected_modality) & bool(self.selected_file)):  
            self.set_quite_appearance()
            reply = QMessageBox.question(self, "QListView", "Remove this file: " + self.selected_file + "?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            self.set_run_appearance()
            if reply == QMessageBox.Yes:     
                self.data_loading[self.selected_group][self.selected_modality]["file"] = list(set(self.data_loading[self.selected_group][self.selected_modality]["file"]) - set([self.selected_file]))
                self.selected_file = None
                self.display_files()
        else:
            self.set_quite_appearance()
            QMessageBox.warning( self, 'Warning', 'No group or modality or file selected!')
            self.set_run_appearance() 

    def clear_all_file(self):
        """
        Remove all files of the selected modality of selected group
        """  
        print(self.selected_group)
        print(self.selected_modality)
        if (bool(self.selected_group) & bool(self.selected_modality)):  
            self.data_loading[self.selected_group][self.selected_modality]["file"] = []
            self.selected_file = None
            self.display_files() 
    #%% -----------------------------------------------------------------
    
    def display_groups(self):
        """
        Display groups' name in the list view
        """
        self.slm_group.setStringList(self.data_loading.keys())  
        self.listView_groups.setModel(self.slm_group)
    
    def display_modalities(self):
        """
        Display modalities' name in the list view
        """
        if bool(self.selected_group):
            self.slm_modality.setStringList(self.data_loading[self.selected_group].keys())  
            self.listView_modalities.setModel(self.slm_modality)
        else:
            self.slm_modality.setStringList([])  
            self.listView_modalities.setModel(self.slm_modality)
        
    def display_files(self):
        """
        Display files of the selected modality of the selected group
        """
        if (bool(self.selected_group) & bool(self.selected_modality)):
            if ((self.selected_modality in list(self.data_loading[self.selected_group].keys()))):
                # Add numpy  to file name
                file_to_display = [str(i+1) + ":  "+ file_name for (i, file_name) in enumerate(self.data_loading[self.selected_group][self.selected_modality]["file"])]
                self.slm_file.setStringList(file_to_display)  
                self.listView_files.setModel(self.slm_file)
            else:
                self.slm_file.setStringList([])  
                self.listView_files.setModel(self.slm_file)
        else:
            self.slm_file.setStringList([])  
            self.listView_files.setModel(self.slm_file)
    


    def closeEvent(self, event):
        """This function is called when exit icon of the window is clicked.

        This function make sure the program quit safely.
        """
        # Set qss to make sure the QMessageBox can be seen
        self.set_quite_appearance()
        reply = QMessageBox.question(self, 'Quit',"Are you sure to quit?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
            self.set_run_appearance()  # Make appearance back


if __name__=='__main__':
    app=QApplication(sys.argv)
    md=EasylearnDataLoadingRun()
    md.show()
    sys.exit(app.exec_())
