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
    def __init__(self):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        # set appearance
        self.set_run_appearance()

        # initiating
        self.configuration_file = ""
        self.configuration = ""
        self.group_count = 0
        self.modality_count = 0

        # initialize list_view for groups, modalities and files
        self.slm_group = QStringListModel()
        self.selected_groups = []
        self.current_list_group = []  
        self.slm_modality = QStringListModel()
        self.selected_modalities = []
        self.current_list_modality = [] 
        self.slm_file = QStringListModel()
        self.selected_files = []
        self.current_list_file = {}

        # connections
        self.actionChoose_configuration_file.triggered.connect(self.load_configuration)
        self.actionSave_configuration.triggered.connect(self.save_configuration)

        self.listView_groups.clicked.connect(self.identify_selected_group)
        self.pushButton_addgroups.clicked.connect(self.add_groups)
        self.listView_groups.doubleClicked.connect(self.remove_selected_group)
        self.pushButton_removegroups.clicked.connect(self.remove_selected_group)
        self.pushButton_cleargroups.clicked.connect(self.clear_all_group)

        self.listView_modalities.clicked.connect(self.identify_selected_modality)
        self.pushButton_addmodalities.clicked.connect(self.add_modalities)
        self.listView_modalities.doubleClicked.connect(self.remove_selected_modality)
        self.pushButton_removemodalites.clicked.connect(self.remove_selected_modality)
        self.pushButton_clearmodalities.clicked.connect(self.clear_all_modality)

        self.listView_files.clicked.connect(self.identify_selected_file)
        self.pushButton_addfiles.clicked.connect(self.add_files)
        self.listView_files.doubleClicked.connect(self.remove_selected_file)
        self.pushButton_removefiles.clicked.connect(self.remove_selected_file)
        self.pushButton_clearfiles.clicked.connect(self.clear_all_file)

        self.setWindowTitle('Data Loading')
        self.setWindowIcon(QIcon('./easylearn.jpg')) 
    
    def set_run_appearance(self):
        """Set dart style appearance
        """
        qss_string_all = """
        QPushButton{color: rgb(200,200,200); border: 2px solid rgb(100,100,100); border-radius:9}
        QPushButton:hover {background-color: black; color: white; font-size:20px; font-weight: bold}
        #MainWindow{background-color: rgb(50, 50, 50)}    
        QListView{background-color:rgb(200,200,200); color:black; font-size:15px; border: 2px solid rgb(100,100,100); border-radius:15} 
        QListView::item:selected {font-weight:bold; background-color:white; color:black}   
        """
        

        self.setStyleSheet(qss_string_all)
        self.label_group.setStyleSheet("color:white; font-weight: bold")   
        self.label_modalities.setStyleSheet("color:white; font-weight: bold")   
        self.label_file.setStyleSheet("color:white; font-weight: bold")    
    
    def set_quite_appearance(self):
        """This function is set appearance when quit program.

        This function make sure quit message can be see clearly.
        """
        qss_string_message = """
        QPushButton{color: black; border: 2px solid rgb(100,100,100); border-radius:9}
        QPushButton:hover {background-color: black; color: white; font-size:20px; font-weight: bold}
        #MainWindow{background-color: rgb(50, 50, 50)}    
        QListView{background-color:rgb(200,200,200); color:black; font-size:15px; border: 2px solid rgb(100,100,100); border-radius:15}                   
        """  
        # """
        self.setStyleSheet(qss_string_message)
        self.label_group.setStyleSheet("color:white; font-weight: bold")   
        self.label_modalities.setStyleSheet("color:white; font-weight: bold")   
        self.label_file.setStyleSheet("color:white; font-weight: bold") 

    def load_configuration(self):
        """Load configuration
        """
        if self.configuration_file == "":
            self.configuration_file, filetype = QFileDialog.getOpenFileName(self,  
                                    "Select configuration file",  
                                    os.getcwd(), "All Files (*);;Text Files (*.txt)") 
        else:
            self.set_quite_appearance()
            QMessageBox.warning( self, 'Warning', f"Configuration was given!: {self.configuration_file}")
            self.set_run_appearance()

        # Read configuration
        if self.configuration_file != "": 
        # TODO: 解决中文编码的问题 
            with open(self.configuration_file, 'r') as config:
                self.configuration = config.read()

            print(self.configuration)

            # Check the configuration is valid JSON, then transform the configuration to dict
            # If the configuration is not valid JSON, then give configuration and configuration_file to ""
            try:
                self.configuration = json.loads(self.configuration)
            except json.decoder.JSONDecodeError:
                self.configuration_file = ""
                self.configuration = ""
                self.set_quite_appearance()
                QMessageBox.warning( self, 'Warning', 'Configuration in configuration file is not valid JSON')
                self.set_run_appearance()
        else:
            self.set_quite_appearance()
            QMessageBox.warning( self, 'Warning', 'Configuration file was not selected')
            self.set_run_appearance()

    def save_configuration(self):
        """Save configuration
        """
        if self.configuration != "":
            self.configuration["data_loading"] = self.current_list_file
            self.configuration = json.dumps(self.configuration)
            with open(self.configuration_file, 'w') as config:
                config.write(self.configuration)
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
            self.selected_files = os.listdir(self.directory)
            self.current_list_file = self.selected_files  # Every time selecting directory, the current list will be initiated once.
            self.slm.setStringList(self.selected_files)  
            self.listView_groups.setModel(self.slm) 
        except FileNotFoundError:
            self.lineEdit_workingdir.setText("You need to choose a working directory")

    def add_groups(self):
        """Add a group
        """
        
        self.group_name = "group" + str(self.group_count)
        self.group_name, ok = QInputDialog.getText(self, "Initialize configuration", "Please name the configuration file:", QLineEdit.Normal, self.group_name)
        self.current_list_group.append(self.group_name)
        self.slm_group.setStringList(self.current_list_group)  
        self.listView_groups.setModel(self.slm_group)
        self.set_run_appearance()
        self.group_count += 1

    
    def add_modalities(self):
        """Add a group
        """
        self.modality_name = "modality" + str(self.modality_count)
        self.modality_name, ok = QInputDialog.getText(self, "Initialize configuration", "Please name the configuration file:", QLineEdit.Normal, self.modality_name)
        self.current_list_modality.append(self.modality_name)
        self.slm_modality.setStringList(self.current_list_modality)  
        self.listView_modalities.setModel(self.slm_modality)
        self.modality_count += 1

    def add_files(self):
        """Add files for a modality of a group.
        """
        # Try if selected self.selected_group and self.selected_modality
        try:
            self.selected_group
            self.selected_modality
        except AttributeError:
            is_selected_group_and_modality = False
        else:
            is_selected_group_and_modality = True

        if is_selected_group_and_modality:
            self.selected_files, filetype = QFileDialog.getOpenFileNames(self,  
                                    "Select files",  os.getcwd(), 
                                    "All Files (*);;PDF Files (*.pdf);;Text Files (*.txt)")
            
            # If self.current_list_file do not has the self.selected_group key, then create a empty item (dict) for the key.
            # So that the self.current_list_file[self.selected_group] can be add (item) modalitie.
            if (self.selected_group not in list(self.current_list_file.keys())):
                self.current_list_file[self.selected_group] = {}

            # If self.current_list_file[self.selected_group] do not has the self.selected_modality key, then create a empty item (list) for the key.
            # So that the self.current_list_file[self.selected_group][self.selected_modality] can be used to append.
            if (self.selected_modality not in list(self.current_list_file[self.selected_group].keys())):
                self.current_list_file[self.selected_group][self.selected_modality] = []

            print(self.selected_files)
            self.current_list_file[self.selected_group][self.selected_modality].extend(self.selected_files)
            print(self.current_list_file)
            self.display()
        else:
            self.set_quite_appearance()
            QMessageBox.warning( self, 'Warning', 'Please select group and modality first!')
            self.set_run_appearance()       

    def display(self):
        """
        Display files of the current modality of the current group
        """
        try:
            self.slm_file.setStringList(self.current_list_file[self.selected_group][self.selected_modality])  
            self.listView_files.setModel(self.slm_file)
        except KeyError:
            self.slm_file.setStringList([])  
            self.listView_files.setModel(self.slm_file)

    def identify_selected_group(self, index):
        """Identify the selected file in the list_view_groups and display the files
        """
        self.selected_group = self.current_list_group[index.row()]
        print(self.selected_group)
        self.display()

    def identify_selected_modality(self, index):
        """Identify the selected file in the list_view_modalities and display the files
        """
        self.selected_modality = self.current_list_modality[index.row()]
        print(self.selected_modality)
        self.display()

    def identify_selected_file(self, index):
        """Identify the selected file in the list_view_files
        """
        self.selected_file = self.current_list_file[self.selected_group][self.selected_modality][index.row()]
        print(self.selected_file)

    def remove_selected_group(self):
        """
        This function is used to remove selected group
        """
        if self.current_list_group != []:
            self.set_quite_appearance()
            reply = QMessageBox.question(self, "QListView", "Remove this group: " + self.selected_group + "?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:     
                self.current_list_group = list(set(self.current_list_group) - set([self.selected_group]))  # Note. the second item in set is list
                self.slm_group.setStringList(self.current_list_group)  
                self.listView_groups.setModel(self.slm_group)
            self.set_run_appearance()

        else:
            self.set_quite_appearance()
            QMessageBox.warning( self, 'Warning', 'No group selected!')
            self.set_run_appearance()

    def remove_selected_modality(self):
        """
        This function is used to remove selected modality
        """
        if self.current_list_modality != []:
            self.set_quite_appearance()
            reply = QMessageBox.question(self, "QListView", "Remove this modality: " + self.selected_modality + "?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:     
                self.current_list_modality = list(set(self.current_list_modality) - set([self.selected_modality]))  # Note. the second item in set is list
                self.slm_modality.setStringList(self.current_list_modality)  
                self.listView_modalities.setModel(self.slm_modality)
            self.set_run_appearance()

        else:
            self.set_quite_appearance()
            QMessageBox.warning( self, 'Warning', 'No modality selected!')
            self.set_run_appearance()
    
    def remove_selected_file(self):
        """
        This function is used to remove selected file
        """
        # Try if selected self.selected_group and self.selected_modality
        try:
            self.selected_group
            self.selected_modality
        except AttributeError:
            is_selected_group_and_modality = False
        else:
            is_selected_group_and_modality = True

        if is_selected_group_and_modality & (self.current_list_file != {}):  
            if self.current_list_file[self.selected_group][self.selected_modality] != []:
                self.set_quite_appearance()
                reply = QMessageBox.question(self, "QListView", "Remove this file: " + self.selected_file + "?",
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.Yes:     
                    self.current_list_file[self.selected_group][self.selected_modality] = list(set(self.current_list_file[self.selected_group][self.selected_modality]) - set([self.selected_file]))
                    self.slm_file.setStringList(self.current_list_file[self.selected_group][self.selected_modality])  
                    self.listView_files.setModel(self.slm_file)
                self.set_run_appearance()
    
            else:
                self.set_quite_appearance()
                QMessageBox.warning( self, 'Warning', 'No file selected!')
                self.set_run_appearance() 

    def clear_all_group(self):
        """
        Remove all selections
        """
        self.current_list_group = []
        self.selected_groups = []
        self.slm_group.setStringList(self.selected_groups) 
        self.listView_groups.setModel(self.slm_group)  
    
    def clear_all_modality(self):
        """
        Remove all selections
        """
        self.current_list_modality = []
        self.selected_modalities = []
        self.slm_modality.setStringList(self.selected_modalities)  
        self.listView_modalities.setModel(self.slm_modality) 

    def clear_all_file(self):
        """
        Remove all selections
        """
        # Try if selected self.selected_group and self.selected_modality
        try:
            self.selected_group
            self.selected_modality
        except AttributeError:
            is_selected_group_and_modality = False
        else:
            is_selected_group_and_modality = True
            
        if is_selected_group_and_modality & (self.current_list_file != {}):  
            self.current_list_file[self.selected_group][self.selected_modality] = []
            self.selected_files = []
            self.slm_file.setStringList(self.selected_files)  
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
