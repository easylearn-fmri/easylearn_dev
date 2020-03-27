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
        # self.configuration = configuration
        self.slm = QStringListModel()
        self.qList = []
        self.current_list = []  # Initializing current list
        self.selected_file = ""

        # connections
        self.actionChoose_configuration_file.triggered.connect(self.load_configuration)
        self.actionSave_configuration.triggered.connect(self.save_configuration)
        self.pushButton_removegroups.clicked.connect(self.remove_selected_file)
        self.pushButton_cleargroups.clicked.connect(self.clear_all_selection)
        self.listView_groups.doubleClicked.connect(self.remove_selected_file)
        self.listView_groups.clicked.connect(self.identify_selected_file)
        self.pushButton_addgroups.clicked.connect(self.select_file)
        self.setWindowTitle('Data Loading')
        self.setWindowIcon(QIcon('./easylearn.jpg')) 
    
    def set_run_appearance(self):
        """Set dart style appearance
        """
        qss_string_all = """
        QPushButton{color: rgb(200,200,200); border: 2px solid rgb(100,100,100); border-radius:9}
        QPushButton:hover {background-color: black; color: white; font-size:20px; font-weight: bold}
        #MainWindow{background-color: rgb(50, 50, 50)}    
        QListView{background-color:rgb(200,200,200); color:black; border: 2px solid rgb(100,100,100); border-radius:15; font-size:20px}  
        QFrame{color:white; font-weight: bold}                    
        """
        # qss_string_label = """
        # color:rgb(200,200,200); font-weight: bold;     
        # """
        self.setStyleSheet(qss_string_all)
    
    def set_quite_appearance(self):
        """This function is set appearance when quit program.

        This function make sure quit message can be see clearly.
        """
        qss_string_message = """
        QPushButton{color: black; border: 2px solid rgb(100,100,100); border-radius:9}
        QPushButton:hover {background-color: black; color: white; font-size:20px; font-weight: bold}
        #MainWindow{background-color: rgb(50, 50, 50)}    
        QListView{background-color:rgb(200,200,200); color:black; border: 2px solid rgb(100,100,100); border-radius:15; font-size:20px}  
        QFrame{color:black; font-weight: bold}                    
        """  
        # """
        self.setStyleSheet(qss_string_message)

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
            with open(self.configuration_file, 'r') as config:
                self.configuration = config.read()

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
            self.configuration = json.dumps(self.configuration)
            with open(self.configuration_file, 'w') as config:
                config.write(self.configuration)
        else:
            self.set_quite_appearance()
            QMessageBox.warning( self, 'Warning', 'Configuration is empty, please choose a configuration file first (press button at top left corner)!')
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
            self.qList = os.listdir(self.directory)
            self.current_list = self.qList  # Every time selecting directory, the current list will be initiated once.
            self.slm.setStringList(self.qList)  
            self.listView_groups.setModel(self.slm) 
        except FileNotFoundError:
            self.lineEdit_workingdir.setText("You need to choose a working directory")

    def select_file(self):
        pass

    def identify_selected_file(self, QModelIndex):
        self.selected_file = self.current_list[QModelIndex.row()]

    def remove_selected_file(self):
        """
        This function is used to remove selected file
        """
        if self.current_list != []:
            QMessageBox.information(self, "QListView", "Remove this file: " + self.selected_file)
            self.current_list = list(set(self.current_list) - set([self.selected_file]))  # Note. the second item in set is list
            self.slm.setStringList(self.current_list)  
            self.listView_groups.setModel(self.slm)
        else:
            print(f'No file selected\n') 

    def clear_all_selection(self):
        """
        Remove all selections
        """
        self.current_list = []  # Re-initiating the current_list
        self.qList = []
        self.selected_file = ""
        self.slm.setStringList(self.qList)  # 设置模型列表视图，加载数据列表
        self.listView_groups.setModel(self.slm)  #设置列表视图的模型

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
