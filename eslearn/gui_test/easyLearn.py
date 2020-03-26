import sys
import os
from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog
from PyQt5.QtWidgets import *
from PyQt5 import *
from PyQt5.QtGui import QIcon
import sys
from PyQt5.QtWidgets import QApplication,QWidget,QVBoxLayout,QListView,QMessageBox
from PyQt5.QtCore import QStringListModel

from easyLearn_gui_data_loading import Ui_MainWindow

class MainCode_easylearn(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        # set appearance
        self.set_appearance()

        # initiating listView
        self.slm = QStringListModel()
        self.qList = []
        self.current_list = []  # Initializing current list
        self.selected_file = ""

        # connections
        self.browser_workingdir.clicked.connect(self.select_workingdir)
        self.pushButton_removegroups.clicked.connect(self.remove_selected_file)
        self.pushButton_cleargroups.clicked.connect(self.clear_all_selection)
        self.listView_groups.doubleClicked.connect(self.remove_selected_file)
        self.listView_groups.clicked.connect(self.identify_selected_file)
        self.pushButton_addgroups.clicked.connect(self.select_file)
        self.setWindowTitle('Data Loading')
        self.setWindowIcon(QIcon('./easylearn.jpg')) 
    
    def set_appearance(self):
        """Set dart style appearance
        """
        qss_string_all_pushbutton = """
        QPushButton{color: rgb(200,200,200); border: 2px solid rgb(100,100,100); border-radius:10}
        QPushButton:hover {background-color: black; color: white; font-size:20px; font-weight: bold}
        #MainWindow{background-color: rgb(50, 50, 50)}                               
        """
        qss_string_run_pushbutton = """
        QPushButton{background-color:rgb(100,200,100); color:white; border: 2px solid rgb(100,100,100); border-radius:15}        
        QPushButton:hover {background-color:green; color:white; border: 2px solid rgb(100,100,100); border-radius:15; font-weight: bold}                   
        """
        qss_string_quit_pushbutton = """
        QPushButton{background-color:rgb(200,100,100); color:white; border: 2px solid rgb(100,100,100); border-radius:15}   
        QPushButton:hover {background-color:red; color:white; border: 2px solid rgb(100,100,100); border-radius:15; font-weight: bold}                        
        """
        qss_string_textbrowser = """
        background-color:rgb(200,200,200); color:black; border: 2px solid rgb(100,100,100); border-radius:15; font-size:20px
        """
        self.setStyleSheet(qss_string_all_pushbutton)
        # self.run.setStyleSheet(qss_string_run_pushbutton)
        # self.quit.setStyleSheet(qss_string_quit_pushbutton)
        # self.textBrowser.setStyleSheet(qss_string_textbrowser)

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


if __name__=='__main__':
    app=QApplication(sys.argv)
    md=MainCode_easylearn()
    md.show()
    sys.exit(app.exec_())
