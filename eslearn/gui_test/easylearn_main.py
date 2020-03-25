import sys
import os
from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog
from PyQt5.QtWidgets import *
from PyQt5 import *
from PyQt5.QtGui import QIcon
import sys
from PyQt5.QtWidgets import QApplication,QWidget,QVBoxLayout,QListView,QMessageBox
from PyQt5.QtCore import QStringListModel

from easylearn_gui_main import Ui_MainWindow

class easylearn_main(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        # Set appearance
        self.set_appearance()
        
        # Connecting to functions
        self.data_input.clicked.connect(self.data_input_fun)
        self.feature_reduction.clicked.connect(self.feature_reduction_fun)
        self.feature_selection.clicked.connect(self.feature_selection_fun)
        self.model_choosing.clicked.connect(self.model_choosing_fun)
        self.model_training.clicked.connect(self.model_training_fun)
        self.statistical_analysis.clicked.connect(self.statistical_analysis_fun)
        self.save_workflow.clicked.connect(self.save_workflow_fun)
        self.load_workflow.clicked.connect(self.load_workflow_fun)
        self.setWindowTitle('easylearn')
        self.setWindowIcon(QIcon('D:/My_Codes/easylearn-fmri/eslearn/gui_test/bitbug_favicon.ico')) 
    
    def set_appearance(self):
        """
        Set dart style appearance
        """
        qss_string_pbt = """
        QPushButton:hover {background-color: white; color: black}

        QPushButton{color:white; border: 2px solid rgb(100,100,100); border-radius:5}

        #formLayoutWidget_2{color:white; border: 2px solid rgb(100,100,100); border-radius:9}

        #MainWindow{background-color: rgb(50, 50, 50)}
                                       
        """
        self.setStyleSheet(qss_string_pbt)


    def data_input_fun(self):

        print('data_input_fun')

    def feature_reduction_fun(self):

        print('feature_reduction_fun')

    def feature_selection_fun(self):

        print('feature_selection_fun')

    def model_choosing_fun(self):

        print('model_choosing_fun')
   
    def model_training_fun(self):

        print('model_training_fun')

    def statistical_analysis_fun(self):

        print('statistical_analysis_fun')

    def save_workflow_fun(self):
        print('save_workflow_fun')

    def load_workflow_fun(self):
        print('load_workflow_fun')

    def closeEvent(self, event):
        # Set qss to make sure the QMessageBox can be seen
        qss_string_qmessage = """
        QPushButton:hover {background-color: white; color: black}

        QPushButton{color:white; border: 2px solid rgb(100,100,100); border-radius:5}

        #formLayoutWidget_2{color:white; border: 2px solid rgb(100,100,100); border-radius:9}

        #MainWindow{background-color: rgb(50, 50, 50)}

        QMessageBox{background-color: gray; color: white}                       
        """
        self.setStyleSheet(qss_string_qmessage)
        reply = QMessageBox.question(self, 'Quit',"Are you sure to quit?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

if __name__=='__main__':
    app=QApplication(sys.argv)
    md=easylearn_main()
    md.show()
    sys.exit(app.exec_())
