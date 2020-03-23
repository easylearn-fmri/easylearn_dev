from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtWidgets import QApplication, QWidget
from easyLearn import MainCode_easylearn
from PyQt5.QtWebEngineWidgets import *
from PyQt5.QtCore import QUrl
from PyQt5.QtCore import *
from PyQt5 import *
from PyQt5.QtWidgets import *
import os
import numpy as np
from test_gui import Ui_MainWindow
import sys


class MainCode(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)

        self.setupUi(self)
        self.myset()


    def myset(self):
        # self.setStyleSheet("#MainWindow{background-image:url(D:/My_Codes/LC_Machine_Learning/lc_rsfmri_tools/lc_rsfmri_tools_python/gui_test/a.jpg)}")
        # self.setStyleSheet("#MainWindow{background-color: gray}")
        # self.treeWidget_2.setStyleSheet("background-color: rgb(100, 100, 100);font-size: 40px; color:red")
        self.pushButton_3.setStyleSheet(
            "background-color:red; font-size: 30px; color:white")
        self.pushButton_5.setStyleSheet(
            "background-color:green; font-size: 30px; color:white")
        self.pushButton_6.setStyleSheet(
            "background-color:green; font-size: 30px; color:white")
        self.listWidget.setStyleSheet(
            "border-image:url(D:/My_Codes/LC_Machine_Learning/lc_rsfmri_tools/lc_rsfmri_tools_python/gui_test/7d8f087292ae6cfd6fa0a758ac5e0aa2.jpg)")
        
        # easylearn thread 
        self.pushButton_5.clicked.connect(self.start_easylearn)
        self.pushButton_6.clicked.connect(self.end_easylearn)

    def start_easylearn(self):
        print("Start easylearn")
        self._new_window = MainCode_easylearn()
        self._new_window.show()


    def end_easylearn(self):
        print("End easylearn")
        try:
            self._new_window.close()    
        except AttributeError:
            print("easylearn did not start\n")


    def openUrl(self):
        import webbrowser
        print("Going to web...")
        webbrowser.open_new_tab(
            'https://github.com/easylearn-fmri/easylearn')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    md = MainCode()
    md.show()
    sys.exit(app.exec_())