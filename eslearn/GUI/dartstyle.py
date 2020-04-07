# import sys
# import qdarkstyle
# from PyQt5 import QtWidgets

# # create the application and the main window
# app = QtWidgets.QApplication(sys.argv)
# window = QtWidgets.QMainWindow()

# # setup stylesheet
# app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

# # run
# window.show()
# app.exec_()

from PyQt5 import QtWidgets
from PyQt5.QtCore import QFile, QTextStream
import BreezeStyleSheets
import sys


def main():
    app = QtWidgets.QApplication(sys.argv)

    # set stylesheet
    file = QFile(":/dark.qss")
    file.open(QFile.ReadOnly | QFile.Text)
    stream = QTextStream(file)
    app.setStyleSheet(stream.readAll())

    # code goes here

    app.exec_()

if __name__ == "__main__":
    main()