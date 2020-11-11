# -*- coding: utf-8 -*-

"""Change skin
"""

from eslearn.stylesheets.PyQt5_stylesheets import pyqt5_loader


def change_skin(app):
    """Set skins"""
    if app:
        sender = app.sender()
    if sender:
        if (sender.text() in list(app.skins.keys())):
            app.setStyleSheet(pyqt5_loader.load_stylesheet_pyqt5(style=app.skins[sender.text()]))
        else:
            app.setStyleSheet(pyqt5_loader.load_stylesheet_pyqt5(style="style_Dark"))
    else:
        app.setStyleSheet(pyqt5_loader.load_stylesheet_pyqt5(style="style_Dark"))