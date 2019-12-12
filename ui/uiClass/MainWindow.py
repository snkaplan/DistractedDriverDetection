# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.13.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QAction, QFileDialog)
from PyQt5.QtGui import  QPixmap
import os
import sys
scriptpath = "../"

# Add the directory containing your module to the Python path (wants absolute paths)
sys.path.append('../')

# Do the import
#import MyModule
#import mainClass.Model
from classFolder import mainClass
modelClass= mainClass.Klasa()

class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        modelClass.loadModel()
#        print(modelClass.batch_size)
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(809, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.loadImageButton = QtWidgets.QPushButton(self.centralwidget)
        self.loadImageButton.setGeometry(QtCore.QRect(90, 90, 211, 61))
        self.loadImageButton.setObjectName("loadImageButton")
        self.loadImageButton.clicked.connect(self.showDialogForImage)
        self.loadFilePathButton = QtWidgets.QPushButton(self.centralwidget)
        self.loadFilePathButton.setGeometry(QtCore.QRect(90, 170, 211, 61))
        self.loadFilePathButton.setObjectName("loadFilePathButton")
        self.loadFilePathButton.clicked.connect(self.showDialogForPath)
        self.filePathLabel = QtWidgets.QLabel(self.centralwidget)
        self.filePathLabel.setGeometry(QtCore.QRect(340, 190, 451, 31))
        self.filePathLabel.setObjectName("filePathLabel")
        self.imagePathLabel = QtWidgets.QLabel(self.centralwidget)
        self.imagePathLabel.setGeometry(QtCore.QRect(340, 110, 441, 16))
        self.imagePathLabel.setObjectName("imagePathLabel")
        self.analyzeButton = QtWidgets.QPushButton(self.centralwidget)
        self.analyzeButton.setGeometry(QtCore.QRect(90, 300, 211, 41))
        self.analyzeButton.setObjectName("analyzeButton")
        self.analyzeButton.clicked.connect(self.analyze)
        self.graphicLabel = QtWidgets.QLabel(self.centralwidget)
        self.graphicLabel.setScaledContents(True)
        self.graphicLabel.setGeometry(QtCore.QRect(450, 300, 320, 320))
        self.graphicLabel.setObjectName("graphicLabel")
        self.imageResultLabel = QtWidgets.QLabel(self.centralwidget)
        self.imageResultLabel.setGeometry(QtCore.QRect(260, 520, 171, 16))
        self.imageResultLabel.setObjectName("imageResultLabel")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "DDD"))
        self.loadImageButton.setText(_translate("MainWindow", "Load Image"))
        self.loadFilePathButton.setText(_translate("MainWindow", "Load File Path"))
        self.filePathLabel.setText(_translate("MainWindow", " "))
        self.imagePathLabel.setText(_translate("MainWindow", " "))
        self.analyzeButton.setText(_translate("MainWindow", "Analyze"))
        self.imageResultLabel.setText(_translate("MainWindow", "Image Result"))


    def showDialogForImage(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '/home')
        print(fname[0])
        self.imagePathLabel.setText( fname[0])
        print("2",self.imagePathLabel.text())
    def showDialogForPath(self):
        dir_name = QFileDialog.getExistingDirectory(self, 'Select Directory')
        self.filePathLabel.setText( dir_name)
        print(dir_name)

    def analyze(self):
        prediction=modelClass.analyze(self.imagePathLabel.text())
        pixmap = QPixmap(self.imagePathLabel.text())
#        pixmap.scaled(self.graphicLabel.size())
        self.graphicLabel.setPixmap(pixmap)
        self.imageResultLabel.setText(prediction)
#        pixmap.save
