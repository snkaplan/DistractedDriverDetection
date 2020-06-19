from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QAction, QFileDialog,QDialog)
from PyQt5.QtGui import  QPixmap
import os
import sys
scriptpath = "../"

sys.path.append('../')
import ResultsWindow

import videoWindow
from classFolder import mainClass


class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        self.modelClass= mainClass.Klasa()
        self.modelClass.loadModel()
        self.fileName=""
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(400, 300)


        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 409, 21))
        self.menubar.setObjectName("menubar")
        self.menuAnalyze = QtWidgets.QMenu(self.menubar)
        self.menuAnalyze.setObjectName("menuAnalyze")
        self.menuResults = QtWidgets.QMenu(self.menubar)
        self.menuResults.setObjectName("menuResults")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionImage = QtWidgets.QAction(MainWindow)
        self.actionImage.setObjectName("actionImage")
        self.actionImage.triggered.connect(lambda action: self.showDialogForImage())
        self.actionVideo = QtWidgets.QAction(MainWindow)
        self.actionVideo.setObjectName("actionVideo")
        self.actionVideo.triggered.connect(lambda action: self.showDialogForVideo())
        self.actionOld_Result_List = QtWidgets.QAction(MainWindow)
        self.actionOld_Result_List.setObjectName("actionOld_Result_List")
        self.actionOld_Result_List.triggered.connect(lambda action: self.showOldResults())
        self.menuAnalyze.addAction(self.actionImage)
        self.menuAnalyze.addAction(self.actionVideo)
        self.menubar.addAction(self.menuAnalyze.menuAction())
        self.menubar.addAction(self.menuResults.menuAction())
        self.menuResults.addAction(self.actionOld_Result_List)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        # self.loadImageButton = QtWidgets.QPushButton(self.centralwidget)
        # self.loadImageButton.setGeometry(QtCore.QRect(90, 90, 211, 61))
        # self.loadImageButton.setObjectName("loadImageButton")
        # self.loadImageButton.clicked.connect(self.showDialogForImage)
        # self.loadVideoButton = QtWidgets.QPushButton(self.centralwidget)
        # self.loadVideoButton.setGeometry(QtCore.QRect(90, 170, 211, 61))
        # self.loadVideoButton.setObjectName("loadVideoButton")
        # self.loadVideoButton.clicked.connect(self.showDialogForVideo)
        # self.analyzeButton = QtWidgets.QPushButton(self.centralwidget)
        # self.analyzeButton.setGeometry(QtCore.QRect(90, 300, 211, 41))
        # self.analyzeButton.setObjectName("analyzeButton")
        # self.analyzeButton.clicked.connect(self.analyze)
        self.graphicLabel = QtWidgets.QLabel(self.centralwidget)
        self.graphicLabel.setScaledContents(True)
        self.graphicLabel.setGeometry(QtCore.QRect(0, 0, 400, 220))
        self.graphicLabel.setObjectName("graphicLabel")
        self.imageResultLabel = QtWidgets.QLabel(self.centralwidget)
        self.imageResultLabel.setGeometry(QtCore.QRect(170, 230, 171, 16))
        self.imageResultLabel.setObjectName("imageResultLabel")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


# self.actionImage.triggered.connect(lambda action: print("asd"))
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "DDD"))

        self.menuAnalyze.setTitle(_translate("MainWindow", "Analyze"))
        self.menuResults.setTitle(_translate("MainWindow", "Results"))
        self.actionImage.setText(_translate("MainWindow", "Image"))
        self.actionVideo.setText(_translate("MainWindow", "Video"))
        self.actionOld_Result_List.setText(_translate("MainWindow", "Old Result List"))

        # self.loadImageButton.setText(_translate("MainWindow", "Load Image"))
        # self.loadVideoButton.setText(_translate("MainWindow", "Load Video"))
        # self.analyzeButton.setText(_translate("MainWindow", "Analyze"))
        self.imageResultLabel.setText(_translate("MainWindow", ""))

    def showDialogForImage(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '/home')
        self.fileName=fname[0]
        self.analyze()
    def showDialogForVideo(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Video",
                ".", "Video Files (*.mp4 *.flv *.ts *.mts *.avi)")
        if fileName != '':
            dialog = QDialog()
            dialog.ui = videoWindow.VideoPlayer()
            dialog.setWindowTitle("Analyzing Video")
            dialog.ui.setupUi(dialog)
            dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose)
            dialog.ui.startVideo(fileName,self.modelClass)
            dialog.exec_()

    def analyze(self):
        if self.fileName != '':
            prediction=self.modelClass.analyze(self.fileName)
            pixmap = QPixmap(self.fileName)
            self.graphicLabel.setPixmap(pixmap)
            self.imageResultLabel.setText(prediction)
    #        pixmap.save

    
    def showOldResults(self):
            dialog = QDialog()
            dialog.ui = ResultsWindow.ResultsWindow()
            dialog.setWindowTitle("Old Results")
            dialog.ui.setupUi(dialog)
            dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose)
            dialog.exec_()
