# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.13.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
import DBContext
from PyQt5.QtWidgets import (QWidget)
class ResultsWindow(object):
    def getCollectionNames(self):
        collections = self.dbContext.getCollections()
        self.collectionList.clear()
        self.collectionList.addItems(collections)
    def loadData(self,collectionName):
        result = self.dbContext.getDataFromCollection(str(self.collectionList.currentText()))
        self.resultTable.setRowCount(0)
        for row_number,row_data in enumerate(result):
            self.resultTable.insertRow(row_number)
            row_data.pop('_id', None)
            for column_number,data in enumerate(row_data):
                item = QtWidgets.QTableWidgetItem(str(row_data[data]))
                item.setFlags(QtCore.Qt.ItemIsEnabled)
                self.resultTable.setItem(row_number,column_number,item)
    def setupUi(self, MainWindow):
        self.columnNames = ["FileName", "Message","Time","Level" ]
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(677, 387)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.resultTable = QtWidgets.QTableWidget(self.centralwidget)
        # self.resultTable.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.resultTable.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.resultTable.setGeometry(QtCore.QRect(10, 60, 651, 301))
        self.resultTable.setObjectName("resultTable")
        self.resultTable.setRowCount(5)
        self.resultTable.setColumnCount(4)
        self.resultTable.setHorizontalHeaderLabels(self.columnNames)
        header = self.resultTable.horizontalHeader()       
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        self.collectionList = QtWidgets.QComboBox(self.centralwidget)
        self.collectionList.setGeometry(QtCore.QRect(10, 30, 181, 22))
        self.collectionList.setObjectName("collectionList")
        self.loadButton = QtWidgets.QPushButton(self.centralwidget)
        self.loadButton.setGeometry(QtCore.QRect(210, 30, 75, 23))
        self.loadButton.setObjectName("loadButton")
        self.loadButton.clicked.connect(self.loadData)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 677, 21))
        self.menubar.setObjectName("menubar")
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.dbContext = DBContext.DBContext.getInstance()
        self.dbContext.getCollections()
        self.getCollectionNames()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.loadButton.setText(_translate("MainWindow", "Load"))

