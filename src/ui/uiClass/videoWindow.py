from PyQt5.QtGui import QIcon, QFont, QPixmap
from PyQt5.QtCore import QDir, Qt, QUrl, QSize
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
        QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget, QStatusBar)
import cv2
import os
import time
from threading import Thread
import sys
from shutil import copy
from datetime import datetime
import json
import shutil
sys.path.append('../')
from classFolder import mainClass
class VideoPlayer(QWidget):


    def startVideo(self,fileName,modelClass):
        self.mediaPlayer.setMedia(
                QMediaContent(QUrl.fromLocalFile(fileName)))
        self.playButton.setEnabled(True)
        self.statusBar.showMessage(fileName)
        self.fileName = fileName
        self.modelClass=modelClass
        Thread(target = self.takeFramesFromVideo).start()
        Thread(target = self.play).start()

    def play(self):
        if self.mediaPlayer.state() != QMediaPlayer.PlayingState:
            self.mediaPlayer.play()
        else:
            self.mediaPlayer.pause()




    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))

    def positionChanged(self, position):
        if(self.mediaPlayer.state() == QMediaPlayer.PlayingState):
            self.analyzeVideo(position//1000)
        self.positionSlider.setValue(position)

    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

    def handleError(self):
        self.playButton.setEnabled(False)
        self.statusBar.showMessage("Error: " + self.mediaPlayer.errorString())

    def writeLog(self,logFileName,nowTime,status):
        f= open("C:/DistractedDriverDetection/DistractedDriverDetection/log/"+os.path.basename(self.fileName).split(".")[0]+".txt","a")
        obj ={"level":"WARNING","time":nowTime,"fileName":logFileName,"msg":status,"hostname":"pc","version":"1.0"}
        f.write(json.dumps(obj) + "\n")
        f.close()

    def analyzeVideo(self,position):
        if self.fileName == '':
            pass
        else:
            if(os.path.isfile(self.outputFolder+"/"+str(position)+".jpg")):
                prediction=self.modelClass.analyze(self.outputFolder+"/"+str(position)+".jpg")
                if(prediction != "Safe driving"):
                    self.graphicLabel.setText("<img src=../images/warning.png align=middle> " + "Not Safe Driving ---- "+prediction)
                    resultsFolder='C:/DistractedDriverDetection/Frames/VideoResults/'+os.path.basename(self.fileName)
                    nowTime=datetime.now().strftime("%d-%m-%Y %H-%M-%S.%f'")
                    if not os.path.exists(resultsFolder):
                        os.makedirs(resultsFolder)
                    shutil.copyfile(self.outputFolder+"/"+str(position)+".jpg", 'C:/DistractedDriverDetection/Frames/VideoResults/'+os.path.basename(self.fileName)+"/"+str(position)+".jpg")
                    # src ='C:/DistractedDriverDetection/Frames/VideoResults/'+os.path.basename(self.fileName)+"/"+str(position)+".jpg"
                    # dst ='C:/DistractedDriverDetection/Frames/VideoResults/'+os.path.basename(self.fileName)+"/"+str(position)+"_"+nowTime+".jpg"
                    # os.rename(src, dst)
                    self.safeDrivingCounter=0
                    # pixmap = QPixmap(self.warningImage)
                    imageName= str(position)+".jpg"
                    self.writeLog(imageName,nowTime,prediction)
                else:
                    self.safeDrivingCounter = self.safeDrivingCounter + 1
                    self.graphicLabel.setText("<img src=../images/check.png align=middle> " + "Safe Driving")
                    # if(self.safeDrivingCounter>=2):
                    # else:
                    #     self.graphicLabel.setText(" ")

            else:
                pass


    def takeFramesFromVideo(self):
        video=self.fileName
        self.outputFolder='C:/DistractedDriverDetection/Frames/VideoParse/'+os.path.basename(video)
        if not os.path.exists(self.outputFolder):
            os.makedirs(self.outputFolder)
        vidcap = cv2.VideoCapture(video)
        count = 0
        while vidcap.isOpened():
            success, image = vidcap.read()
            if success:
                if(count%3==0):
                    cv2.imwrite(os.path.join(self.outputFolder, '%d.jpg') % (count/3), image)

                count += 1
            else:
                break
        cv2.destroyAllWindows()
        vidcap.release()

    def setupUi(self,parent=None):
        super(VideoPlayer, self).__init__(parent)
        self.fileName=""
        self.outputFolder=""
        self.resize(600,400)
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        btnSize = QSize(16, 16)
        videoWidget = QVideoWidget()
        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setFixedHeight(24)
        self.playButton.setIconSize(btnSize)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        self.statusBar = QStatusBar()
        self.statusBar.setFont(QFont("Noto Sans", 7))
        self.statusBar.setFixedHeight(14)

        self.graphicLabel = QLabel()
        self.graphicLabel.setScaledContents(True)
        self.graphicLabel.setObjectName("graphicLabel")
        self.graphicLabel.setFixedSize(300, 80)
        self.graphicLabel.setTextFormat(Qt.RichText)

        controlLayout = QHBoxLayout()
        controlLayout.setContentsMargins(0, 0, 0, 0)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.positionSlider)

        layout = QVBoxLayout()
        layout.addWidget(videoWidget)
        layout.addLayout(controlLayout)
        layout.addWidget(self.statusBar)
        layout.addWidget(self.graphicLabel)

        self.setLayout(layout)

        self.mediaPlayer.setVideoOutput(videoWidget)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.error.connect(self.handleError)
        self.statusBar.showMessage("Ready")
        self.modelClass=None
        self.safeDrivingCounter=2
        self.checkImage="C:/DistractedDriverDetection/DistractedDriverDetection/ui/images/check.png"
        self.warningImage="C:/DistractedDriverDetection/DistractedDriverDetection/ui/images/warning.png"
