from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import QDir, Qt, QUrl, QSize
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
        QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget, QStatusBar)
import cv2
import os
from threading import Thread
class VideoPlayer(QWidget):


    def startVideo(self,fileName):
        self.mediaPlayer.setMedia(
                QMediaContent(QUrl.fromLocalFile(fileName)))
        self.playButton.setEnabled(True)
        self.statusBar.showMessage(fileName)
        self.fileName=fileName
        Thread(target = self.analyzeVideo).start()
        Thread(target = self.play).start()

        # self.play()
        # self.analyzeVideo()

    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
            self.isVideoPause=not self.isVideoPause
        else:
            self.mediaPlayer.play()
            self.isVideoPause=not self.isVideoPause


    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))

    def positionChanged(self, position):
        self.positionSlider.setValue(position)

    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

    def handleError(self):
        self.playButton.setEnabled(False)
        self.statusBar.showMessage("Error: " + self.mediaPlayer.errorString())

    def analyzeVideo(self):
        video=self.fileName
        path_output_dir='C:/Users/s_ina/Desktop/videoParse'
        vidcap = cv2.VideoCapture(video)
        count = 0
        while vidcap.isOpened():
            while(not self.isVideoPause):
                success, image = vidcap.read()
                count += 1
                if success:
                    if(count%30==0):
                        cv2.imwrite(os.path.join(path_output_dir, '%d.png') % count, image)

                    # count += 1
                else:
                    break
            cv2.destroyAllWindows()
            vidcap.release()

    def setupUi(self,parent=None):
        super(VideoPlayer, self).__init__(parent)
        self.fileName=""
        self.isVideoPause=True
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

        controlLayout = QHBoxLayout()
        controlLayout.setContentsMargins(0, 0, 0, 0)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.positionSlider)

        layout = QVBoxLayout()
        layout.addWidget(videoWidget)
        layout.addLayout(controlLayout)
        layout.addWidget(self.statusBar)

        self.setLayout(layout)

        self.mediaPlayer.setVideoOutput(videoWidget)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.error.connect(self.handleError)
        self.statusBar.showMessage("Ready")
