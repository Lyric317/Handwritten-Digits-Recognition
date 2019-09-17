import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import numpy as np
import preprocess
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class HandwritingWidget(QWidget):
    
    def __init__(self):
        #initialize all widgets
        QWidget.__init__(self, windowTitle=u"Digit hand writing recognition")
        palette1= QPalette()
        palette1.setColor(self.backgroundRole(),QColor(250,240,230))
        self.setPalette(palette1)
        self.resize(800,800)
        self.outputArea = QTextBrowser(self)
        self.rennButton = QPushButton(self)
        self.rennButton.setText('Neuro Network Recognize')
        self.recnnButton = QPushButton(self)
        self.recnnButton.setText('Convolutional Neuro Network Recognize')
        self.clrButton = QPushButton(self)
        self.clrButton.setText('Clear')
        
        self.rennButton.resize(500,50)
        self.rennButton.move(150,600)
        self.recnnButton.resize(500,50)
        self.recnnButton.move(150,670)
        self.outputArea.resize(360,50)
        self.outputArea.move(150,730)
        self.clrButton.resize(120,50)
        self.clrButton.move(530,730)
        
        self.setMouseTracking(False)
        self.pos_xy = []

        self.rennButton.clicked.connect(self.postrans)
        self.recnnButton.clicked.connect(self.postrans)
        self.clrButton.clicked.connect(self.clrwidget)
    
    def paintEvent(self, event):
        self.painter = QPainter()
        self.painter.begin(self)
        pen = QPen(Qt.black, 8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        self.painter.setPen(pen)

        #explorer every point of self.pos_xy and paint it
        if len(self.pos_xy) > 1:
            point_start = self.pos_xy[0]
            for pos_tmp in self.pos_xy:
                point_end = pos_tmp

                if point_end == (-1, -1):
                    point_start = (-1, -1)
                    continue
                if point_start == (-1, -1):
                    point_start = point_end
                    continue

                self.painter.drawLine(point_start[0], point_start[1], point_end[0], point_end[1])
                point_start = point_end
        self.painter.end()
    
    def mouseMoveEvent(self, event):
        '''
            event of mouse movement: add current point to pos_xy
            when update() run，paintEvent() will be cleared
        '''
        #set pos_tmp as event current point
        pos_tmp = (event.pos().x(), event.pos().y())
        print(pos_tmp)
        #add pos_tmp to pos_xy[]
        self.pos_xy.append(pos_tmp)

        self.update()
    
    def mouseReleaseEvent(self, event):
        '''
            When mouse is released,
            add a break point (-1, -1)
            then skip the break point to draw next one
        '''
        pos_test = (-1, -1)
        self.pos_xy.append(pos_test)

        self.update()

    def postrans(self):
        pixim = np.ones((800,800))
        for pos in self.pos_xy:
            pixim[pos[0]][pos[1]] = 0
        pixim = np.transpose(pixim)
        image,label = preprocess.imreset(pixim)
        #self.restore_nn()
        sess = tf.Session()
        saver = tf.train.import_meta_graph('nn.ckpt.meta')
        saver.restore(sess,'nn.ckpt')

        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        y_ = graph.get_tensor_by_name("y_:0")
        pkeep = graph.get_tensor_by_name("pkeep:0")
        accuracy = graph.get_tensor_by_name("accuracy:0")
        y = graph.get_tensor_by_name("y:0")
        y = sess.run(y, feed_dict={x:image , y_:label,pkeep:1.0})
    
        with tf.Session() as sess:
            #print("Recognition result:",sess.run(tf.argmax(y,1)))
            self.outputArea.setText(str(sess.run(tf.argmax(y,1))))
    
    def clrwidget(self):
        self.pos_xy = []
        #self.painter = QPainter()
        self.painter.begin(self)
        self.painter.eraseRect(QRect(0,0,800,800))
        print("done")

app=QApplication(sys.argv)
app.setStyle('Fusion')
HWWidget=HandwritingWidget()
HWWidget.show()
sys.exit(app.exec_())       