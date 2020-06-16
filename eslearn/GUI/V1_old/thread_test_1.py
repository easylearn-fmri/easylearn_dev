import sys,time
from PyQt5.QtWidgets import QWidget,QPushButton,QApplication,QListWidget,QGridLayout

class WinForm(QWidget):
    def __init__(self,parent=None):
        super(WinForm, self).__init__(parent)
        #设置标题与布局方式
        self.setWindowTitle('实时刷新界面的例子')
        layout=QGridLayout()

        #实例化列表控件与按钮控件
        self.listFile=QListWidget()
        self.btnStart=QPushButton('开始')

        #添加到布局中指定位置
        layout.addWidget(self.listFile,0,0,1,2)
        layout.addWidget(self.btnStart,1,1)

        #按钮的点击信号触发自定义的函数
        self.btnStart.clicked.connect(self.slotAdd)
        self.setLayout(layout)

    def slotAdd(self):
        for n in range(10):
            #获取条目文本
            str_n='File index{0}'.format(n)
            #添加文本到列表控件中
            self.listFile.addItem(str_n)
            #实时刷新界面
            QApplication.processEvents()
            #睡眠一秒
            time.sleep(0.3)
if __name__ == '__main__':
    app=QApplication(sys.argv)
    win=WinForm()
    win.show()
    sys.exit(app.exec_())
