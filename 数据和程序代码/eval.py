# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sys

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.treeWidget = QtWidgets.QTreeWidget(self.centralwidget)
        self.treeWidget.setGeometry(QtCore.QRect(20, 60, 191, 491))
        self.treeWidget.setObjectName("treeWidget")
        self.treeWidget.headerItem().setText(0, "1")
        self.tableView = QtWidgets.QTableView(self.centralwidget)
        self.tableView.setGeometry(QtCore.QRect(230, 60, 250, 481))
        self.tableView.setObjectName("tableView")
        self.tableView_2 = QtWidgets.QTableView(self.centralwidget)
        self.tableView_2.setGeometry(QtCore.QRect(500, 60, 250, 471))
        self.tableView_2.setObjectName("tableView_2")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(230, 20, 121, 22))
        self.comboBox.setObjectName("comboBox")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 956, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.comboBox.addItems(['2014','2015','2016'])
        self.comboBox.activated[str].connect(self.select_value)
        self.comboBox.currentIndexChanged[str].connect(self.chg_value)

        sw = pd.read_excel('sw.xlsx')
        ind = sw.iloc[:, 0].value_counts()
        indname = list(ind.index)
        root = QTreeWidgetItem(self.treeWidget)
        root.setText(0, '申银万国行业分类')
        root.setText(1, '0')
        for i in range(len(indname)):
            child = QTreeWidgetItem(root)
            child.setText(0, indname[i])
            child.setText(1, str(i))

        self.treeWidget.clicked.connect(self.selectname)
        self.chg_i = '2014'
        self.select = 0

    def selectname(self):
        self.select = 1
        self.eval_fun('2014')
        if self.chg_i!='2014':
            self.eval_fun(self.chg_i)

    def select_value(self, i):
        if self.select != 0:
            self.eval_fun(i)

    def chg_value(self, i):
        self.chg_i = i

    def mataindustry(self,x1, x2):
        '''
        x1：股票代码
        x2:行业数据
        '''
        # 获得行业名
        w = list(x2.iloc[x2.iloc[:, 1].values == x1, 0])
        if w == []:
            w = "其他"
        else:
            w = str(w)[2:-2]
        # 获得股票代码名
        return w

    def eval_fun(self, year):
        import fun
        import pandas as pd
        hy = self.treeWidget.currentItem()  # 获得点击树的值
        hy=hy.text(0)
        data = pd.read_excel('Data' + year + '.xlsx')
        code = []
        for i in range(len(data)):
            code.append(data.iloc[i, 0][:6])
        sw = pd.read_excel('sw.xlsx', dtype=str)
        code1 = list(sw.iloc[sw['行业名称'].values ==hy, 1].values)
        index = []

        for c in code1:
            a = c in code
            if a == True:
                index.append(code.index(c))
        dt = data.iloc[index, :]
        r = fun.Fr(dt)
        s1 = r[1]
        if len(s1) > 0:
            self.model = QStandardItemModel(len(s1), 2)
            self.model.setHorizontalHeaderLabels(['股票名称', '综合得分排名'])
            for row in range(len(s1)):
                for column in range(2):
                    if column == 0:
                        a = QStandardItem(s1.index[row])

                    else:
                        a = QStandardItem(str(s1[row]))
                    self.model.setItem(row, column, a)
            self.tableView.setModel(self.model)

        import pandas as pd
        import fun1
        import numpy as np
        import Re_comput
        # 构建投资组合
        dt = pd.read_excel('上市公司总体规模与投资效率指标.xlsx')
        x2 = pd.read_excel('sw.xlsx')
        r = fun1.Fr1(dt,year)
        c = r[0]
        c['行业'] = c['Stkcd'].apply(lambda x: self.mataindustry(x1=x, x2=x2))
        c = c.iloc[c.iloc[:, 2].values == hy, :]
        data1 = c.iloc[:10, :]
        list_code = []
        list_22 = []
        code = data1.iloc[:, 0].values
        code = list(code)
        year = '201' + str(int(year[-1]) + 1)  # 年数加1
        path = year + '年所有上市股票交易数据.xlsx'  # 获取后一年的量化投资数据
        DA = pd.read_excel(path)  # 2015年所有上市股票交易数据
        for i in range(len(code)):
            data3 = DA.iloc[DA.iloc[:, 0].values == code[i], :]
            if len(data3) > 1:
                list_code.append(code[i])
                z2 = Re_comput.Re(data3, year)
                list_22.append(z2[2])
                a=sum(list_22)
        D = {'Stkcd': list_code, 'lr_total': list_22}
        a={'Stkcd':'总收益','lr_total':a}
        a=pd.DataFrame(a)
        D=pd.DataFrame(D)
        D=np.vstack((D,a))  
        D=pd.DataFrame(D)      
        if len(D) > 0:
            self.model1 = QStandardItemModel(len(D), 2)
            self.model1.setHorizontalHeaderLabels(['股票代码', '收益率'])
            for row in range(len(D)):
                for column in range(2): 
                    b= QStandardItem(str(D.iloc[row,column]))
                    self.model1.setItem(row, column, b)
            self.tableView_2.setModel(self.model1)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "上市公司综合评价"))

if __name__=='__main__':
    app=QtWidgets.QApplication(sys.argv)
    MainWindow=QtWidgets.QMainWindow()
    ui_test=Ui_MainWindow()
    ui_test.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())