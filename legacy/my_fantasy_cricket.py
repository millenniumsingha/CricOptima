# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'fantasy_cricket.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(973, 760)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        
        self.selection_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setWeight(75)
        self.selection_label.setFont(font)
        self.selection_label.setObjectName("selection_label")
        self.verticalLayout.addWidget(self.selection_label)
        
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        
        self.bat_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.bat_label.setFont(font)
        self.bat_label.setObjectName("bat_label")
        self.horizontalLayout.addWidget(self.bat_label)
        
        self.bat_line = QtWidgets.QLineEdit(self.centralwidget)
        self.bat_line.setObjectName("bat_line")
        self.horizontalLayout.addWidget(self.bat_line)
        
        self.bow_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.bow_label.setFont(font)
        self.bow_label.setObjectName("bow_label")
        self.horizontalLayout.addWidget(self.bow_label)
        
        self.bow_line = QtWidgets.QLineEdit(self.centralwidget)
        self.bow_line.setObjectName("bow_line")
        self.horizontalLayout.addWidget(self.bow_line)
        
        self.ar_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.ar_label.setFont(font)
        self.ar_label.setObjectName("ar_label")
        self.horizontalLayout.addWidget(self.ar_label)
        
        self.ar_line = QtWidgets.QLineEdit(self.centralwidget)
        self.ar_line.setObjectName("ar_line")
        self.horizontalLayout.addWidget(self.ar_line)
        
        self.wk_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.wk_label.setFont(font)
        self.wk_label.setObjectName("wk_label")
        self.horizontalLayout.addWidget(self.wk_label)
        
        self.wk_line = QtWidgets.QLineEdit(self.centralwidget)
        self.wk_line.setObjectName("wk_line")
        self.horizontalLayout.addWidget(self.wk_line)
        
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout_3.addLayout(self.verticalLayout)
        
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        
        self.pa_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pa_label.setFont(font)
        self.pa_label.setObjectName("pa_label")
        self.horizontalLayout_2.addWidget(self.pa_label)
        
        self.pa_line = QtWidgets.QLineEdit(self.centralwidget)
        self.pa_line.setObjectName("pa_line")
        self.horizontalLayout_2.addWidget(self.pa_line)
        
        self.pu_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pu_label.setFont(font)
        self.pu_label.setObjectName("pu_label")
        self.horizontalLayout_2.addWidget(self.pu_label)
        
        self.pu_line = QtWidgets.QLineEdit(self.centralwidget)
        self.pu_line.setObjectName("pu_line")
        self.horizontalLayout_2.addWidget(self.pu_line)
        
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        
        self.team_label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.team_label.setFont(font)
        self.team_label.setObjectName("team_label")
        self.horizontalLayout_3.addWidget(self.team_label)
        
        self.team_line = QtWidgets.QLineEdit(self.centralwidget)
        self.team_line.setObjectName("team_line")
        self.horizontalLayout_3.addWidget(self.team_line)
        
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        
        self.bat_btn = QtWidgets.QRadioButton(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.bat_btn.setFont(font)
        self.bat_btn.setObjectName("bat_btn")
        self.horizontalLayout_4.addWidget(self.bat_btn)
        
        self.bow_btn = QtWidgets.QRadioButton(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.bow_btn.setFont(font)
        self.bow_btn.setObjectName("bow_btn")
        self.horizontalLayout_4.addWidget(self.bow_btn)
        
        self.ar_btn = QtWidgets.QRadioButton(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.ar_btn.setFont(font)
        self.ar_btn.setObjectName("ar_btn")
        self.horizontalLayout_4.addWidget(self.ar_btn)
        
        self.wk_btn = QtWidgets.QRadioButton(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.wk_btn.setFont(font)
        self.wk_btn.setObjectName("wk_btn")
        self.horizontalLayout_4.addWidget(self.wk_btn)
        
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        
        self.listWidget1 = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget1.setObjectName("listWidget1")
        self.horizontalLayout_5.addWidget(self.listWidget1)
        
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem)
        
        self.listWidget2 = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget2.setObjectName("listWidget2")
        self.horizontalLayout_5.addWidget(self.listWidget2)
        
        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        self.verticalLayout_3.addLayout(self.verticalLayout_2)
        MainWindow.setCentralWidget(self.centralwidget)
        
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 973, 31))
        self.menubar.setObjectName("menubar")
        self.menuManage_Teams = QtWidgets.QMenu(self.menubar)
        self.menuManage_Teams.setObjectName("menuManage_Teams")
        MainWindow.setMenuBar(self.menubar)
        
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        self.actionNew_Team = QtWidgets.QAction(MainWindow)
        self.actionNew_Team.setObjectName("actionNew_Team")
        self.actionOpen_Team = QtWidgets.QAction(MainWindow)
        self.actionOpen_Team.setObjectName("actionOpen_Team")
        self.actionSave_Team = QtWidgets.QAction(MainWindow)
        self.actionSave_Team.setObjectName("actionSave_Team")
        self.actionEvaluate_Team = QtWidgets.QAction(MainWindow)
        self.actionEvaluate_Team.setObjectName("actionEvaluate_Team")
        self.menuManage_Teams.addAction(self.actionNew_Team)
        self.menuManage_Teams.addAction(self.actionOpen_Team)
        self.menuManage_Teams.addAction(self.actionSave_Team)
        self.menuManage_Teams.addAction(self.actionEvaluate_Team)
        self.menubar.addAction(self.menuManage_Teams.menuAction())

        self.listWidget1.itemDoubleClicked.connect(self.removelistWidget1)
        self.listWidget2.itemDoubleClicked.connect(self.removelistWidget2)

        self.bat_btn.toggled.connect(self.ctg)
        self.bow_btn.toggled.connect(self.ctg)
        self.ar_btn.toggled.connect(self.ctg)
        self.wk_btn.toggled.connect(self.ctg)

        self.menuManage_Teams.triggered[QtWidgets.QAction].connect(self.menu)
        
        self.bat = 0
        self.bwl = 0
        self.ar = 0
        self.wk = 0

        self.points_available = 2000
        self.points_used = 0

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.selection_label.setText(_translate("MainWindow", "Your Selection"))
        self.bat_label.setText(_translate("MainWindow", "Batsmen (BAT)"))
        self.bow_label.setText(_translate("MainWindow", "Bowlers (BOW)"))
        self.ar_label.setText(_translate("MainWindow", "AllRounders (AR)"))
        self.wk_label.setText(_translate("MainWindow", "Wicket-keeper (WK)"))
        self.pa_label.setText(_translate("MainWindow", "Points Available"))
        self.pu_label.setText(_translate("MainWindow", "Points Used"))
        self.team_label.setText(_translate("MainWindow", "Your Team Name"))
        self.bat_btn.setText(_translate("MainWindow", "BAT"))
        self.bow_btn.setText(_translate("MainWindow", "BOW"))
        self.ar_btn.setText(_translate("MainWindow", "AR"))
        self.wk_btn.setText(_translate("MainWindow", "WK"))
        self.menuManage_Teams.setTitle(_translate("MainWindow", "Manage Teams"))
        self.actionNew_Team.setText(_translate("MainWindow", "New Team"))
        self.actionOpen_Team.setText(_translate("MainWindow", "Open Team"))
        self.actionSave_Team.setText(_translate("MainWindow", "Save Team"))
        self.actionEvaluate_Team.setText(_translate("MainWindow", "Evaluate Team"))

    def menu(self,action):
        txt = (action.text())

        if txt=='New Team':
            self.bat = 0
            self.bwl = 0
            self.ar = 0
            self.wk = 0

            self.points_available = 2000
            self.points_used = 0

            self.listWidget1.clear()
            self.listWidget2.clear()

            text, ok = QtWidgets.QInputDialog.getText(MainWindow, "Team", "Enter name of team:")
            if ok:
                self.team_line.setText(str(text))
            self.show()

        if txt=='Save Team':
            count = self.listWidget2.count()
            selected = ""

            for i in range(count):
                selected+=self.listWidget2.item(i).text()
                if i<count-1:
                    selected+=","
                else:
                    selected+=""
            self.saveteam(self.team_line.text(),selected,self.points_used)

        if txt=='Open Team':
            self.bat = 0
            self.bwl = 0
            self.ar = 0
            self.wk = 0

            self.points_available = 2000
            self.points_used  = 0

            self.listWidget1.clear()
            self.listWidget2.clear()

            self.show()
            self.openteam()

        if txt=='Evaluate Team':
            from match_evaluation import Ui_Dialog
            Dialog = QtWidgets.QDialog()
            ui = Ui_Dialog()
            ui.setupUi(Dialog)
            ret = Dialog.exec()

    def show(self):
        self.bat_line.setText(str(self.bat))
        self.bow_line.setText(str(self.bwl))
        self.wk_line.setText(str(self.wk))
        self.ar_line.setText(str(self.ar))

        self.pa_line.setText(str(self.points_available))
        self.pu_line.setText(str(self.points_used))

    def criteria(self,ctgr,item):
        msg=''
        if ctgr=='BAT' and self.bat>=5:
            msg="Batsmen not more than 5"
        if ctgr=='BWL' and self.bwl>=5:
            msg="Bowlers not more than 5"
        if ctgr=='AR' and self.ar>=3:
            msg="All-rounders not more tha 3"
        if ctgr=='WK' and self.wk>=1:
            msg="Wicket-keepers not more than 1"
        if msg!='':
            self.showdlg(msg)
            return False

        if self.points_available<=0:
            msg = "You have exhausted your points"
            self.showdlg(msg)
            return False

        if ctgr=="BAT":
            self.bat+=1
        if ctgr=="BWL":
            self.bwl+=1
        if ctgr=="AR":
            self.ar+=1
        if ctgr=="WK":
            self.wk+=1

        sql = "SELECT value from stats where player='"+item.text()+"'"
        data = mydata.execute(sql)
        arr = data.fetchone()

        self.points_available-=int(arr[0])
        self.points_used+=int(arr[0])
        return True

    def removelistWidget1(self,item):
        ctgr=''

        if self.bat_btn.isChecked()==True:
            ctgr='BAT'
        if self.bow_btn.isChecked()==True:
            ctgr='BWL'
        if self.ar_btn.isChecked()==True:
            ctgr='AR'
        if self.wk_btn.isChecked()==True:
            ctgr='WK'
        ret = self.criteria(ctgr,item)

        if ret==True:
            self.listWidget1.takeItem(self.listWidget1.row(item))
            self.listWidget2.addItem(item.text())
            self.show()

    def ctg(self):
        ctgr=''

        if self.bat_btn.isChecked()==True:
            ctgr='BAT'
        if self.bow_btn.isChecked()==True:
            ctgr='BWL'
        if self.ar_btn.isChecked()==True:
            ctgr='AR'
        if self.wk_btn.isChecked()==True:
            ctgr='WK'

        self.fillList(ctgr)

    def removelistWidget2(self,item):
        self.listWidget2.takeItem(self.listWidget2.arr(item))
        
        data = mydata.execute("SELECT player, value, ctg from stats where player='"+item.text()+"'")
        arr = data.fetchone()

        self.points_available+=int(arr[1])
        self.points_used-=int(arr[1])

        ctgr = arr[2]

        if ctgr=="BAT":
            self.bat-=1
            if self.bat_btn.isChecked()==True:
                self.listWidget1.addItem(item.text())
        if ctgr=="BWL":
            self.bwl-=1
            if self.bow_btn.isChecked()==True:
                self.listWidget1.addItem(item.text())        
        if ctgr=="AR":
            self.ar-=1
            if self.ar_btn.isChecked()==True:
                self.listWidget1.addItem(item.text())
        if ctgr=="WK":
            self.wk-=1
            if self.wk_btn.isChecked()==True:
                self.listWidget1.addItem(item.text())

        self.show()

    def fillList(self,ctgr):
        if self.team_line.text()=='':
            self.showdlg("Enter name of team")
            return

        self.listWidget1.clear()
        sql = "SELECT player from players where ctg='"+ctgr+"';"

        data = mydata.execute(sql)

        for arr in data:
            selected = []
            for i in range(self.listWidget2.count()):
                selected.append(self.listWidget2.item(i).text())
            if arr[0] not in selected:
                self.listWidget1.addItem(arr[0])

    def openteam(self):
        sql = "select name from teams;"
        data = mydata.execute(sql)
        teams = []

        for arr in data:
            teams.append(arr[0])

        team, ok = QtWidgets.QInputDialog.getItem(MainWindow,"Dream","Choose a Team",teams,0,False)

        if ok and team:
            self.team_line.setText(team)

        sql1 = "SELECT players, value from teams where name='"+team+"';"
        data = mydata.execute(sql1)
        arr = data.fetchone()

        selected = arr[0].split(',')

        self.listWidget2.addItems(selected)
        self.points_used = arr[1]
        self.points_available = 2000 - arr[1]

        count = self.listWidget2.count()

        for i in range(count-1):
            play = self.listWidget2.item(i).text()
            sql = "select ctg from stats where player='"+play+"';"

            data = mydata.execute(sql)
            arr = data.fetchone()

            ctgr = arr[0]

            if ctgr=="BAT":
                self.bat+=1
            if ctgr=="BWL":
                self.bwl+=1
            if ctgr=="AR":
                self.ar+=1
            if ctgr=="WK":
                self.wk+=1
        self.show()

    def saveteam(self, name, play, val):
        if self.bat+self.bwl+self.ar+self.wk!=11:
            self.showdlg("Insufficient players")
            return

        sql = "INSERT INTO teams(name, players, value) VALUES ('"+name+"','"+play+"','"+str(val)+"');"

        try:
            data = mydata.execute(sql)
            self.showdlg("Team saved successfully")
            mydata.commit()
        except:
            self.showdlg("Error in operation")
            mydata.rollback()

    def showdlg(self,msg):
        Dialog = QtWidgets.QMessageBox()
        Dialog.setText(msg)
        Dialog.setWindowTitle("Dream team selector")
        ret = Dialog.exec()

if __name__ == "__main__":
    import sqlite3
    mydata = sqlite3.connect('cricket_match.db')
    
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
