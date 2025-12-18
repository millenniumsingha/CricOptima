# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'evaluate.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(617, 574)
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        
        self.team_label = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.team_label.setFont(font)
        self.team_label.setObjectName("team_label")
        self.horizontalLayout.addWidget(self.team_label)
        
        self.team_combo = QtWidgets.QComboBox(Dialog)
        self.team_combo.setObjectName("team_combo")
        self.horizontalLayout.addWidget(self.team_combo)

        import sqlite3
        mydata = sqlite3.connect('cricket_match.db')
        sql = "select name from teams"
        data = mydata.execute(sql)
        teams = []
        for arr in data:
            self.team_combo.addItem(arr[0])
        mydata.close()
        
        self.match_label = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.match_label.setFont(font)
        self.match_label.setObjectName("match_label")
        self.horizontalLayout.addWidget(self.match_label)
        
        self.match_combo = QtWidgets.QComboBox(Dialog)
        self.match_combo.setObjectName("match_combo")
        self.match_combo.addItem("")
        self.horizontalLayout.addWidget(self.match_combo)
        
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        
        self.players_label = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.players_label.setFont(font)
        self.players_label.setObjectName("players_label")
        self.horizontalLayout_2.addWidget(self.players_label)
        
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        
        self.score_label = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.score_label.setFont(font)
        self.score_label.setObjectName("score_label")
        self.horizontalLayout_2.addWidget(self.score_label)
        
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        
        self.listWidget1 = QtWidgets.QListWidget(Dialog)
        self.listWidget1.setObjectName("listWidget1")
        self.horizontalLayout_3.addWidget(self.listWidget1)
        
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        
        self.listWidget2 = QtWidgets.QListWidget(Dialog)
        self.listWidget2.setObjectName("listWidget2")
        self.horizontalLayout_3.addWidget(self.listWidget2)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        
        self.evaluate_button = QtWidgets.QPushButton(Dialog)
        self.evaluate_button.setObjectName("evaluate_button")

        self.evaluate_button.clicked.connect(self.evaluate)
        
        self.horizontalLayout_4.addWidget(self.evaluate_button)
        
        self.evaluate_line = QtWidgets.QLineEdit(Dialog)
        self.evaluate_line.setObjectName("evaluate_line")
        self.horizontalLayout_4.addWidget(self.evaluate_line)
        self.verticalLayout.addLayout(self.horizontalLayout_4)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.team_label.setText(_translate("Dialog", "Team"))
        self.match_label.setText(_translate("Dialog", "Match"))
        self.match_combo.setItemText(0, _translate("Dialog", "Match"))
        self.players_label.setText(_translate("Dialog", "Players"))
        self.score_label.setText(_translate("Dialog", "Score"))
        self.evaluate_button.setText(_translate("Dialog", "Evaluate Score"))
        self.evaluate_line.setText(_translate("Dialog", "00"))


    def evaluate(self):     
        import sqlite3
        mydata = sqlite3.connect('cricket_match.db')
        team = self.team_combo.currentText()
        self.listWidget1.clear()
        sql1 = "select players, value from teams where name='"+team+"'"
        data = mydata.execute(sql1)
        arr = data.fetchone()

        selected = arr[0].split(',')

        self.listWidget1.addItems(selected)
        team_t1 = 0

        self.listWidget2.clear()
        match = self.match_combo.currentText()

        for i in range(self.listWidget1.count()):
            tt1, batscore, bowlscore, fieldscore = 0,0,0,0
            
            nm = self.listWidget1.item(i).text()
            data = mydata.execute("select * from "+match+" where player='"+nm+"'")
            arr = data.fetchone()
            
            batscore = int(arr[1]/2)

            if batscore>=50:
                batscore+=5

            if batscore>=100:
                batscore+=10

            if arr[1]>0:
                sr = arr[1]/arr[2]
                if sr>=80 and sr<100:
                    batscore+=2
                if sr>=100:
                    batscore+=4

            batscore+=arr[3]
            batscore+=2*arr[4]
            
            bowlscore = arr[8]*10

            if arr[8]>=3:
                bowlscore+=5
            if arr[8]>=5:
                bowlscore+=10
            if arr[7]>0:
                er = 6*arr[7]/arr[5]

                if er<=2:
                    bowlscore+=10
                if er>2 and er<=3.5:
                    bowlscore+=7
                if er>3.5 and er<=4.5:
                    bowlscore+=4

            fieldscore = (arr[9]+arr[10]+arr[11])*10

            tt1 = batscore + bowlscore + fieldscore

            self.listWidget2.addItem(str(tt1))

            team_t1+=tt1

        self.evaluate_line.setText(str(team_t1))
        

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
