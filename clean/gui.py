from PyQt5 import QtGui, QtCore, QtWidgets
import sys

class StateTemplate(object):
	def __init__(self):
		self.path="path"

	def __getitem__(self,frame_i):
		print(frame_i)

	def show(self,frame_i,text_i):
		print(frame_i)
		print(text_i)

	def keys(self):
		return ["A","B","C"]

	def save(self,path_i):
		print(path_i)

class ComboBoxDemo(QtWidgets.QWidget):

    def __init__(self, state=None):
        super().__init__()
        if(state is None):
        	state=StateTemplate()
        self.state=state

        self.delta=50
        self.comboBox = QtWidgets.QComboBox(self)
        self.comboBox.setGeometry(self.delta, 50, 400, 35)
        self.comboBox.addItems(self.state.keys() )
        self.comboBox.currentTextChanged.connect(self.getComboValue)

        self.btn = QtWidgets.QPushButton('Show', self)
        self.btn.setGeometry(self.delta, 4*self.delta, 100, 35)
        self.btn.clicked.connect(self.show_frame)

        self.btn = QtWidgets.QPushButton('Save', self)
        self.btn.setGeometry(3*self.delta, 4*self.delta, 100, 35)
        self.btn.clicked.connect(self.save)

        self.textbox = QtWidgets.QLineEdit(self)
        self.textbox.move(50, 90)
        self.textbox.resize(280,40)

        self.pathbox = QtWidgets.QLineEdit(self)
        self.pathbox.move(50, 150)
        self.pathbox.resize(280,40)
        self.pathbox.setText(self.state.path)
        self.getComboValue()

    def getComboValue(self):
        frame_i=self.comboBox.currentText()
        value_i=self.state[frame_i] 
        self.textbox.setText(value_i)

    def show_frame(self):
        frame_i=self.comboBox.currentText()
        text_i=self.textbox.text()
        self.state.show(frame_i,text_i)

    def save(self):
        path_i=self.pathbox.text()
        print("Saves %s" % path_i)
        self.state.save(path_i)

    def closeEvent(self, event):
        self.save()
#       in_path,fun=self.state.path,self.state.cut
#        out_path="%s/%s" % (os.path.dirname(in_path),"cut")
#        dataset.cut_template(in_path,out_path,self.state.cut)

def gui_exp(state=None):
	app = QtWidgets.QApplication(sys.argv)
	demo=ComboBoxDemo(state)
	demo.show()
	sys.exit(app.exec_())