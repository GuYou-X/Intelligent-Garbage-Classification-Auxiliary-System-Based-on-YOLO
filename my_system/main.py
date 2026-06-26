import sys
import os
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication
from ui import MainWindow

os.environ["OMP_NUM_THREADS"] = "1"

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei"))
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())