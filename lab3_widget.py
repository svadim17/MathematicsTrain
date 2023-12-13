from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import pandas as pd
import seaborn as sns       # for create custom graphs
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


class Lab3Widget(QDockWidget, QWidget):
    def __init__(self):
        super().__init__('Lab 3')
        self.central_widget = QWidget(self)
        self.setWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.create_widgets()
        self.add_widgets_to_layout()

    def create_widgets(self):
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setFontPointSize(14)
        self.log_widget.setMaximumHeight(250)
        text = 'Laboratory 3: Bayesian Classifier'
        self.log_widget.setText(text)

        self.tab_widget_graphs = QTabWidget()

        self.btn_start = QPushButton('Start')
        self.btn_start.setFixedHeight(33)

    def add_widgets_to_layout(self):
        spacerItem = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.main_layout.addWidget(self.log_widget)
        self.main_layout.addWidget(self.tab_widget_graphs)
        self.main_layout.addWidget(self.btn_start)
        # self.main_layout.addItem(spacerItem)
