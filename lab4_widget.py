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
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.cluster import KMeans


class Lab4Widget(QDockWidget, QWidget):
    def __init__(self):
        super().__init__('Lab 4')
        self.central_widget = QWidget(self)
        self.setWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.dataset = None
        self.model = None

        self.create_widgets()
        self.add_widgets_to_layout()

    def create_widgets(self):
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setFontPointSize(14)
        self.log_widget.setMaximumHeight(250)
        text = ('Laboratory 4: Cluster analysis (K-means)\n\n'
                'Dataset: Social network ads\n'
                'Number of clusters = 3')
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

    def processor(self):
        self.tab_widget_graphs.clear()

        self.dataset = pd.read_csv('datasets/Social_Network_Ads.csv')
        df = self.dataset.drop(self.dataset.columns[-1], axis=1)     # remove last column (axis=1 means that it is column)

        # Plot unclustered dataframe
        fig1, ax1 = plt.subplots(1, 1)
        fig1.suptitle('Unclustered data')
        plt.scatter(x=df['Age'], y=df['EstimatedSalary'], s=50)
        plt.xlabel('Age'), plt.ylabel('EstimatedSalary')
        self.add_graphs_to_widget(fig1)

        # Calculate graph of the number of clusters depending on inertia
        max_range = 12
        inertia, clusters = [], []
        for i in range(1, max_range):
            k_means = KMeans(n_clusters=i, init='k-means++')
            k_means.fit(df[['Age']])
            inertia.append(k_means.inertia_)
            clusters.append(i)

        fig2, ax2 = plt.subplots(1, 1)
        fig1.suptitle('Graph of the number of clusters depending on inertia')
        plt.scatter(x=clusters, y=inertia, s=70)
        plt.axvline(3, color='k', linewidth=2, linestyle='--')
        plt.xlabel('Clusters'), plt.ylabel('Inertia')
        self.add_graphs_to_widget(fig2)

        # Create model
        clusters_number = 3
        self.model = KMeans(n_clusters=clusters_number)
        self.model.fit(df[['Age']])

        labels = self.model.labels_     # get cluster labels for every data point
        df['Cluster'] = labels          # add cluster labels to dataframe
        print(df.head())

        fig3, ax3 = plt.subplots(1, 1)
        plt.scatter(x=df['Age'], y=df['EstimatedSalary'], c=df['Cluster'], cmap='viridis',
                    s=50)
        plt.xlabel('Age'), plt.ylabel('EstimatedSalary')
        plt.title('Data with Clusters')
        self.add_graphs_to_widget(fig3)

    def add_graphs_to_widget(self, fig):
        """ This function adds graph to widget """
        canvas = FigureCanvas(fig)
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.addWidget(canvas)
        self.tab_widget_graphs.addTab(tab, f'Graph {len(self.tab_widget_graphs) + 1}')