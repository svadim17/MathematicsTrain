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


class Lab3Widget(QDockWidget, QWidget):
    def __init__(self):
        super().__init__('Lab 3')
        self.central_widget = QWidget(self)
        self.setWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.model = None

        self.create_widgets()
        self.add_widgets_to_layout()

    def create_widgets(self):
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setFontPointSize(14)
        self.log_widget.setMaximumHeight(250)
        text = ('Laboratory 3: Bayesian Classifier\n\n\n'
                'Algorithm created model that predict diabete by 8 parameters of people.')
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

        # read dataframe
        self.df = pd.read_csv('datasets/Diabetes_prediction.csv')
        self.df = self.df.head(100000)

        # factorization (to convert string data types to numeric)
        for column in self.df:
            if type(self.df[column][0]) is str:
                self.df[column] = pd.factorize(self.df[column])[0]

        self.df = self.factorization(dataframe=self.df)

        # Checking data for multicollinearity
        fig1, ax1 = plt.subplots()
        sns.heatmap(round(abs(self.df.corr()), 1), annot=True)
        ax1.set_title('Checking data for multicollinearity')
        self.add_graphs_to_widget(fig1)

        # Create model
        train_input, test_input, train_output, test_output = self.split_dataset(dataset=self.df)    # splitting
        self.model = GaussianNB()
        self.model.fit(train_input, train_output)

        predictions = self.model.predict(test_input)
        accuracy = metrics.accuracy_score(predictions, test_output)

        print(f'Accuracy of model = {round(accuracy * 100, 3)} %')
        self.append_log_widget(accuracy=round(accuracy * 100, 3))

        print('\n')
        self.andrey_test()

    def factorization(self, dataframe):
        for column in dataframe:
            if type(dataframe[column][0]) is str:
                dataframe[column] = pd.factorize(dataframe[column])[0]
        return dataframe

    def split_dataset(self, dataset):
        """ This function splits dataset to train and test dataframes with train_test_split.
        dataset.drop - creates new frame excluding column 'diabetes' (class marker = diabetes result);
        dataset['diabetes'] - column with marks that we want to predict;
        test_size - means the size of test dataframe in proportion. """
        train_input, test_input, train_output, test_output = train_test_split(dataset.drop('diabetes', axis=1),
                                                                              dataset['diabetes'],
                                                                              test_size=0.2)
        return train_input, test_input, train_output, test_output

    def add_graphs_to_widget(self, fig1):
        canvas1 = FigureCanvas(fig1)
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        tab1_layout.addWidget(canvas1)
        self.tab_widget_graphs.addTab(tab1, 'Graph 1')

    def append_log_widget(self, accuracy):
        self.log_widget.clear()
        text = f'Accuracy of created model = {accuracy} %\n'
        self.log_widget.setText(text)

    def andrey_test(self):
        andrey_frame = pd.DataFrame({
            'gender': ['Male'],
            'age': [23.0],
            'hypertension': [1],                      # 0 - no,    1 - yes       / no info \
            'heart_disease': [0],
            'smoking_history': ['never'],
            'bmi': [30.2],                            # / no info \
            'HbA1c_level': [6.2],
            'blood_glucose_level': [200]
        })

        andrey_frame = self.factorization(andrey_frame)
        prediction = self.model.predict(andrey_frame)
        if prediction[0] == 1:
            result = 'DIABETE'
        else:
            result = 'NO DIABETE'
        print(f'Andrey`s prediction is {result}')

        probability = self.model.predict_proba(andrey_frame)
        print(f'Andrey`s probability of diabetes is {round(float(probability[:, 1] * 100), 3)} %')

