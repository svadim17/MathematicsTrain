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


class Lab2Widget(QDockWidget, QWidget):
    def __init__(self):
        super().__init__('Lab 2')
        self.central_widget = QWidget(self)
        self.setWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.polinom_degree = 4
        self.create_widgets()
        self.add_widgets_to_layout()

    def create_widgets(self):
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setFontPointSize(14)
        self.log_widget.setMaximumHeight(250)
        text = 'Laboratory 2: Nonlinear Regression\n\n'
        text += 'Dataset: People`s salary in different countries according to the age\n'
        text += f'Polinom`s degree: {self.polinom_degree}'
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

        # Reading dataframe about people's salary in different countries (age, years, salary: float64;  country: object)
        self.df = pd.read_csv('datasets/Salary.csv', usecols=['Age', 'Salary'])
        # df = pd.read_csv('datasets/Salary.csv', usecols=['Age', 'Years of Experience', 'Salary', 'Country'])
        self.df = self.df.head(6000)  # choose first 6000 rows

        # Plot dependency graph from input data
        fig1, ax1 = plt.subplots()
        sns.lineplot(x=self.df['Age'], y=self.df['Salary'], label='Input data')
        ax1.set_title('Dependency graph'), ax1.set_xlabel('Age'), ax1.set_ylabel('Salary')
        ax1.legend()
        # plt.show()

        # sklearn input is 1D array, so transform frames to array
        age = np.array(self.df['Age']).reshape(-1, 1)  # -1 - all elements; 1 - one column
        salary = np.array(self.df['Salary']).reshape(-1, 1)

        # Create Model
        DEGREE = self.polinom_degree  # polinom's degree
        regression = make_pipeline(PolynomialFeatures(DEGREE), LinearRegression())
        regression.fit(age, salary)
        predictions = regression.predict(age)
        MSE = np.sqrt(np.mean((predictions - salary) ** 2))
        print(f'Mean squared error = {MSE}')

        # Compare input values with predictions
        fig2, ax2 = plt.subplots()
        sns.lineplot(x=self.df['Age'], y=self.df['Salary'], label='Actual')
        sns.lineplot(x=self.df['Age'], y=predictions.reshape(-1), label='Predicted')
        ax2.set_title('Actual vs Predicted'), ax2.set_xlabel('Age'), ax2.set_ylabel('Salary')
        ax2.legend()
        # plt.show()

        # Export coefficients
        x_param_temp = np.append(regression['linearregression'].intercept_[0],
                                 regression['linearregression'].coef_[0][1:])
        x_parameters = list(round(value, 2) for value in x_param_temp)

        # Calculate equation
        equation = self.formatting_to_equation(DEGREE, x_parameters)
        print(f'Equation: {equation}')

        # Compare input values with predictions
        fig3, ax3 = plt.subplots()
        sns.lineplot(x=self.df['Age'], y=self.df['Salary'], label='Actual', zorder=1)
        sns.lineplot(x=self.df['Age'], y=predictions.reshape(-1), label='Regression line', linestyle='--', zorder=2)
        sns.scatterplot(x=self.df['Age'], y=self.df['Salary'], label='Data Points', color='red',
                        zorder=3)  # add points from input data
        ax3.set_title('Comparing'), ax3.set_xlabel('Age'), ax3.set_ylabel('Salary')
        plt.legend()
        # plt.show()

        self.add_graphs_to_widget(fig1, fig2, fig3)
        self.append_log_widget(MSE, equation)

    def add_graphs_to_widget(self, fig1, fig2, fig3):
        canvas1 = FigureCanvas(fig1)
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        tab1_layout.addWidget(canvas1)
        self.tab_widget_graphs.addTab(tab1, 'Graph 1')

        canvas2 = FigureCanvas(fig2)
        tab2 = QWidget()
        tab2_layout = QVBoxLayout(tab2)
        tab2_layout.addWidget(canvas2)
        self.tab_widget_graphs.addTab(tab2, 'Graph 2')

        canvas3 = FigureCanvas(fig3)
        tab3 = QWidget()
        tab3_layout = QVBoxLayout(tab3)
        tab3_layout.addWidget(canvas3)
        self.tab_widget_graphs.addTab(tab3, 'Graph 3')

    def formatting_to_equation(self, degree, parameters):
        output = 'y ='
        for i in range(degree + 1):
            if parameters[i] >= 0:
                output += ' + '
            else:
                output += ' - '
                parameters[i] = abs(parameters[i])

            output += str(parameters[i])
            if i == 0:
                pass
            elif i == 1:
                output += f'x'
            elif i == 2:
                output += f'x\u00B2'
            elif i == 3:
                output += f'x\u00B3'
            elif i == 4:
                output += f'x\u2074'
            elif i == 5:
                output += f'x\u2075'
            elif i == 6:
                output += f'x\u2076'
            elif i == 7:
                output += f'x\u2077'
            else:
                raise ValueError('Very high polinom`s degree!')

        return output

    def append_log_widget(self, mse, equation):
        self.log_widget.clear()
        text = f'Mean Squared Error = {mse}\n'
        text += f'Equation: {equation}'
        self.log_widget.setText(text)



