import sklearn.metrics
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd
import seaborn as sns       # for create custom graphs
from sklearn.model_selection import train_test_split
from sklearn import tree
import time


class Lab5Widget(QDockWidget, QWidget):
    def __init__(self):
        super().__init__('Lab 5')
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
        text = ('Laboratory 5: Decision trees\n\n'
                'Dataset: Peoples`s parameters that have or not laptops')
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

        print('Reading dataset...')
        start_time = time.time()
        self.dataset = pd.read_csv('datasets/Laptop_Users.csv')
        X = self.dataset.drop(labels='Has Laptop', axis=1)        # remove column 'Has Laptop'
        print(f'Time to read dataset: {time.time() - start_time}')
        print(X.head(10))

        print('Factorization (converting string data types to numeric)...')
        start_time = time.time()
        X, factorization_table = self.factorization(dataset=X, columns_for_factorization=['Gender', 'Region', 'Occupation'])
        print(f'Time to factorization: {time.time() - start_time}')

        # Checking for originality of features and their correlation
        fig1, ax1 = plt.subplots()
        sns.heatmap(round(abs(X.corr()), 1), annot=True)
        ax1.set_title('Checking for originality of features and their correlation')
        self.add_graphs_to_widget(fig1)

        # Split dataset
        print('Splitting dataset...')
        start_time = time.time()
        train_input, test_input, train_output, test_output = train_test_split(X, self.dataset['Has Laptop'], test_size=0.2)
        print(f'Time to split dataset: {time.time() - start_time}')
        print("Shape of train_input:", train_input.shape)
        print("Shape of test_input:", test_input.shape)
        print("Shape of train_output:", train_output.shape)
        print("Shape of test_output:", test_output.shape)

        # Build model of tree decision
        self.model = tree.DecisionTreeClassifier()
        print('Training the model...')
        start_time = time.time()
        self.model.fit(train_input, train_output)
        print(f'Time to train model: {time.time() - start_time}')

        predictions = self.model.predict(test_input)
        print(predictions)

        confusion_matrix = sklearn.metrics.confusion_matrix(predictions, test_output)

        fig2, ax2 = plt.subplots()
        sns.heatmap(confusion_matrix, annot=True)
        ax2.set_title('Confusion matrix')
        self.add_graphs_to_widget(fig2)

        # Get decision tree
        fig3, ax3 = plt.subplots()
        tree.plot_tree(self.model)
        ax2.set_title('Decision Tree')
        self.add_graphs_to_widget(fig3)

        self.custom_test_tree()
        # print(self.dataset.head())

    def factorization(self, dataset, columns_for_factorization: list):
        factorization_table = {}
        for column in dataset.columns:
            if column in columns_for_factorization:
                dataset[column], table = pd.factorize(dataset[column])
                factorization_table[column] = pd.DataFrame(columns=[column], data=table)
        return dataset, factorization_table

    def custom_test_tree(self):
        """ Example:
           Age  Gender       Region Occupation  Income
           'int' 'str'       'str'    'str'      'int'
           14    male         city    student       0
           34  female         city    teacher   22000
           42    male  countryside     banker   24000
           30    male  countryside    teacher   25000
           16    male         city    student       0 """
        df = {
            'Age': [23],
            'Gender': ['male'],
            'Region': ['city'],
            'Occupation': ['student'],
            'Income': [10000]
        }

        custom_df = pd.DataFrame(df)

        custom_df_factor, _ = self.factorization(custom_df, columns_for_factorization=['Gender', 'Region', 'Occupation'])
        custom_prediction = self.model.predict(custom_df_factor)
        self.log_widget.append(f'\nPrediction on this dataset is: "{custom_prediction[0]} laptop"\n{df} ')

    def add_graphs_to_widget(self, fig):
        """ This function adds graph to widget """
        canvas = FigureCanvas(fig)
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.addWidget(canvas)
        self.tab_widget_graphs.addTab(tab, f'Graph {len(self.tab_widget_graphs) + 1}')
