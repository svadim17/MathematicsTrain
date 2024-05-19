import time

from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, r2_score


class Lab1Widget(QDockWidget, QWidget):
    def __init__(self):
        super().__init__('Lab 1')
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
        text = ('Laboratory 1: Time series forecasting using the Holt-Winters method\n\n'
                'Dataset: The price of gold since 2012 year')
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
        print('Reading dataset...')
        start_time = time.time()
        self.dataset = pd.read_csv('datasets/Gold_data.csv')
        print(f'Time to read dataset: {time.time() - start_time}')
        df = self.dataset.drop(labels=['Open', 'High', 'Low'], axis=1)      # remove columns
        df = df.sort_index(ascending=False)
        df['Date'] = pd.to_datetime(df['Date'])           # convert 'Date' to date format
        print(df.head(10))

        # Convert to pandas.series
        df_series = pd.Series(df['Price'].values, index=df['Date'])

        # Plot every 10 element of dataframe
        df.set_index('Date', inplace=True)          # set 'Date' column as indexes
        self.plot_input_data(data=df)

        # Split dataframe on train and test frames
        print('Splitting dataset...')
        start_time = time.time()
        train_size = int(len(df_series) * 0.9)
        train = df_series.iloc[:train_size]
        test = df_series.iloc[train_size:]
        print(f'Time to split dataset: {time.time() - start_time}')
        print("Shape of train:", train.shape)
        print("Shape of test:", test.shape)

        # Apply Holt-Winters method
        print('Training the model...')
        start_time = time.time()
        exp_smooth = ExponentialSmoothing(train, trend='mul', seasonal='mul', seasonal_periods=730)
        model = exp_smooth.fit(smoothing_level=0.1, smoothing_seasonal=0.1)
        print(f'Time to train model: {time.time() - start_time}')

        # Get predictions
        predictions = model.forecast(len(test))
        predictions.index = test.index

        # Visualize predictions
        fig1, ax1 = plt.subplots()
        plt.plot(train, label='Train')
        plt.plot(test.index, test, label='Test')
        plt.plot(test.index, predictions, label='Predictions')
        plt.title('Gold price predictions using Holt-Winters')
        plt.xlabel('Date'), plt.ylabel('Price'), plt.legend()
        self.add_graphs_to_widget(fig1)

        # Calculate difference between test and predictions data
        difference = predictions - test

        # Visualize difference
        fig2, ax2 = plt.subplots()
        plt.plot(difference, label='Difference')
        plt.title('Difference between test data and predictions')
        plt.xlabel('Date'), plt.ylabel('Price'), plt.legend()
        self.add_graphs_to_widget(fig2)

        # Calculate MSE
        mse = mean_squared_error(test, predictions)
        print(f'\nMean Squared Error = {round(mse, 4)}')
        self.log_widget.append(f'Mean Squared Error = {round(mse, 4)}')

        # Calculate R2_score
        r2 = r2_score(test, predictions)
        print(f'R2_score = {round(r2, 4)}')
        self.log_widget.append(f'R2_score = {round(r2, 4)}')

        # Forecast future
        future_periods = 200
        last_test_date = test.index[-1]
        future_index = pd.date_range(start=last_test_date, periods=future_periods, freq='D')
        future_predictions = model.forecast(steps=future_periods)
        future_predictions.index = future_index

        # Visualize future predictions
        fig3, ax3 = plt.subplots()
        plt.plot(train, label='Train')
        plt.plot(test.index, test, label='Test')
        plt.plot(predictions.index, predictions, label='Predictions')
        plt.plot(future_predictions.index, future_predictions, label='Future Predictions')
        plt.title('Gold price with future predictions')
        plt.xlabel('Date'), plt.ylabel('Price'), plt.legend()
        self.add_graphs_to_widget(fig3)

    def add_graphs_to_widget(self, fig):
        """ This function adds graph to widget """
        canvas = FigureCanvas(fig)
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.addWidget(canvas)
        self.tab_widget_graphs.addTab(tab, f'Graph {len(self.tab_widget_graphs) + 1}')

    def plot_input_data(self, data):
        # # Create new dataframe with every 10 element of original dataframe
        # data_small = data.iloc[::10].reset_index(drop=True)
        #
        # data_small['Date'] = data_small['Date'].str[-4:]       # edit column 'Date' to get only year
        #
        # fig1, ax1 = plt.subplots()
        # sns.lineplot(x=data_small['Date'][::10], y=(data_small['Price'][::10]), label='Input data')
        # ax1.set_title('Input data'), ax1.set_xlabel('Date'), ax1.set_ylabel('Price')
        # self.add_graphs_to_widget(fig1)

        fig, ax = plt.subplots()
        plt.plot(data['Price'], label='Gold price')
        plt.title('Gold price time series')
        plt.xlabel('Date'), plt.ylabel('Price')
        self.add_graphs_to_widget(fig)
