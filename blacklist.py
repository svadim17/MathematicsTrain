def processor(self):
    self.dataset = pd.read_csv('datasets/Gold_data.csv')
    df = self.dataset.drop(labels=['Open', 'High', 'Low'], axis=1)  # remove columns
    df['Date'] = pd.to_datetime(df['Date'])  # convert 'Date' to date format

    # Convert to pandas.series
    df_series = pd.Series(df['Price'].values, index=df['Date'])

    # Plot every 10 element of dataframe
    df.set_index('Date', inplace=True)  # set 'Date' column as indexes
    self.plot_input_data(data=df)

    # Split dataframe on train and test frames
    train_size = int(len(df_series) * 0.8)
    train = df_series.iloc[:train_size]
    test = df_series.iloc[train_size:]

    # Apply Holt-Winters method
    model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=7).fit()

    # Get predictions
    predictions = model.forecast(len(test))
    print(test, predictions)

    # Visualize predictions
    fig, ax = plt.subplots()
    plt.plot(train, label='Train')
    plt.plot(test.index, test, label='Test')  # Use test.index to ensure correct alignment
    plt.plot(test.index, predictions, label='Predictions')  # Use test.index to ensure correct alignment
    plt.title('Gold price predictions using Holt-Winters')
    plt.xlabel('Date'), plt.ylabel('Price'), plt.legend()
    self.add_graphs_to_widget(fig)
