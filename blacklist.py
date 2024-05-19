exp_smooth = ExponentialSmoothing(train, trend='mul', seasonal='mul', seasonal_periods=730)


def factorization(self, dataframe):
    for column in dataframe:
        if type(dataframe[column][0]) is str:
            dataframe[column] = pd.factorize(dataframe[column])[0]
    return dataframe