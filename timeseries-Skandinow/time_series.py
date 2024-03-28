import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from methods import AR, adf_check, MA, split_train_and_test, difference, mls_vector, undo_difference, predict_value
from scratch.models import to_df
from scratch.show import show_predicted_prices


def prepare_data():
    df = pd.read_csv('./data.csv')
    df = df.replace(',', '.', regex=True)

    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

    df = df.set_index(['Date'])
    df.sort_index(inplace=True)
    df['Price'] = df['Price'].astype(float)
    df = df.drop(columns=['C3', 'C4', 'C5', 'C6', 'C7'])
    return df


def lib_predict(train, test, order):
    model = ARIMA(train, order=order).fit()
    predictions = model.predict(start=len(train), end=len(train) + len(test) - 1, typ='levels')

    return predictions
    # plt.figure(figsize=(10, 5))
    # plt.plot(train['Price'], label='Main')
    # plt.plot(predictions, label='Predicted')
    # plt.plot(test['Price'], label='Actual')
    # plt.legend()
    # plt.title('ARIMA Model Predictions')
    # plt.show()


def i_predict(df, train, order):
    p, d, q = order
    data = difference(train.values, d)
    ar_vector = mls_vector(data, p)

    residuals = [0.] * p
    for i in range(p, train.size):
        residuals.append(data[i] - predict_value(data[i - p:i], ar_vector))

    ma_vector = mls_vector(residuals, q)

    size = df.size - train.size
    result = []
    for i in range(size):
        predicted = predict_value(data[-p:], ar_vector) + predict_value(residuals[-q:], ma_vector)
        result.append(predicted)
        data = np.append(data, predicted)
        residuals.append(0)
    predictions = to_df(undo_difference(data, d), size)

    predictions.index = predictions.index.map(lambda x: df.index[x])
    return predictions


def log_mape(test, lib_prediction, custom_prediction):
    lib_mape = mean_absolute_percentage_error(test.values, lib_prediction.values)
    custom_mape = mean_absolute_percentage_error(test.values, custom_prediction.values)

    print(f'Lib MAPE={lib_mape}, Custom MAPE={custom_mape}')
    print(f'MAPE Diff={round(abs(lib_mape - custom_mape), 2)}')


def main():
    df = prepare_data()
    price = df['Price']
    train, test = split_train_and_test(price)
    predictions_lib = lib_predict(train, test, order=(50, 1, 50))
    predictions_mine = i_predict(df, train, order=(50, 1, 50))

    log_mape(test, predictions_lib, predictions_mine)

    show_predicted_prices(train, test, predictions_lib, predictions_mine)

    # my_arima(df)
    # lib_predict()


main()
