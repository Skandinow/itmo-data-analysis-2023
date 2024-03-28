import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller


def adf_check(time_series):
    result = adfuller(time_series)
    labels = ['ADF Test Statistic', 'p-value', 'Number of Lags Used', 'Number of Observations Used']

    for value, label in zip(result, labels):
        print(label + ' : ' + str(value))

    if result[1] <= 0.05:
        print(
            "strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary \n")


def AR(p, df):
    df_temp = df
    for i in range(1, p + 1):
        df_temp['Shifted_values_%d' % i] = df_temp['Price'].shift(i)

    df_train, df_test = split_train_and_test(df_temp)

    df_train_2, x_train, y_train = split_df_train(df_train, p)

    # Running linear regression to generate the coefficents of lagged terms
    lr = LinearRegression()
    lr.fit(x_train, y_train)

    theta = lr.coef_.T
    intercept = lr.intercept_
    df_train_2['Predicted_Values'] = x_train.dot(lr.coef_.T) + lr.intercept_
    # df_train_2[['Price','Predicted_Values']].plot()

    x_test = df_test.iloc[:, 1:].values.reshape(-1, p)
    df_test['Predicted_Values'] = x_test.dot(lr.coef_.T) + lr.intercept_
    # df_test[['Price','Predicted_Values']].plot()

    RMSE = np.sqrt(mean_squared_error(df_test['Price'], df_test['Predicted_Values']))

    print("The RMSE is :", RMSE, ", Price of p : ", p)
    return df_train_2, df_test, theta, intercept, RMSE


def MA(q, res):
    for i in range(1, q + 1):
        res['Shifted_values_%d' % i] = res['Residuals'].shift(i)

    res_train, res_test = split_train_and_test(res)

    res_train_2, x_train, y_train = split_df_train(res_train, q)

    lr = LinearRegression()
    lr.fit(x_train, y_train)

    theta = lr.coef_.T
    intercept = lr.intercept_
    res_train_2['Predicted_Values'] = x_train.dot(lr.coef_.T) + lr.intercept_
    # res_train_2[['Residuals','Predicted_Values']].plot()

    x_test = res_test.iloc[:, 1:].values.reshape(-1, q)
    res_test['Predicted_Values'] = x_test.dot(lr.coef_.T) + lr.intercept_
    # res_test[['Residuals', 'Predicted_Values']].plot()

    RMSE = np.sqrt(mean_squared_error(res_test['Residuals'], res_test['Predicted_Values']))

    print("The RMSE is :", RMSE, ", Value of q : ", q)
    return [res_train_2, res_test, theta, intercept, RMSE]


def split_train_and_test(df_temp):
    train_size = (int)(0.8 * df_temp.size)
    df_train = df_temp.iloc[:train_size]
    df_test = df_temp.iloc[train_size:]

    return df_train, df_test


def split_df_train(df_train, p):
    df_train_2 = df_train.dropna()
    x_train = df_train_2.iloc[:, 1:].values.reshape(-1, p)
    y_train = df_train_2.iloc[:, 0].values.reshape(-1, 1)
    return df_train_2, x_train, y_train


def mls_vector(data, p):
    X = []
    for i in range(len(data) - p):
        X.append([1.0, *data[i:i + p][::-1]])
    X = np.matrix(X)

    Y = np.matrix(data[-(len(data) - p):]).transpose()

    return np.linalg.pinv(np.dot(X.T, X)).dot(X.T).dot(Y)


def predict_value(data, ar_or_ma_vector):
    data_vector = [1., *data[::-1]]
    return np.dot(data_vector, ar_or_ma_vector)[0, 0]


def difference(x, d=1):
    if d == 0:
        return x
    else:
        x = np.r_[x[0], np.diff(x)]
        return difference(x, d - 1)


def undo_difference(x, d=1):
    if d == 1:
        return np.cumsum(x)
    else:
        x = np.cumsum(x)
        return undo_difference(x, d - 1)


def to_df(data, size):
    df = pd.DataFrame({'Price': data[-size:], 'Index': [data.size - size + i for i in range(size)]})
    return df.set_index(['Index'])


def show_prices(prices):
    plt.figure(figsize=(14, 6))
    plt.title('Microsoft Stock Prices')
    plt.xlabel('Date', fontsize=15)
    plt.ylabel('Price ($)', rotation=0, labelpad=30, fontsize=15)
    prices.plot()
    plt.show()


def show_predicted_prices(train, test, lib_predicted, custom_predicted):
    plt.figure(figsize=(14, 6))
    plt.title('Microsoft Stock Prices')
    plt.xlabel('Date', fontsize=15)
    plt.ylabel('Price ($)', rotation=0, labelpad=30, fontsize=15)

    plt.plot(train, label='Train')
    plt.plot(test, label='Test', color='#808080')
    plt.plot(lib_predicted, label='Lib Predicted', color='g')
    plt.plot(custom_predicted, label='Custom Predicted', color='r')

    plt.legend()
    plt.show()
