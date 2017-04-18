
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from matplotlib.ticker import MaxNLocator
from matplotlib.finance import candlestick_ohlc
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.neural_network import MLPRegressor, MLPClassifier


def plot_raw_data(data, fig_path=''):
    r"""Plot raw data.

    """
    _date = data.index.tolist()
    _open = data.loc[:, 'Open'].tolist()
    _high = data.loc[:, 'High'].tolist()
    _low = data.loc[:, 'Low'].tolist()
    _close = data.loc[:, 'Close'].tolist()
    _volume = data.loc[:, 'Volume'].tolist()
    fig, axarr = plt.subplots(2, sharex=True)
    # plot open high, low, close
    axarr[0].grid(True)
    candlestick_ohlc(axarr[0], zip(date2num(_date), _open, _high, _low, _close),
                     width=.2, colorup='k', colordown='r')
    axarr[0].xaxis_date()
    axarr[0].autoscale_view()
    # plot volume
    axarr[1].grid(True)
    axarr[1].bar(_date, _volume, linewidth=1)
    fig.subplots_adjust(hspace=0, bottom=0.2)
    #  fig.tight_layout()
    plt.setp(axarr[1].get_xticklabels(), rotation=45,
             horizontalalignment='right')
    axarr[1].yaxis.set_major_locator(MaxNLocator(prune='upper'))
    # save figure as pdf or show it.
    if fig_path:
        fig.savefig(fig_path)
        plt.close()
    else:
        plt.show()


def EWMA(data, period, min_periods=0, column='Close', add_column='EWMA', **kwargs):
    r"""Calculate Exponential Weighted Moving Average (EWMA).

    Using Smoothing Factor 2/(1+N), where N is periods.

    """
    ewma = data[column].ewm(span=period, min_periods=min_periods, **kwargs).mean()
    return data.join(ewma.to_frame(add_column))


def MACD(data, min_periods=[0,0], column='Close', add_column='MACD', **kwargs):
    r"""Calculate Moving Average Convergence/Divergence (MACD)

    The MACD line is the 12-day EWMA less the 26-day EWMA.

    Using Smoothing Factor 2/(1+N), where N is periods.

    NOTE: similar to Price Oscillator (OSCP).

    """
    ewma12 = data[column].ewm(span=12, min_periods=min_periods[0],
                              **kwargs).mean()
    ewma26 = data[column].ewm(span=26, min_periods=min_periods[1],
                              **kwargs).mean()
    macd = ewma12 - ewma26
    return data.join(macd.to_frame(add_column))


def RSI(data, period=14, min_periods=0, column='Close', add_column='RSI'):
    r""" Calculate Relative Strength Index (RSI).

    Here using Smoothed Modified Moving Average (SMMA), where smoothing
    parameter is 1/N.

    """
    delta = data[column].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down < 0] = 0
    SMMA_up = up.ewm(com=period-1, min_periods=min_periods, adjust=False).mean()
    SMMA_down = down.ewm(com=period-1, min_periods=min_periods, adjust=False).mean()
    rsi = 100 - 100 / (1 + SMMA_up / SMMA_down)
    return data.join(rsi.to_frame(add_column))


def _ATR(data, period=14, min_periods=0, columns=['High', 'Low', 'Close']):
    r"""Calculate Average True Range.

    Here using Smoothed Modified Moving Average (SMMA), where smoothing
    parameter is 1/N.

    """
    # Calculate the range of a day's trading
    HL_delta = (data[columns[0]] - data[columns[1]])
    HC_delta = np.abs(data[columns[0]] - data[columns[2]].shift(1))
    LC_delta = np.abs(data[columns[1]] - data[columns[2]].shift(1))
    TR = map(max, zip(HL_delta, HC_delta, LC_delta))
    TR_series = pd.Series(index=data.index, data=TR)
    atr = TR_series.ewm(com=period-1, min_periods=min_periods,
                        adjust=False).mean()
    return atr



def ADX(data, period=14, min_periods=0, columns=['High', 'Low', 'Close'],
        add_column='ADX'):
    r"""Calculate Average Directional Index (ADX)

    Here using Smoothed Modified Moving Average (SMMA), where smoothing
    parameter is 1/N.

    """
    DM_plus_list = []
    DM_minus_list = []
    up_move = data[columns[0]].diff()
    down_move = data[columns[1]].diff()
    for i in range(len(up_move)):
        if i == 0:
            DM_plus_list.append(np.nan)
            DM_minus_list.append(np.nan)
        else:
            if up_move[i] > down_move[i] and up_move[i] > 0:
                DM_plus_list.append(up_move[i])
            else:
                DM_plus_list.append(0)
            if down_move[i] > up_move[i] and down_move[i] > 0:
                DM_minus_list.append(down_move[i])
            else:
                DM_minus_list.append(0)
    DM_plus = pd.Series(data=DM_plus_list, index=data.index)
    DM_minus = pd.Series(data=DM_minus_list, index=data.index)
    # Smoothed moving average of DM_plus and DM_minus
    SMA_DM_plus = DM_plus.ewm(com=period-1, min_periods=min_periods,
                              adjust=False).mean()
    SMA_DM_minus = DM_minus.ewm(com=period-1, min_periods=min_periods,
                                adjust=False).mean()
    # DI_plus and DI_minus
    DI_plus = 100 * SMA_DM_plus / _ATR(data, period=period,
                                       min_periods=min_periods, columns=columns)
    DI_minus = 100 * SMA_DM_minus / _ATR(data, period=period,
                                       min_periods=min_periods, columns=columns)
    DI_plus_minus = abs(DI_plus - DI_minus) / (DI_plus + DI_minus)
    adx = 100 * DI_plus_minus.ewm(com=period-1, min_periods=min_periods,
                      adjust=False).mean()
    return data.join(adx.to_frame(add_column))


def fast_stochastic_oscillator(data, period=14, smoothing=3,\
                               columns=['Close', 'High', 'Low'],
                               add_columns=['fast_%K', 'fast_%D']):
    r""" Calculate Fast Stochastic Oscillator.

    """
    _K = 100 * (data[columns[0]] - data[columns[2]].rolling(period).min()) /\
    (data[columns[1]].rolling(period).max() -\
     data[columns[2]].rolling(period).min())
    _D = _K.rolling(smoothing).mean()
    return data.join(_K.to_frame(add_columns[0]).join\
                     (_D.to_frame(add_columns[1]))), _K, _D


def slow_stochastic_oscillator(data, period=14, smoothing=3,\
                               columns=['Close', 'High', 'Low'],
                               add_columns=['slow_%K', 'slow_%D']):
    r""" Calculate Slow Stochastic Oscillator.

    """
    _, fast_K, fast_D = fast_stochastic_oscillator(data, period=period,
                                                smoothing=smoothing,
                                                columns=columns,
                                                add_columns=['a', 'b'])
    _K = fast_D
    _D = _K.rolling(smoothing).mean()
    return data.join(_K.to_frame(add_columns[0]).join\
                     (_D.to_frame(add_columns[1])))


def momentum(data, period=4, column='Close', add_column='Momentum'):
    r""" Calculate Momentum.

    """
    M = data[column] / data[column].shift(period) * 100
    return data.join(M.to_frame(add_column))


def acceleration(data, periods=[5, 34, 5], columns=['Close', 'High', 'Low'],\
                 add_column='Acceleration'):
    r""" Calculate Acceleration.

    """
    median = (data[columns[1]] + data[columns[2]]) / 2
    AO = median.rolling(periods[0]).mean() - median.rolling(periods[1]).mean()
    AC = AO - AO.rolling(periods[2]).mean()
    return data.join(AC.to_frame(add_column))


def williams_R(data, period=14, columns=['Close', 'High', 'Low'],
               add_column='Williams_%R'):
    r""" Calculate Williams R.

    """
    williams = (data[columns[1]].rolling(period).max() - data[columns[0]]) /\
            (data[columns[1]].rolling(period).max() -\
             data[columns[2]].rolling(period).min()) * -100
    return data.join(williams.to_frame(add_column))


def accDist(data, columns=['Close', 'High', 'Low', 'Volume'],
            add_column='Acc/Dist'):
    r""" Calculate Accumulation/Distribution

    """
    CLV = ((data[columns[0]] - data[columns[2]]) - \
           (data[columns[1]] - data[columns[0]])) /\
            (data[columns[1]] - data[columns[2]])
    volume_clv = data[columns[3]] * CLV
    accdist = volume_clv.cumsum()
    return data.join(accdist.to_frame(add_column)), accdist


def chaikin_oscillator(data, periods=[3, 10], min_periods=[0, 0],\
                       columns=['Close', 'High', 'Low', 'Volume'],
                       add_column='Chaikin_Oscillator', **kwargs):
    r""" Calculate Chaikin Oscillator.

    """
    _, accdist_series = accDist(data, columns=columns, add_column='a')
    ewma1 = accdist_series.ewm(span=periods[0], min_periods=min_periods[0],
                              **kwargs).mean()
    ewma2 = accdist_series.ewm(span=periods[1], min_periods=min_periods[1],
                              **kwargs).mean()
    chaikin = ewma1 - ewma2
    return data.join(chaikin.to_frame(add_column))


def william_AccDist(data, columns=['Close', 'High', 'Low'],
                    add_column='William_AccDist'):
    r""" Calculate William Accumulation/distribution.

    """
    AD_list = []
    for i in range(len(data[columns[0]])):
        if i == 0:
            AD_list.append(0)
        else:
            TRH = max(data[columns[0]][i-1], data[columns[1]][i])
            TRL = min(data[columns[0]][i-1], data[columns[2]][i])
            if data[columns[0]][i] > data[columns[0]][i-1]:
                AD = data[columns[0]][i] - TRL
            elif data[columns[0]][i] < data[columns[0]][i-1]:
                AD = data[columns[0]][i] - TRH
            else:
                AD = 0
            AD_list.append(AD)
    AD_series = pd.Series(data=AD_list, index=data.index).cumsum()
    return data.join(AD_series.to_frame(add_column))


def on_balance_volume(data, columns=['Close', 'Volume'], add_column='OBV'):
    r""" Calculate On Balance Volume (OBV).

    """
    OBV_list = [0]
    for i in range(len(data[columns[1]]))[1: ]:
        today_volume = data[columns[1]][i]
        if data[columns[0]][i] > data[columns[0]][i-1]:
            OBV_list.append(OBV_list[i-1] + today_volume)
        elif data[columns[0]][i] < data[columns[0]][i-1]:
            OBV_list.append(OBV_list[i-1] - today_volume)
        else:
            OBV_list.append(OBV_list[i-1])
    OBV = pd.Series(data=OBV_list, index=data.index)
    return data.join(OBV.to_frame(add_column))


def disparity_index(data, period=5, min_periods=0, column='Close',
                    add_column='Disparity', **kwargs):
    r""" Calculate Disparity Index.

    Using EWMA here.

    """
    moving_average = data[column].ewm(span=period, min_periods=min_periods,
                                      **kwargs).mean()
    disparity = 100 * (data[column] - moving_average) / moving_average
    return data.join(disparity.to_frame(add_column))


def CCI(data, period=14, columns=['Close', 'High', 'Low'], add_column='CCI'):
    r""" Calculate Commodity Channel Index (CCI).

    """
    TP = (data[columns[0]] + data[columns[1]] + data[columns[2]]) / 3
    moving_average = TP.rolling(period).mean()
    # mean absolute deviation
    mad = lambda x: np.fabs(x - x.mean()).mean()
    mean_absolute_deviation = TP.rolling(period).apply(mad)
    cci = (TP - moving_average) / (0.015 * mean_absolute_deviation)
    return data.join(cci.to_frame(add_column))


def _standardization(X_train, columns_to_standardize=[], X_test=None):
    r""" Standardize predictors by removing the mean and scaling to unit variance.

    """
    standardize = StandardScaler(with_mean=True, with_std=True)
    fit_params = standardize.fit(X_train[:, columns_to_standardize])  # compute mean and std
    X_train[:, columns_to_standardize] = standardize.transform(X_train[:, columns_to_standardize])
    if X_test is not None:
        X_test[:, columns_to_standardize] = standardize.transform(X_test[:, columns_to_standardize])
    else:
        X_test = None
    return X_train, X_test


def _KFolds_filter(data, n_splits=10):
    r""" Cross Validation.

    """
    groups = data.shape[0] % n_splits
    folds_list = []
    for i in range(n_splits):
        if i < groups:
            size = data.shape[0] // n_splits + 1
            folds_list.append(range(i*size, (i+1)*size))
        else:
            size = data.shape[0] // n_splits
            folds_list.append(range(i*size, (i+1)*size))
    return folds_list


def _accuracy(data_true, data_pred):
    r""" Accuracy for binary data.

    """
    false = sum(abs(data_pred - data_true))
    result = 1 - false / data_pred.shape[0]
    return result


def logistic_CV(X, Y, Cs=[], n_splits=10,
                columns_to_standardize=[], penalty='l1'):
    r""" Tune parameters in logistic regression using CV.

    """
    folds = _KFolds_filter(X, n_splits=10)
    accuracy_list = []
    for c in Cs:
        accuracy = 0
        for i in range(len(folds))[1:]:
            test_index = folds[i]
            train_index = [item for sublist in folds[:i] for item in sublist]
            X_train = X[train_index, :]
            X_test = X[test_index, :]
            Y_train = Y[train_index]
            Y_test = Y[test_index]
            # standardization
            X_train_standard, X_test_standard =\
                    _standardization(X_train,
                                     columns_to_standardize=columns_to_standardize,
                                     X_test=X_test)
            # make predictions
            LR = LogisticRegression(penalty=penalty, C=c, solver='liblinear')
            LR_fit = LR.fit(X_train_standard, Y_train)
            # out of sample prediction
            Y_test_pred = LR.predict(X_test_standard)
            accuracy = accuracy + _accuracy(Y_test, Y_test_pred)
        accuracy_list.append(accuracy / (n_splits - 1))
    result = max(zip(accuracy_list, Cs))
    accuracy = result[0]
    c = result[1]
    return accuracy_list, accuracy, c


def logistic_Pred(X, Y, c, penalty='l1',
                  columns_to_standardize=[], X_test=None,
                  Y_test=None):
    r""" Prediction using logistic regression.

    """
    # standardization
    X_train_standard, X_test_standard =\
            _standardization(X, columns_to_standardize=columns_to_standardize,
                             X_test=X_test)
    # make predictions
    LR = LogisticRegression(penalty=penalty, C=c, solver='liblinear')
    LR_fit = LR.fit(X_train_standard, Y)
    # in sample prediction
    Y_train_pred = LR.predict(X_train_standard)
    in_accuracy = _accuracy(Y, Y_train_pred)
    if X_test is not None and Y_test is not None:
        # out of sample prediction
        Y_test_pred = LR.predict(X_test_standard)
        out_accuracy = _accuracy(Y_test, Y_test_pred)
    else:
        Y_test_pred = None
        out_accuracy = None
    return Y_test_pred, out_accuracy, Y_train_pred, in_accuracy


def data_transform(data, data_format='percentage'):
    r""" Transform percentage/ratio to binary data.

    """
    data_copy = copy.deepcopy(data)
    if data_format is 'percentage':
        data_copy[data_copy < 0] = 0
        data_copy[data_copy > 0] = 1
    elif data_format is 'ratio':
        data_copy[data_copy < 1] = 0
        data_copy[data_copy > 1] = 1
    return data_copy


def lasso_CV(X, Y, Alphas=[], n_splits=10,
             columns_to_standardize=[], Y_format='percentage'):
    r""" Tune parameters in Lasso regression using CV.

    """
    folds = _KFolds_filter(X, n_splits=10)
    accuracy_list = []
    for alpha in Alphas:
        accuracy = 0
        for i in range(len(folds))[1:]:
            test_index = folds[i]
            train_index = [item for sublist in folds[:i] for item in sublist]
            X_train = X[train_index, :]
            X_test = X[test_index, :]
            Y_train = Y[train_index]
            Y_test = Y[test_index]
            # standardization
            X_train_standard, X_test_standard =\
                    _standardization(X_train,
                                     columns_to_standardize=columns_to_standardize,
                                     X_test=X_test)
            # make predictions
            LR = Lasso(alpha=alpha)
            LR_fit = LR.fit(X_train_standard, Y_train)
            # out of sample prediction
            Y_test_pred = LR.predict(X_test_standard)
            # transform Y to binary data before calculating accuracy.
            Y_test_transform = data_transform(Y_test, data_format=Y_format)
            Y_test_pred_transform = data_transform(Y_test_pred,
                                                    data_format=Y_format)
            accuracy = accuracy + _accuracy(Y_test_transform,
                                            Y_test_pred_transform)
        accuracy_list.append(accuracy / (n_splits - 1))
    result = max(zip(accuracy_list, Alphas))
    accuracy = result[0]
    alpha = result[1]
    return accuracy_list, accuracy, alpha


def lasso_Pred(X, Y, alpha, columns_to_standardize=[], X_test=None, Y_test=None, Y_format='percentage'):
    r""" Prediction using Lasso regression.

    """
    # standardization
    X_train_standard, X_test_standard =\
            _standardization(X, columns_to_standardize=columns_to_standardize,
                             X_test=X_test)
    # make predictions
    LR = Lasso(alpha=alpha)
    LR_fit = LR.fit(X_train_standard, Y)
    # in sample prediction
    Y_train_pred = LR.predict(X_train_standard)
    # transform Y to binary data before calculating accuracy.
    Y_train_transform = data_transform(Y, data_format=Y_format)
    Y_train_pred_transform = data_transform(Y_train_pred,
                                            data_format=Y_format)
    in_accuracy = _accuracy(Y_train_transform, Y_train_pred_transform)
    if X_test is not None and Y_test is not None:
        # out of sample prediction
        Y_test_pred = LR.predict(X_test_standard)
        # transform Y to binary data before calculating accuracy.
        Y_test_transform = data_transform(Y_test, data_format=Y_format)
        Y_test_pred_transform = data_transform(Y_test_pred,
                                                data_format=Y_format)
        out_accuracy = _accuracy(Y_test_transform, Y_test_pred_transform)
    else:
        Y_test_pred = None
        out_accuracy = None
    return Y_test_pred, out_accuracy, Y_train_pred, in_accuracy




def ridge_CV(X, Y, Alphas=[], n_splits=10, columns_to_standardize=[], Y_format='percentage'):
    r""" Tune parameters in Ridge regression using CV.

    """
    folds = _KFolds_filter(X, n_splits=10)
    accuracy_list = []
    for alpha in Alphas:
        accuracy = 0
        for i in range(len(folds))[1:]:
            test_index = folds[i]
            train_index = [item for sublist in folds[:i] for item in sublist]
            X_train = X[train_index, :]
            X_test = X[test_index, :]
            Y_train = Y[train_index]
            Y_test = Y[test_index]
            # standardization
            X_train_standard, X_test_standard =\
                    _standardization(X_train,
                                     columns_to_standardize=columns_to_standardize,
                                     X_test=X_test)
            # make predictions
            LR = Ridge(alpha=alpha)
            LR_fit = LR.fit(X_train_standard, Y_train)
            # out of sample prediction
            Y_test_pred = LR.predict(X_test_standard)
            # transform Y to binary data before calculating accuracy.
            Y_test_transform = data_transform(Y_test, data_format=Y_format)
            Y_test_pred_transform = data_transform(Y_test_pred,
                                                    data_format=Y_format)
            accuracy = accuracy + _accuracy(Y_test_transform,
                                            Y_test_pred_transform)
        accuracy_list.append(accuracy / (n_splits - 1))
    result = max(zip(accuracy_list, Alphas))
    accuracy = result[0]
    alpha = result[1]
    return accuracy_list, accuracy, alpha


def ridge_Pred(X, Y, alpha, columns_to_standardize=[],
               X_test=None, Y_test=None, Y_format='percentage'):
    r""" Prediction using Ridge regression.

    """
    # standardization
    X_train_standard, X_test_standard =\
            _standardization(X, columns_to_standardize=columns_to_standardize,
                             X_test=X_test)
    # make predictions
    LR = Ridge(alpha=alpha)
    LR_fit = LR.fit(X_train_standard, Y)
    # in sample prediction
    Y_train_pred = LR.predict(X_train_standard)
    # transform Y to binary data before calculating accuracy.
    Y_train_transform = data_transform(Y, data_format=Y_format)
    Y_train_pred_transform = data_transform(Y_train_pred,
                                            data_format=Y_format)
    in_accuracy = _accuracy(Y_train_transform, Y_train_pred_transform)
    if X_test is not None and Y_test is not None:
        # out of sample prediction
        Y_test_pred = LR.predict(X_test_standard)
        # transform Y to binary data before calculating accuracy.
        Y_test_transform = data_transform(Y_test, data_format=Y_format)
        Y_test_pred_transform = data_transform(Y_test_pred,
                                                data_format=Y_format)
        out_accuracy = _accuracy(Y_test_transform, Y_test_pred_transform)
    else:
        Y_test_pred = None
        out_accuracy = None
    return Y_test_pred, out_accuracy, Y_train_pred, in_accuracy


def MLPRegressor_CV(X, Y, hidden_layer_sizes=[], Alphas=[], n_splits=10, columns_to_standardize=[], Y_format='percentage'):
    r""" Tune parameters in MLP regressor using CV.

    """
    folds = _KFolds_filter(X, n_splits=10)
    accuracy_list = []
    for sizes in hidden_layer_sizes:
        for alpha in Alphas:
            accuracy = 0
            for i in range(len(folds))[1:]:
                test_index = folds[i]
                train_index = [item for sublist in folds[:i] for item in sublist]
                X_train = X[train_index, :]
                X_test = X[test_index, :]
                Y_train = Y[train_index]
                Y_test = Y[test_index]
                # standardization
                X_train_standard, X_test_standard =\
                        _standardization(X_train,
                                         columns_to_standardize=columns_to_standardize,
                                         X_test=X_test)
                # make predictions
                MLPR = MLPRegressor(hidden_layer_sizes=sizes, alpha=alpha,
                                  early_stopping=True)
                MLPR_fit = MLPR.fit(X_train_standard, Y_train)
                # out of sample prediction
                Y_test_pred = MLPR.predict(X_test_standard)
                # transform Y to binary data before calculating accuracy.
                Y_test_transform = data_transform(Y_test, data_format=Y_format)
                Y_test_pred_transform = data_transform(Y_test_pred,
                                                        data_format=Y_format)
                accuracy = accuracy + _accuracy(Y_test_transform,
                                                Y_test_pred_transform)
            accuracy_list.append(accuracy / (n_splits - 1))
    result = max(zip(accuracy_list, hidden_layer_sizes, Alphas))
    accuracy = result[0]
    sizes = result[1]
    alpha = result[2]
    return accuracy_list, accuracy, sizes, alpha


def MLPRegressor_Pred(X, Y, hidden_layer_sizes, alpha, columns_to_standardize=[],
               X_test=None, Y_test=None, Y_format='percentage'):
    r""" Prediction using MLP regressor.

    """
    # standardization
    X_train_standard, X_test_standard =\
            _standardization(X, columns_to_standardize=columns_to_standardize,
                             X_test=X_test)
    # make predictions
    MLPR = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha)
    MLPR_fit = MLPR.fit(X_train_standard, Y)
    # in sample prediction
    Y_train_pred = MLPR.predict(X_train_standard)
    # transform Y to binary data before calculating accuracy.
    Y_train_transform = data_transform(Y, data_format=Y_format)
    Y_train_pred_transform = data_transform(Y_train_pred,
                                            data_format=Y_format)
    in_accuracy = _accuracy(Y_train_transform, Y_train_pred_transform)
    if X_test is not None and Y_test is not None:
        # out of sample prediction
        Y_test_pred = MLPR.predict(X_test_standard)
        # transform Y to binary data before calculating accuracy.
        Y_test_transform = data_transform(Y_test, data_format=Y_format)
        Y_test_pred_transform = data_transform(Y_test_pred,
                                                data_format=Y_format)
        out_accuracy = _accuracy(Y_test_transform, Y_test_pred_transform)
    else:
        Y_test_pred = None
        out_accuracy = None
    return Y_test_pred, out_accuracy, Y_train_pred, in_accuracy


def MLPClassifier_CV(X, Y, hidden_layer_sizes=[], Alphas=[], n_splits=10, columns_to_standardize=[]):
    r""" Tune parameters in MLP classifier using CV.

    """
    folds = _KFolds_filter(X, n_splits=10)
    accuracy_list = []
    for sizes in hidden_layer_sizes:
        for alpha in Alphas:
            accuracy = 0
            for i in range(len(folds))[1:]:
                test_index = folds[i]
                train_index = [item for sublist in folds[:i] for item in sublist]
                X_train = X[train_index, :]
                X_test = X[test_index, :]
                Y_train = Y[train_index]
                Y_test = Y[test_index]
                # standardization
                X_train_standard, X_test_standard =\
                        _standardization(X_train,
                                         columns_to_standardize=columns_to_standardize,
                                         X_test=X_test)
                # make predictions
                MLPC = MLPClassifier(hidden_layer_sizes=sizes, alpha=alpha,
                                  early_stopping=True)
                MLPC_fit = MLPC.fit(X_train_standard, Y_train)
                # out of sample prediction
                Y_test_pred = MLPC.predict(X_test_standard)
                accuracy = accuracy + _accuracy(Y_test,
                                                Y_test_pred)
            accuracy_list.append(accuracy / (n_splits - 1))
    result = max(zip(accuracy_list, hidden_layer_sizes, Alphas))
    accuracy = result[0]
    sizes = result[1]
    alpha = result[2]
    return accuracy_list, accuracy, sizes, alpha


def MLPClassifier_Pred(X, Y, hidden_layer_sizes, alpha, columns_to_standardize=[],
               X_test=None, Y_test=None):
    r""" Prediction using MLP classifier.

    """
    # standardization
    X_train_standard, X_test_standard =\
            _standardization(X, columns_to_standardize=columns_to_standardize,
                             X_test=X_test)
    # make predictions
    MLPC = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha)
    MLPC_fit = MLPC.fit(X_train_standard, Y)
    # in sample prediction
    Y_train_pred = MLPC.predict(X_train_standard)
    in_accuracy = _accuracy(Y, Y_train_pred)
    if X_test is not None and Y_test is not None:
        # out of sample prediction
        Y_test_pred = MLPC.predict(X_test_standard)
        out_accuracy = _accuracy(Y_test, Y_test_pred)
    else:
        Y_test_pred = None
        out_accuracy = None
    return Y_test_pred, out_accuracy, Y_train_pred, in_accuracy


def trading_strategy(data, predicted_direction, columns=['Open', 'Close']):
    r""" Calculate cumulative daily profit-and-loss (P&L) using the strategy
    based on prediction.

    """
    direction = copy.deepcopy(predicted_direction)
    direction[direction == 0] = -1
    daily = (data[columns[1]][1:] - data[columns[0]][1:]) /\
            data[columns[0]][1:] * direction[:-1]
    cumsum_PL = daily.cumsum()
    return daily, cumsum_PL


def SPY_longOnly_strategy(data, columns=['Open', 'Close']):
    r""" Calculate cumulative daily profit-and-loss (P&L) using SPY long-only
    strategy.

    """
    daily = 9 * (data[columns[1]][1:] - data[columns[0]][1:]) /\
            data[columns[0]][1:]
    cumsum_PL = daily.cumsum()
    return daily, cumsum_PL


def all_longOnly_strategy(data, columns=['Open', 'Close']):
    r""" Calculate cumulative daily profit-and-loss (P&L) using all long-only
    strategy.

    """
    daily = (data[columns[1]][1:] - data[columns[0]][1:]) /\
            data[columns[0]][1:]
    cumsum_PL = daily.cumsum()
    return daily, cumsum_PL


def annualized_sharpe(data):
    r""" Calculate the annualized sharpe ratio.

    """
    results = data.mean() / data.std() * np.sqrt(252)
    return results

