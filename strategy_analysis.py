import copy
import pandas_datareader.data as web
import pandas as pd
import datetime as dt
#  import matplotlib.pyplot as plt
import functions

# collect data
start = dt.datetime(2000, 1, 1)
end = dt.datetime(2017, 1, 1)


# ================================= XLE ====================================

XLE = web.DataReader('XLE', 'yahoo', start, end).iloc[:, range(5)]

###################
# data processing #
###################

cl_XLE = copy.deepcopy(XLE)

# convert direction to binary
direction1 = cl_XLE['Close'] - cl_XLE['Open']
direction1[direction1 < 0] = 0
direction1[direction1 > 0] = 1
cl_XLE['Direction_Binary'] = direction1.shift(-1)
# convert direction to percentage
direction2 = (cl_XLE['Close'] - cl_XLE['Open']) / cl_XLE['Open']
cl_XLE['Direction_percentage'] = direction2.shift(-1)
# convert direction to ratio
direction3 = cl_XLE['Close'] / cl_XLE['Open']
cl_XLE['Direction_ratio'] = direction3.shift(-1)

# calculate technical indicators
cl_XLE = functions.EWMA(cl_XLE, period=7, min_periods=6, add_column='EWMA_7')
cl_XLE = functions.EWMA(cl_XLE, period=50, min_periods=49,
                        add_column='EWMA_50')
cl_XLE = functions.EWMA(cl_XLE, period=200, min_periods=199,
                        add_column='EWMA_200')
cl_XLE = functions.MACD(cl_XLE, min_periods=[11, 25])
cl_XLE = functions.RSI(cl_XLE, min_periods=13, add_column='RSI_14')
cl_XLE = functions.ADX(cl_XLE, min_periods=13, add_column='ADX_14')
cl_XLE, _, _ = functions.fast_stochastic_oscillator(cl_XLE, add_columns=['fast_%K14',
                                                                  'fast_%D14'])
cl_XLE = functions.slow_stochastic_oscillator(cl_XLE, add_columns=['slow_%K3',
                                                                  'slow_%D3'])
cl_XLE = functions.momentum(cl_XLE, add_column='Momentum4')
cl_XLE = functions.acceleration(cl_XLE)
cl_XLE = functions.williams_R(cl_XLE, add_column='Williams_%R14')
cl_XLE, _ = functions.accDist(cl_XLE)
cl_XLE = functions.chaikin_oscillator(cl_XLE, min_periods=[2,9])
cl_XLE = functions.william_AccDist(cl_XLE)
cl_XLE = functions.on_balance_volume(cl_XLE)
cl_XLE = functions.disparity_index(cl_XLE, min_periods=4,
                                   add_column='Disparity5')
cl_XLE = functions.disparity_index(cl_XLE, period=10, min_periods=9,
                                   add_column='Disparity10')
cl_XLE = functions.CCI(cl_XLE, add_column='CCI14')

cl_XLE.drop('slow_%K3', axis=1, inplace=True)
cl_XLE.dropna(inplace=True)  # drop rows including NaN

# add weekday info as predictors
weekdays = ['Mon', 'Tues', 'Wed', 'Thur', 'Fri']
for i in range(5):
    cl_XLE[weekdays[i]] = [0] * cl_XLE.shape[0]
    cl_XLE.loc[cl_XLE.index.weekday == i, weekdays[i]] = 1

# split train/test dateset
XLE_train = cl_XLE[cl_XLE.index < pd.to_datetime('2016-01-01')]
XLE_test = cl_XLE[cl_XLE.index >= pd.to_datetime('2016-01-01')]

# choose features - OHLC and weekdays
indices = range(5) + range(32)[27:]
X_XLE_train = XLE_train.iloc[:, indices].values
X_XLE_test = XLE_test.iloc[:, indices].values

# Direction_Binary as Y
Y_XLE_train = XLE_train.loc[:, 'Direction_Binary']
Y_XLE_test = XLE_test.loc[:, 'Direction_Binary']

# MODEL2: Logistic regression with L2 penalty #
XLE_Y_test_pred, XLE_out_accuracy, XLE_Y_train_pred, XLE_in_accuracy =\
        functions.logistic_Pred(X_XLE_train, Y_XLE_train, 5,
                                columns_to_standardize=range(5), penalty='l2',
                                X_test=X_XLE_test, Y_test=Y_XLE_test)


# ============================== XLU ======================================

XLU = web.DataReader('XLU', 'yahoo', start, end).iloc[:, range(5)]

###################
# data processing #
###################

cl_XLU = copy.deepcopy(XLU)

# convert direction to binary
direction1 = cl_XLU['Close'] - cl_XLU['Open']
direction1[direction1 < 0] = 0
direction1[direction1 > 0] = 1
cl_XLU['Direction_Binary'] = direction1.shift(-1)
# convert direction to percentage
direction2 = (cl_XLU['Close'] - cl_XLU['Open']) / cl_XLU['Open']
cl_XLU['Direction_percentage'] = direction2.shift(-1)
# convert direction to ratio
direction3 = cl_XLU['Close'] / cl_XLU['Open']
cl_XLU['Direction_ratio'] = direction3.shift(-1)

# calculate technical indicators
cl_XLU = functions.EWMA(cl_XLU, period=7, min_periods=6, add_column='EWMA_7')
cl_XLU = functions.EWMA(cl_XLU, period=50, min_periods=49,
                        add_column='EWMA_50')
cl_XLU = functions.EWMA(cl_XLU, period=200, min_periods=199,
                        add_column='EWMA_200')
cl_XLU = functions.MACD(cl_XLU, min_periods=[11, 25])
cl_XLU = functions.RSI(cl_XLU, min_periods=13, add_column='RSI_14')
cl_XLU = functions.ADX(cl_XLU, min_periods=13, add_column='ADX_14')
cl_XLU, _, _ = functions.fast_stochastic_oscillator(cl_XLU, add_columns=['fast_%K14',
                                                                  'fast_%D14'])
cl_XLU = functions.slow_stochastic_oscillator(cl_XLU, add_columns=['slow_%K3',
                                                                  'slow_%D3'])
cl_XLU = functions.momentum(cl_XLU, add_column='Momentum4')
cl_XLU = functions.acceleration(cl_XLU)
cl_XLU = functions.williams_R(cl_XLU, add_column='Williams_%R14')
cl_XLU, _ = functions.accDist(cl_XLU)
cl_XLU = functions.chaikin_oscillator(cl_XLU, min_periods=[2,9])
cl_XLU = functions.william_AccDist(cl_XLU)
cl_XLU = functions.on_balance_volume(cl_XLU)
cl_XLU = functions.disparity_index(cl_XLU, min_periods=4,
                                   add_column='Disparity5')
cl_XLU = functions.disparity_index(cl_XLU, period=10, min_periods=9,
                                   add_column='Disparity10')
cl_XLU = functions.CCI(cl_XLU, add_column='CCI14')

cl_XLU.drop('slow_%K3', axis=1, inplace=True)
cl_XLU.dropna(inplace=True)  # drop rows including NaN

# add weekday info as predictors
weekdays = ['Mon', 'Tues', 'Wed', 'Thur', 'Fri']
for i in range(5):
    cl_XLU[weekdays[i]] = [0] * cl_XLU.shape[0]
    cl_XLU.loc[cl_XLU.index.weekday == i, weekdays[i]] = 1

# split train/test dateset
XLU_train = cl_XLU[cl_XLU.index < pd.to_datetime('2016-01-01')]
XLU_test = cl_XLU[cl_XLU.index >= pd.to_datetime('2016-01-01')]

# choose features - OHLC and weekdays
indices = range(5) + range(32)[27:]
X_XLU_train = XLU_train.iloc[:, indices].values
X_XLU_test = XLU_test.iloc[:, indices].values

# Direction_percentage as Y
Y_XLU_train = XLU_train.loc[:, 'Direction_percentage']
Y_XLU_test = XLU_test.loc[:, 'Direction_percentage']


# MODEL3: ANN: Multi-layer Perceptron regressor
XLU_Y_test_pred, XLU_out_accuracy, XLU_Y_train_pred, XLU_in_accuracy =\
        functions.MLPRegressor_Pred(X_XLU_train, Y_XLU_train,
                                    (3), 5, columns_to_standardize=range(5),
                                    X_test=X_XLU_test, Y_test=Y_XLU_test,
                                    Y_format='percentage')
XLU_Y_test_pred = functions.data_transform(XLU_Y_test_pred,
                                           data_format='percentage')

# ============================= XLK ====================================

XLK = web.DataReader('XLK', 'yahoo', start, end).iloc[:, range(5)]

cl_XLK = copy.deepcopy(XLK)

# convert direction to binary
direction1 = cl_XLK['Close'] - cl_XLK['Open']
direction1[direction1 < 0] = 0
direction1[direction1 > 0] = 1
cl_XLK['Direction_Binary'] = direction1.shift(-1)
# convert direction to percentage
direction2 = (cl_XLK['Close'] - cl_XLK['Open']) / cl_XLK['Open']
cl_XLK['Direction_percentage'] = direction2.shift(-1)
# convert direction to ratio
direction3 = cl_XLK['Close'] / cl_XLK['Open']
cl_XLK['Direction_ratio'] = direction3.shift(-1)

# calculate technical indicators
cl_XLK = functions.EWMA(cl_XLK, period=7, min_periods=6, add_column='EWMA_7')
cl_XLK = functions.EWMA(cl_XLK, period=50, min_periods=49,
                        add_column='EWMA_50')
cl_XLK = functions.EWMA(cl_XLK, period=200, min_periods=199,
                        add_column='EWMA_200')
cl_XLK = functions.MACD(cl_XLK, min_periods=[11, 25])
cl_XLK = functions.RSI(cl_XLK, min_periods=13, add_column='RSI_14')
cl_XLK = functions.ADX(cl_XLK, min_periods=13, add_column='ADX_14')
cl_XLK, _, _ = functions.fast_stochastic_oscillator(cl_XLK, add_columns=['fast_%K14',
                                                                  'fast_%D14'])
cl_XLK = functions.slow_stochastic_oscillator(cl_XLK, add_columns=['slow_%K3',
                                                                  'slow_%D3'])
cl_XLK = functions.momentum(cl_XLK, add_column='Momentum4')
cl_XLK = functions.acceleration(cl_XLK)
cl_XLK = functions.williams_R(cl_XLK, add_column='Williams_%R14')
cl_XLK, _ = functions.accDist(cl_XLK)
cl_XLK = functions.chaikin_oscillator(cl_XLK, min_periods=[2,9])
cl_XLK = functions.william_AccDist(cl_XLK)
cl_XLK = functions.on_balance_volume(cl_XLK)
cl_XLK = functions.disparity_index(cl_XLK, min_periods=4,
                                   add_column='Disparity5')
cl_XLK = functions.disparity_index(cl_XLK, period=10, min_periods=9,
                                   add_column='Disparity10')
cl_XLK = functions.CCI(cl_XLK, add_column='CCI14')

cl_XLK.drop('slow_%K3', axis=1, inplace=True)
cl_XLK.dropna(inplace=True)  # drop rows including NaN

# add weekday info as predictors
weekdays = ['Mon', 'Tues', 'Wed', 'Thur', 'Fri']
for i in range(5):
    cl_XLK[weekdays[i]] = [0] * cl_XLK.shape[0]
    cl_XLK.loc[cl_XLK.index.weekday == i, weekdays[i]] = 1

# split train/test dateset
XLK_train = cl_XLK[cl_XLK.index < pd.to_datetime('2016-01-01')]
XLK_test = cl_XLK[cl_XLK.index >= pd.to_datetime('2016-01-01')]

# choose features - OHLC and weekdays
indices = range(5) + range(32)[27:]
X_XLK_train = XLK_train.iloc[:, indices].values
X_XLK_test = XLK_test.iloc[:, indices].values

# Direction_Binary as Y
Y_XLK_train = XLK_train.loc[:, 'Direction_Binary']
Y_XLK_test = XLK_test.loc[:, 'Direction_Binary']

# MODEL2: Logistic regression with L2 penalty #
XLK_Y_test_pred, XLK_out_accuracy, XLK_Y_train_pred, XLK_in_accuracy =\
        functions.logistic_Pred(X_XLK_train, Y_XLK_train,
                                2, columns_to_standardize=range(5),
                                penalty='l2', X_test=X_XLK_test,
                                Y_test=Y_XLK_test)

# ========================== XLB =====================================

XLB = web.DataReader('XLB', 'yahoo', start, end).iloc[:, range(5)]

cl_XLB = copy.deepcopy(XLB)

# convert direction to binary
direction1 = cl_XLB['Close'] - cl_XLB['Open']
direction1[direction1 < 0] = 0
direction1[direction1 > 0] = 1
cl_XLB['Direction_Binary'] = direction1.shift(-1)
# convert direction to percentage
direction2 = (cl_XLB['Close'] - cl_XLB['Open']) / cl_XLB['Open']
cl_XLB['Direction_percentage'] = direction2.shift(-1)
# convert direction to ratio
direction3 = cl_XLB['Close'] / cl_XLB['Open']
cl_XLB['Direction_ratio'] = direction3.shift(-1)

# calculate technical indicators
cl_XLB = functions.EWMA(cl_XLB, period=7, min_periods=6, add_column='EWMA_7')
cl_XLB = functions.EWMA(cl_XLB, period=50, min_periods=49,
                        add_column='EWMA_50')
cl_XLB = functions.EWMA(cl_XLB, period=200, min_periods=199,
                        add_column='EWMA_200')
cl_XLB = functions.MACD(cl_XLB, min_periods=[11, 25])
cl_XLB = functions.RSI(cl_XLB, min_periods=13, add_column='RSI_14')
cl_XLB = functions.ADX(cl_XLB, min_periods=13, add_column='ADX_14')
cl_XLB, _, _ = functions.fast_stochastic_oscillator(cl_XLB, add_columns=['fast_%K14',
                                                                  'fast_%D14'])
cl_XLB = functions.slow_stochastic_oscillator(cl_XLB, add_columns=['slow_%K3',
                                                                  'slow_%D3'])
cl_XLB = functions.momentum(cl_XLB, add_column='Momentum4')
cl_XLB = functions.acceleration(cl_XLB)
cl_XLB = functions.williams_R(cl_XLB, add_column='Williams_%R14')
cl_XLB, _ = functions.accDist(cl_XLB)
cl_XLB = functions.chaikin_oscillator(cl_XLB, min_periods=[2,9])
cl_XLB = functions.william_AccDist(cl_XLB)
cl_XLB = functions.on_balance_volume(cl_XLB)
cl_XLB = functions.disparity_index(cl_XLB, min_periods=4,
                                   add_column='Disparity5')
cl_XLB = functions.disparity_index(cl_XLB, period=10, min_periods=9,
                                   add_column='Disparity10')
cl_XLB = functions.CCI(cl_XLB, add_column='CCI14')

cl_XLB.drop('slow_%K3', axis=1, inplace=True)
cl_XLB.dropna(inplace=True)  # drop rows including NaN

# add weekday info as predictors
weekdays = ['Mon', 'Tues', 'Wed', 'Thur', 'Fri']
for i in range(5):
    cl_XLB[weekdays[i]] = [0] * cl_XLB.shape[0]
    cl_XLB.loc[cl_XLB.index.weekday == i, weekdays[i]] = 1

# split train/test dateset
XLB_train = cl_XLB[cl_XLB.index < pd.to_datetime('2016-01-01')]
XLB_test = cl_XLB[cl_XLB.index >= pd.to_datetime('2016-01-01')]

# choose features - indicators and weekdays
indices = range(32)[8:]
X_XLB_train = XLB_train.iloc[:, indices].values
X_XLB_test = XLB_test.iloc[:, indices].values

# Direction_ratio as Y
Y_XLB_train = XLB_train.loc[:, 'Direction_ratio']
Y_XLB_test = XLB_test.loc[:, 'Direction_ratio']

# MODEL3: ANN: Multi-layer Perceptron regressor
XLB_Y_test_pred, XLB_out_accuracy, XLB_Y_train_pred, XLB_in_accuracy =\
        functions.MLPRegressor_Pred(X_XLB_train, Y_XLB_train, (6), 1,
                                    columns_to_standardize=range(19),
                                    X_test=X_XLB_test, Y_test=Y_XLB_test,
                                    Y_format='ratio')
XLB_Y_test_pred = functions.data_transform(XLB_Y_test_pred,
                                           data_format='ratio')

# ============================== XLP ====================================

XLP = web.DataReader('XLP', 'yahoo', start, end).iloc[:, range(5)]

cl_XLP = copy.deepcopy(XLP)

# convert direction to binary
direction1 = cl_XLP['Close'] - cl_XLP['Open']
direction1[direction1 < 0] = 0
direction1[direction1 > 0] = 1
cl_XLP['Direction_Binary'] = direction1.shift(-1)
# convert direction to percentage
direction2 = (cl_XLP['Close'] - cl_XLP['Open']) / cl_XLP['Open']
cl_XLP['Direction_percentage'] = direction2.shift(-1)
# convert direction to ratio
direction3 = cl_XLP['Close'] / cl_XLP['Open']
cl_XLP['Direction_ratio'] = direction3.shift(-1)

# calculate technical indicators
cl_XLP = functions.EWMA(cl_XLP, period=7, min_periods=6, add_column='EWMA_7')
cl_XLP = functions.EWMA(cl_XLP, period=50, min_periods=49,
                        add_column='EWMA_50')
cl_XLP = functions.EWMA(cl_XLP, period=200, min_periods=199,
                        add_column='EWMA_200')
cl_XLP = functions.MACD(cl_XLP, min_periods=[11, 25])
cl_XLP = functions.RSI(cl_XLP, min_periods=13, add_column='RSI_14')
cl_XLP = functions.ADX(cl_XLP, min_periods=13, add_column='ADX_14')
cl_XLP, _, _ = functions.fast_stochastic_oscillator(cl_XLP, add_columns=['fast_%K14',
                                                                  'fast_%D14'])
cl_XLP = functions.slow_stochastic_oscillator(cl_XLP, add_columns=['slow_%K3',
                                                                  'slow_%D3'])
cl_XLP = functions.momentum(cl_XLP, add_column='Momentum4')
cl_XLP = functions.acceleration(cl_XLP)
cl_XLP = functions.williams_R(cl_XLP, add_column='Williams_%R14')
cl_XLP, _ = functions.accDist(cl_XLP)
cl_XLP = functions.chaikin_oscillator(cl_XLP, min_periods=[2,9])
cl_XLP = functions.william_AccDist(cl_XLP)
cl_XLP = functions.on_balance_volume(cl_XLP)
cl_XLP = functions.disparity_index(cl_XLP, min_periods=4,
                                   add_column='Disparity5')
cl_XLP = functions.disparity_index(cl_XLP, period=10, min_periods=9,
                                   add_column='Disparity10')
cl_XLP = functions.CCI(cl_XLP, add_column='CCI14')

cl_XLP.drop('slow_%K3', axis=1, inplace=True)
cl_XLP.dropna(inplace=True)  # drop rows including NaN

# add weekday info as predictors
weekdays = ['Mon', 'Tues', 'Wed', 'Thur', 'Fri']
for i in range(5):
    cl_XLP[weekdays[i]] = [0] * cl_XLP.shape[0]
    cl_XLP.loc[cl_XLP.index.weekday == i, weekdays[i]] = 1

# split train/test dateset
XLP_train = cl_XLP[cl_XLP.index < pd.to_datetime('2016-01-01')]
XLP_test = cl_XLP[cl_XLP.index >= pd.to_datetime('2016-01-01')]

# choose features - indicators and weekdays
indices = range(32)[8:]
X_XLP_train = XLP_train.iloc[:, indices].values
X_XLP_test = XLP_test.iloc[:, indices].values

# Direction_ratio as Y
Y_XLP_train = XLP_train.loc[:, 'Direction_ratio']
Y_XLP_test = XLP_test.loc[:, 'Direction_ratio']

# MODEL3: ANN: Multi-layer Perceptron regressor
XLP_Y_test_pred, XLP_out_accuracy, XLP_Y_train_pred, XLP_in_accuracy =\
        functions.MLPRegressor_Pred(X_XLP_train, Y_XLP_train, (9), 0.1,
                             columns_to_standardize=range(19),
                             X_test=X_XLP_test, Y_test=Y_XLP_test,
                             Y_format='ratio')
XLP_Y_test_pred = functions.data_transform(XLP_Y_test_pred,
                                           data_format='ratio')

# ============================== XLY =====================================

XLY = web.DataReader('XLY', 'yahoo', start, end).iloc[:, range(5)]

cl_XLY = copy.deepcopy(XLY)

# convert direction to binary
direction1 = cl_XLY['Close'] - cl_XLY['Open']
direction1[direction1 < 0] = 0
direction1[direction1 > 0] = 1
cl_XLY['Direction_Binary'] = direction1.shift(-1)
# convert direction to percentage
direction2 = (cl_XLY['Close'] - cl_XLY['Open']) / cl_XLY['Open']
cl_XLY['Direction_percentage'] = direction2.shift(-1)
# convert direction to ratio
direction3 = cl_XLY['Close'] / cl_XLY['Open']
cl_XLY['Direction_ratio'] = direction3.shift(-1)

# calculate technical indicators
cl_XLY = functions.EWMA(cl_XLY, period=7, min_periods=6, add_column='EWMA_7')
cl_XLY = functions.EWMA(cl_XLY, period=50, min_periods=49,
                        add_column='EWMA_50')
cl_XLY = functions.EWMA(cl_XLY, period=200, min_periods=199,
                        add_column='EWMA_200')
cl_XLY = functions.MACD(cl_XLY, min_periods=[11, 25])
cl_XLY = functions.RSI(cl_XLY, min_periods=13, add_column='RSI_14')
cl_XLY = functions.ADX(cl_XLY, min_periods=13, add_column='ADX_14')
cl_XLY, _, _ = functions.fast_stochastic_oscillator(cl_XLY, add_columns=['fast_%K14',
                                                                  'fast_%D14'])
cl_XLY = functions.slow_stochastic_oscillator(cl_XLY, add_columns=['slow_%K3',
                                                                  'slow_%D3'])
cl_XLY = functions.momentum(cl_XLY, add_column='Momentum4')
cl_XLY = functions.acceleration(cl_XLY)
cl_XLY = functions.williams_R(cl_XLY, add_column='Williams_%R14')
cl_XLY, _ = functions.accDist(cl_XLY)
cl_XLY = functions.chaikin_oscillator(cl_XLY, min_periods=[2,9])
cl_XLY = functions.william_AccDist(cl_XLY)
cl_XLY = functions.on_balance_volume(cl_XLY)
cl_XLY = functions.disparity_index(cl_XLY, min_periods=4,
                                   add_column='Disparity5')
cl_XLY = functions.disparity_index(cl_XLY, period=10, min_periods=9,
                                   add_column='Disparity10')
cl_XLY = functions.CCI(cl_XLY, add_column='CCI14')

cl_XLY.drop('slow_%K3', axis=1, inplace=True)
cl_XLY.dropna(inplace=True)  # drop rows including NaN

# add weekday info as predictors
weekdays = ['Mon', 'Tues', 'Wed', 'Thur', 'Fri']
for i in range(5):
    cl_XLY[weekdays[i]] = [0] * cl_XLY.shape[0]
    cl_XLY.loc[cl_XLY.index.weekday == i, weekdays[i]] = 1

# split train/test dateset
XLY_train = cl_XLY[cl_XLY.index < pd.to_datetime('2016-01-01')]
XLY_test = cl_XLY[cl_XLY.index >= pd.to_datetime('2016-01-01')]

# choose features - OHLC and weekdays
indices = range(5) + range(32)[27:]
X_XLY_train = XLY_train.iloc[:, indices].values
X_XLY_test = XLY_test.iloc[:, indices].values

# Direction_ratio as Y
Y_XLY_train = XLY_train.loc[:, 'Direction_ratio']
Y_XLY_test = XLY_test.loc[:, 'Direction_ratio']

# MODEL1: LASSO regression
XLY_Y_test_pred, XLY_out_accuracy, XLY_Y_train_pred, XLY_in_accuracy =\
        functions.lasso_Pred(X_XLY_train, Y_XLY_train, 100,
                            columns_to_standardize=range(5),
                            X_test=X_XLY_test, Y_test=Y_XLY_test,
                            Y_format='ratio')
XLY_Y_test_pred = functions.data_transform(XLY_Y_test_pred,
                                           data_format='ratio')

# ============================== XLI =======================================

XLI = web.DataReader('XLI', 'yahoo', start, end).iloc[:, range(5)]

cl_XLI = copy.deepcopy(XLI)

# convert direction to binary
direction1 = cl_XLI['Close'] - cl_XLI['Open']
direction1[direction1 < 0] = 0
direction1[direction1 > 0] = 1
cl_XLI['Direction_Binary'] = direction1.shift(-1)
# convert direction to percentage
direction2 = (cl_XLI['Close'] - cl_XLI['Open']) / cl_XLI['Open']
cl_XLI['Direction_percentage'] = direction2.shift(-1)
# convert direction to ratio
direction3 = cl_XLI['Close'] / cl_XLI['Open']
cl_XLI['Direction_ratio'] = direction3.shift(-1)

# calculate technical indicators
cl_XLI = functions.EWMA(cl_XLI, period=7, min_periods=6, add_column='EWMA_7')
cl_XLI = functions.EWMA(cl_XLI, period=50, min_periods=49,
                        add_column='EWMA_50')
cl_XLI = functions.EWMA(cl_XLI, period=200, min_periods=199,
                        add_column='EWMA_200')
cl_XLI = functions.MACD(cl_XLI, min_periods=[11, 25])
cl_XLI = functions.RSI(cl_XLI, min_periods=13, add_column='RSI_14')
cl_XLI = functions.ADX(cl_XLI, min_periods=13, add_column='ADX_14')
cl_XLI, _, _ = functions.fast_stochastic_oscillator(cl_XLI, add_columns=['fast_%K14',
                                                                  'fast_%D14'])
cl_XLI = functions.slow_stochastic_oscillator(cl_XLI, add_columns=['slow_%K3',
                                                                  'slow_%D3'])
cl_XLI = functions.momentum(cl_XLI, add_column='Momentum4')
cl_XLI = functions.acceleration(cl_XLI)
cl_XLI = functions.williams_R(cl_XLI, add_column='Williams_%R14')
cl_XLI, _ = functions.accDist(cl_XLI)
cl_XLI = functions.chaikin_oscillator(cl_XLI, min_periods=[2,9])
cl_XLI = functions.william_AccDist(cl_XLI)
cl_XLI = functions.on_balance_volume(cl_XLI)
cl_XLI = functions.disparity_index(cl_XLI, min_periods=4,
                                   add_column='Disparity5')
cl_XLI = functions.disparity_index(cl_XLI, period=10, min_periods=9,
                                   add_column='Disparity10')
cl_XLI = functions.CCI(cl_XLI, add_column='CCI14')

cl_XLI.drop('slow_%K3', axis=1, inplace=True)
cl_XLI.dropna(inplace=True)  # drop rows including NaN

# add weekday info as predictors
weekdays = ['Mon', 'Tues', 'Wed', 'Thur', 'Fri']
for i in range(5):
    cl_XLI[weekdays[i]] = [0] * cl_XLI.shape[0]
    cl_XLI.loc[cl_XLI.index.weekday == i, weekdays[i]] = 1

# split train/test dateset
XLI_train = cl_XLI[cl_XLI.index < pd.to_datetime('2016-01-01')]
XLI_test = cl_XLI[cl_XLI.index >= pd.to_datetime('2016-01-01')]

# choose features - OHLC and weekdays
indices = range(5) + range(32)[27:]
X_XLI_train = XLI_train.iloc[:, indices].values
X_XLI_test = XLI_test.iloc[:, indices].values

# Direction_Binary as Y
Y_XLI_train = XLI_train.loc[:, 'Direction_Binary']
Y_XLI_test = XLI_test.loc[:, 'Direction_Binary']

# MODEL1: Logistic regression with L1 penalty #
XLI_Y_test_pred, XLI_out_accuracy, XLI_Y_train_pred, XLI_in_accuracy =\
        functions.logistic_Pred(X_XLI_train, Y_XLI_train, 100,
                                columns_to_standardize=range(5),
                                penalty='l1', X_test=X_XLI_test,
                                Y_test=Y_XLI_test)

# ========================== XLV =========================================

XLV = web.DataReader('XLV', 'yahoo', start, end).iloc[:, range(5)]

cl_XLV = copy.deepcopy(XLV)

# convert direction to binary
direction1 = cl_XLV['Close'] - cl_XLV['Open']
direction1[direction1 < 0] = 0
direction1[direction1 > 0] = 1
cl_XLV['Direction_Binary'] = direction1.shift(-1)
# convert direction to percentage
direction2 = (cl_XLV['Close'] - cl_XLV['Open']) / cl_XLV['Open']
cl_XLV['Direction_percentage'] = direction2.shift(-1)
# convert direction to ratio
direction3 = cl_XLV['Close'] / cl_XLV['Open']
cl_XLV['Direction_ratio'] = direction3.shift(-1)

# calculate technical indicators
cl_XLV = functions.EWMA(cl_XLV, period=7, min_periods=6, add_column='EWMA_7')
cl_XLV = functions.EWMA(cl_XLV, period=50, min_periods=49,
                        add_column='EWMA_50')
cl_XLV = functions.EWMA(cl_XLV, period=200, min_periods=199,
                        add_column='EWMA_200')
cl_XLV = functions.MACD(cl_XLV, min_periods=[11, 25])
cl_XLV = functions.RSI(cl_XLV, min_periods=13, add_column='RSI_14')
cl_XLV = functions.ADX(cl_XLV, min_periods=13, add_column='ADX_14')
cl_XLV, _, _ = functions.fast_stochastic_oscillator(cl_XLV, add_columns=['fast_%K14',
                                                                  'fast_%D14'])
cl_XLV = functions.slow_stochastic_oscillator(cl_XLV, add_columns=['slow_%K3',
                                                                  'slow_%D3'])
cl_XLV = functions.momentum(cl_XLV, add_column='Momentum4')
cl_XLV = functions.acceleration(cl_XLV)
cl_XLV = functions.williams_R(cl_XLV, add_column='Williams_%R14')
cl_XLV, _ = functions.accDist(cl_XLV)
cl_XLV = functions.chaikin_oscillator(cl_XLV, min_periods=[2,9])
cl_XLV = functions.william_AccDist(cl_XLV)
cl_XLV = functions.on_balance_volume(cl_XLV)
cl_XLV = functions.disparity_index(cl_XLV, min_periods=4,
                                   add_column='Disparity5')
cl_XLV = functions.disparity_index(cl_XLV, period=10, min_periods=9,
                                   add_column='Disparity10')
cl_XLV = functions.CCI(cl_XLV, add_column='CCI14')

cl_XLV.drop('slow_%K3', axis=1, inplace=True)
cl_XLV.dropna(inplace=True)  # drop rows including NaN

# add weekday info as predictors
weekdays = ['Mon', 'Tues', 'Wed', 'Thur', 'Fri']
for i in range(5):
    cl_XLV[weekdays[i]] = [0] * cl_XLV.shape[0]
    cl_XLV.loc[cl_XLV.index.weekday == i, weekdays[i]] = 1

# split train/test dateset
XLV_train = cl_XLV[cl_XLV.index < pd.to_datetime('2016-01-01')]
XLV_test = cl_XLV[cl_XLV.index >= pd.to_datetime('2016-01-01')]

# choose features - OHLC and weekdays
indices = range(5) + range(32)[27:]
X_XLV_train = XLV_train.iloc[:, indices].values
X_XLV_test = XLV_test.iloc[:, indices].values

# Direction_percentage as Y
Y_XLV_train = XLV_train.loc[:, 'Direction_percentage']
Y_XLV_test = XLV_test.loc[:, 'Direction_percentage']

# MODEL2: Ridge regression
XLV_Y_test_pred, XLV_out_accuracy, XLV_Y_train_pred, XLV_in_accuracy =\
        functions.ridge_Pred(X_XLV_train, Y_XLV_train, 0.1,
                             columns_to_standardize=range(5),
                             X_test=X_XLV_test, Y_test=Y_XLV_test,
                             Y_format='percentage')
XLV_Y_test_pred = functions.data_transform(XLV_Y_test_pred,
                                           data_format='percentage')

# ============================== SPY =====================================

SPY = web.DataReader('SPY', 'yahoo', start, end).iloc[:, range(5)]

cl_SPY = copy.deepcopy(SPY)

# convert direction to binary
direction1 = cl_SPY['Close'] - cl_SPY['Open']
direction1[direction1 < 0] = 0
direction1[direction1 > 0] = 1
cl_SPY['Direction_Binary'] = direction1.shift(-1)
# convert direction to percentage
direction2 = (cl_SPY['Close'] - cl_SPY['Open']) / cl_SPY['Open']
cl_SPY['Direction_percentage'] = direction2.shift(-1)
# convert direction to ratio
direction3 = cl_SPY['Close'] / cl_SPY['Open']
cl_SPY['Direction_ratio'] = direction3.shift(-1)

# calculate technical indicators
cl_SPY = functions.EWMA(cl_SPY, period=7, min_periods=6, add_column='EWMA_7')
cl_SPY = functions.EWMA(cl_SPY, period=50, min_periods=49,
                        add_column='EWMA_50')
cl_SPY = functions.EWMA(cl_SPY, period=200, min_periods=199,
                        add_column='EWMA_200')
cl_SPY = functions.MACD(cl_SPY, min_periods=[11, 25])
cl_SPY = functions.RSI(cl_SPY, min_periods=13, add_column='RSI_14')
cl_SPY = functions.ADX(cl_SPY, min_periods=13, add_column='ADX_14')
cl_SPY, _, _ = functions.fast_stochastic_oscillator(cl_SPY, add_columns=['fast_%K14',
                                                                  'fast_%D14'])
cl_SPY = functions.slow_stochastic_oscillator(cl_SPY, add_columns=['slow_%K3',
                                                                  'slow_%D3'])
cl_SPY = functions.momentum(cl_SPY, add_column='Momentum4')
cl_SPY = functions.acceleration(cl_SPY)
cl_SPY = functions.williams_R(cl_SPY, add_column='Williams_%R14')
cl_SPY, _ = functions.accDist(cl_SPY)
cl_SPY = functions.chaikin_oscillator(cl_SPY, min_periods=[2,9])
cl_SPY = functions.william_AccDist(cl_SPY)
cl_SPY = functions.on_balance_volume(cl_SPY)
cl_SPY = functions.disparity_index(cl_SPY, min_periods=4,
                                   add_column='Disparity5')
cl_SPY = functions.disparity_index(cl_SPY, period=10, min_periods=9,
                                   add_column='Disparity10')
cl_SPY = functions.CCI(cl_SPY, add_column='CCI14')

cl_SPY.drop('slow_%K3', axis=1, inplace=True)
cl_SPY.dropna(inplace=True)  # drop rows including NaN

# add weekday info as predictors
weekdays = ['Mon', 'Tues', 'Wed', 'Thur', 'Fri']
for i in range(5):
    cl_SPY[weekdays[i]] = [0] * cl_SPY.shape[0]
    cl_SPY.loc[cl_SPY.index.weekday == i, weekdays[i]] = 1

# split train/test dateset
SPY_train = cl_SPY[cl_SPY.index < pd.to_datetime('2016-01-01')]
SPY_test = cl_SPY[cl_SPY.index >= pd.to_datetime('2016-01-01')]

# choose features - indicators and weekdays
indices = range(32)[8:]
X_SPY_train = SPY_train.iloc[:, indices].values
X_SPY_test = SPY_test.iloc[:, indices].values

# Direction_Binary as Y
Y_SPY_train = SPY_train.loc[:, 'Direction_Binary']
Y_SPY_test = SPY_test.loc[:, 'Direction_Binary']

# MODEL3: ANN: Multi-layer Perceptron classifier
SPY_Y_test_pred, SPY_out_accuracy, SPY_Y_train_pred, SPY_in_accuracy =\
        functions.MLPClassifier_Pred(X_SPY_train, Y_SPY_train, (6), 1,
                                     columns_to_standardize=range(19),
                                     X_test=X_SPY_test, Y_test=Y_SPY_test)


# ======================= Strategies analysis =============================


# strategy based on prediction
XLE_trading = functions.trading_strategy(XLE_test, XLE_Y_test_pred)
XLU_trading = functions.trading_strategy(XLU_test, XLU_Y_test_pred)
XLK_trading = functions.trading_strategy(XLU_test, XLK_Y_test_pred)
XLB_trading = functions.trading_strategy(XLU_test, XLB_Y_test_pred)
XLP_trading = functions.trading_strategy(XLU_test, XLP_Y_test_pred)
XLY_trading = functions.trading_strategy(XLU_test, XLY_Y_test_pred)
XLI_trading = functions.trading_strategy(XLU_test, XLI_Y_test_pred)
XLV_trading = functions.trading_strategy(XLU_test, XLV_Y_test_pred)
SPY_trading = functions.trading_strategy(XLU_test, SPY_Y_test_pred)

PL_trading = XLE_trading[0] + XLU_trading[0] + XLK_trading[0] + XLB_trading[0]\
        + XLP_trading[0] + XLY_trading[0] + XLI_trading[0] + XLV_trading[0] +\
        SPY_trading[0]

PL_trading_cum = XLE_trading[0] + XLU_trading[0] + XLK_trading[0] + XLB_trading[0]\
        + XLP_trading[0] + XLY_trading[0] + XLI_trading[0] + XLV_trading[0] +\
        SPY_trading[0]

# SPY long-only strategy
PL_SPY = functions.SPY_longOnly_strategy(SPY_test)

# All long-only strategy
XLE_allLong = functions.all_longOnly_strategy(XLE_test)
XLU_allLong = functions.all_longOnly_strategy(XLU_test)
XLK_allLong = functions.all_longOnly_strategy(XLU_test)
XLB_allLong = functions.all_longOnly_strategy(XLU_test)
XLP_allLong = functions.all_longOnly_strategy(XLU_test)
XLY_allLong = functions.all_longOnly_strategy(XLU_test)
XLI_allLong = functions.all_longOnly_strategy(XLU_test)
XLV_allLong = functions.all_longOnly_strategy(XLU_test)
SPY_allLong = functions.all_longOnly_strategy(XLU_test)

PL_allLong = XLE_allLong[0] + XLU_allLong[0] + XLK_allLong[0] + XLB_allLong[0]\
        + XLP_allLong[0] + XLY_allLong[0] + XLI_allLong[0] + XLV_allLong[0] +\
        SPY_allLong[0]

PL_allLong_cum = XLE_allLong[1] + XLU_allLong[1] + XLK_allLong[1] + XLB_allLong[1]\
        + XLP_allLong[1] + XLY_allLong[1] + XLI_allLong[1] + XLV_allLong[1] +\
        SPY_allLong[1]

# Plot equity curve
plt.figure(figsize=(12,7))
p1, = plt.plot(PL_trading_cum, label='Strategy based on prediction')
p2, = plt.plot(PL_SPY[1], label='SPY long-only strategy')
p3, = plt.plot(PL_allLong_cum, label='All long-only strategy')
plt.legend(handles=[p1, p2, p3], loc=4)
plt.xticks(rotation=45)
plt.savefig('/Users/kailiu/GitProjects/ETF_Direction_Prediction/results/equity.pdf')

# calculate annualized sharpe ratio

annualized_sharpe_1 = functions.annualized_sharpe(PL_trading)
annualized_sharpe_2 = functions.annualized_sharpe(PL_SPY[0])
annualized_sharpe_3 = functions.annualized_sharpe(PL_allLong)
