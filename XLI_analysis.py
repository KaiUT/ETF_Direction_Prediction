import copy
import pandas_datareader.data as web
import pandas as pd
import datetime as dt
#  import matplotlib.pyplot as plt
import functions

# collect data
start = dt.datetime(2000, 1, 1)
end = dt.datetime(2017, 1, 1)
XLI = web.DataReader('XLI', 'yahoo', start, end).iloc[:, range(5)]


# -------------- XLI direction prediction --------------------------------

# plot raw data
functions.plot_raw_data(XLI, fig_path='/Users/kailiu/GitProjects/ETF_Direction_Prediction/results/XLI_raw.pdf')

###################
# data processing #
###################

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


# ====================== Data-driven models ================================

# choose features - OHLC and weekdays
indices = range(5) + range(32)[27:]
X_XLI_train = XLI_train.iloc[:, indices].values
X_XLI_test = XLI_test.iloc[:, indices].values


#################################
# prediction direction directly #
#################################

# Direction_Binary as Y
Y_XLI_train = XLI_train.loc[:, 'Direction_Binary']
Y_XLI_test = XLI_test.loc[:, 'Direction_Binary']


# MODEL1: Logistic regression with L1 penalty #

accuracy_list, accuracy, c = \
        functions.logistic_CV(X_XLI_train, Y_XLI_train, n_splits=10,
                              columns_to_standardize=range(5),
                              Cs=[2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                                  100], penalty='l1')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.logistic_Pred(X_XLI_train, Y_XLI_train,
                                c, columns_to_standardize=range(5),
                                penalty='l1', X_test=X_XLI_test,
                                Y_test=Y_XLI_test)


# MODEL2: Logistic regression with L2 penalty #

accuracy_list, accuracy, c = \
        functions.logistic_CV(X_XLI_train, Y_XLI_train, n_splits=10,
                              columns_to_standardize=range(5),
                              Cs=[2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                                  100], penalty='l2')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.logistic_Pred(X_XLI_train, Y_XLI_train,
                                c, columns_to_standardize=range(5),
                                penalty='l2', X_test=X_XLI_test,
                                Y_test=Y_XLI_test)


# MODEL3: ANN: Multi-layer Perceptron classifier

accuracy_list, accuracy, hidden_layer_sizes, alpha = \
        functions.MLPClassifier_CV(X_XLI_train, Y_XLI_train, n_splits=10,
                            hidden_layer_sizes = [(9), (6), (3)],
                           columns_to_standardize=range(5),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100])


Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.MLPClassifier_Pred(X_XLI_train, Y_XLI_train,
                                    hidden_layer_sizes, alpha,
                             columns_to_standardize=range(5),
                             X_test=X_XLI_test, Y_test=Y_XLI_test)



###########################################
# prediction percentage (close-open)/open #
###########################################

# Direction_percentage as Y
Y_XLI_train = XLI_train.loc[:, 'Direction_percentage']
Y_XLI_test = XLI_test.loc[:, 'Direction_percentage']

# MODEL1: LASSO regression

accuracy_list, accuracy, alpha =\
        functions.lasso_CV(X_XLI_train, Y_XLI_train, n_splits=10,
                           columns_to_standardize=range(5),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='percentage')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.lasso_Pred(X_XLI_train, Y_XLI_train, alpha,
                            columns_to_standardize=range(5),
                            X_test=X_XLI_test, Y_test=Y_XLI_test,
                            Y_format='percentage')

# MODEL2: Ridge regression

accuracy_list, accuracy, alpha = \
        functions.ridge_CV(X_XLI_train, Y_XLI_train, n_splits=10,
                           columns_to_standardize=range(5),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='percentage')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.ridge_Pred(X_XLI_train, Y_XLI_train, alpha,
                             columns_to_standardize=range(5),
                             X_test=X_XLI_test, Y_test=Y_XLI_test,
                             Y_format='percentage')

# MODEL3: ANN: Multi-layer Perceptron regressor

accuracy_list, accuracy, hidden_layer_sizes, alpha = \
        functions.MLPRegressor_CV(X_XLI_train, Y_XLI_train, n_splits=10,
                            hidden_layer_sizes = [(9), (6), (3)],
                           columns_to_standardize=range(5),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='percentage')


Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.MLPRegressor_Pred(X_XLI_train, Y_XLI_train,
                                    hidden_layer_sizes, alpha,
                             columns_to_standardize=range(5),
                             X_test=X_XLI_test, Y_test=Y_XLI_test,
                             Y_format='percentage')



###############################
# prediction ratio close/open #
###############################

# Direction_ratio as Y
Y_XLI_train = XLI_train.loc[:, 'Direction_ratio']
Y_XLI_test = XLI_test.loc[:, 'Direction_ratio']

# MODEL1: LASSO regression

accuracy_list, accuracy, alpha =\
        functions.lasso_CV(X_XLI_train, Y_XLI_train, n_splits=10,
                           columns_to_standardize=range(5),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='ratio')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.lasso_Pred(X_XLI_train, Y_XLI_train, alpha,
                            columns_to_standardize=range(5),
                            X_test=X_XLI_test, Y_test=Y_XLI_test,
                            Y_format='ratio')

# MODEL2: Ridge regression

accuracy_list, accuracy, alpha = \
        functions.ridge_CV(X_XLI_train, Y_XLI_train, n_splits=10,
                           columns_to_standardize=range(5),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='ratio')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.ridge_Pred(X_XLI_train, Y_XLI_train, alpha,
                             columns_to_standardize=range(5),
                             X_test=X_XLI_test, Y_test=Y_XLI_test,
                             Y_format='ratio')

# MODEL3: ANN: Multi-layer Perceptron regressor

accuracy_list, accuracy, hidden_layer_sizes, alpha = \
        functions.MLPRegressor_CV(X_XLI_train, Y_XLI_train, n_splits=10,
                            hidden_layer_sizes = [(9), (6), (3)],
                           columns_to_standardize=range(5),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='ratio')


Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.MLPRegressor_Pred(X_XLI_train, Y_XLI_train,
                                    hidden_layer_sizes, alpha,
                             columns_to_standardize=range(5),
                             X_test=X_XLI_test, Y_test=Y_XLI_test,
                             Y_format='ratio')



# ====================== Technical indicators models ==========================

# choose features - indicators and weekdays
indices = range(32)[8:]
X_XLI_train = XLI_train.iloc[:, indices].values
X_XLI_test = XLI_test.iloc[:, indices].values


#################################
# prediction direction directly #
#################################

# Direction_Binary as Y
Y_XLI_train = XLI_train.loc[:, 'Direction_Binary']
Y_XLI_test = XLI_test.loc[:, 'Direction_Binary']


# MODEL1: Logistic regression with L1 penalty #

accuracy_list, accuracy, c = \
        functions.logistic_CV(X_XLI_train, Y_XLI_train, n_splits=10,
                              columns_to_standardize=range(19),
                              Cs=[2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                                  100], penalty='l1')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.logistic_Pred(X_XLI_train, Y_XLI_train,
                                c, columns_to_standardize=range(19),
                                penalty='l1', X_test=X_XLI_test,
                                Y_test=Y_XLI_test)


# MODEL2: Logistic regression with L2 penalty #

accuracy_list, accuracy, c = \
        functions.logistic_CV(X_XLI_train, Y_XLI_train, n_splits=10,
                              columns_to_standardize=range(19),
                              Cs=[2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                                  100], penalty='l2')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.logistic_Pred(X_XLI_train, Y_XLI_train,
                                c, columns_to_standardize=range(19),
                                penalty='l2', X_test=X_XLI_test,
                                Y_test=Y_XLI_test)


# MODEL3: ANN: Multi-layer Perceptron classifier

accuracy_list, accuracy, hidden_layer_sizes, alpha = \
        functions.MLPClassifier_CV(X_XLI_train, Y_XLI_train, n_splits=10,
                            hidden_layer_sizes = [(9), (6), (3)],
                           columns_to_standardize=range(19),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100])


Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.MLPClassifier_Pred(X_XLI_train, Y_XLI_train,
                                    hidden_layer_sizes, alpha,
                             columns_to_standardize=range(19),
                             X_test=X_XLI_test, Y_test=Y_XLI_test)



###########################################
# prediction percentage (close-open)/open #
###########################################

# Direction_percentage as Y
Y_XLI_train = XLI_train.loc[:, 'Direction_percentage']
Y_XLI_test = XLI_test.loc[:, 'Direction_percentage']

# MODEL1: LASSO regression

accuracy_list, accuracy, alpha =\
        functions.lasso_CV(X_XLI_train, Y_XLI_train, n_splits=10,
                           columns_to_standardize=range(19),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='percentage')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.lasso_Pred(X_XLI_train, Y_XLI_train, alpha,
                            columns_to_standardize=range(19),
                            X_test=X_XLI_test, Y_test=Y_XLI_test,
                            Y_format='percentage')

# MODEL2: Ridge regression

accuracy_list, accuracy, alpha = \
        functions.ridge_CV(X_XLI_train, Y_XLI_train, n_splits=10,
                           columns_to_standardize=range(19),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='percentage')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.ridge_Pred(X_XLI_train, Y_XLI_train, alpha,
                             columns_to_standardize=range(19),
                             X_test=X_XLI_test, Y_test=Y_XLI_test,
                             Y_format='percentage')

# MODEL3: ANN: Multi-layer Perceptron regressor

accuracy_list, accuracy, hidden_layer_sizes, alpha = \
        functions.MLPRegressor_CV(X_XLI_train, Y_XLI_train, n_splits=10,
                            hidden_layer_sizes = [(9), (6), (3)],
                           columns_to_standardize=range(19),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='percentage')


Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.MLPRegressor_Pred(X_XLI_train, Y_XLI_train,
                                    hidden_layer_sizes, alpha,
                             columns_to_standardize=range(19),
                             X_test=X_XLI_test, Y_test=Y_XLI_test,
                             Y_format='percentage')



###############################
# prediction ratio close/open #
###############################

# Direction_ratio as Y
Y_XLI_train = XLI_train.loc[:, 'Direction_ratio']
Y_XLI_test = XLI_test.loc[:, 'Direction_ratio']

# MODEL1: LASSO regression

accuracy_list, accuracy, alpha =\
        functions.lasso_CV(X_XLI_train, Y_XLI_train, n_splits=10,
                           columns_to_standardize=range(19),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='ratio')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.lasso_Pred(X_XLI_train, Y_XLI_train, alpha,
                            columns_to_standardize=range(19),
                            X_test=X_XLI_test, Y_test=Y_XLI_test,
                            Y_format='ratio')

# MODEL2: Ridge regression

accuracy_list, accuracy, alpha = \
        functions.ridge_CV(X_XLI_train, Y_XLI_train, n_splits=10,
                           columns_to_standardize=range(19),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='ratio')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.ridge_Pred(X_XLI_train, Y_XLI_train, alpha,
                             columns_to_standardize=range(19),
                             X_test=X_XLI_test, Y_test=Y_XLI_test,
                             Y_format='ratio')

# MODEL3: ANN: Multi-layer Perceptron regressor

accuracy_list, accuracy, hidden_layer_sizes, alpha = \
        functions.MLPRegressor_CV(X_XLI_train, Y_XLI_train, n_splits=10,
                            hidden_layer_sizes = [(9), (6), (3)],
                           columns_to_standardize=range(19),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='ratio')


Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.MLPRegressor_Pred(X_XLI_train, Y_XLI_train,
                                    hidden_layer_sizes, alpha,
                             columns_to_standardize=range(19),
                             X_test=X_XLI_test, Y_test=Y_XLI_test,
                             Y_format='ratio')



# ====================== Subset of Technical indicators models ==========================

# choose features - indicators and weekdays
indices = range(32)[14:]
X_XLI_train = XLI_train.iloc[:, indices].values
X_XLI_test = XLI_test.iloc[:, indices].values


#################################
# prediction direction directly #
#################################

# Direction_Binary as Y
Y_XLI_train = XLI_train.loc[:, 'Direction_Binary']
Y_XLI_test = XLI_test.loc[:, 'Direction_Binary']


# MODEL1: Logistic regression with L1 penalty #

accuracy_list, accuracy, c = \
        functions.logistic_CV(X_XLI_train, Y_XLI_train, n_splits=10,
                              columns_to_standardize=range(13),
                              Cs=[2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                                  100], penalty='l1')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.logistic_Pred(X_XLI_train, Y_XLI_train,
                                c, columns_to_standardize=range(13),
                                penalty='l1', X_test=X_XLI_test,
                                Y_test=Y_XLI_test)


# MODEL2: Logistic regression with L2 penalty #

accuracy_list, accuracy, c = \
        functions.logistic_CV(X_XLI_train, Y_XLI_train, n_splits=10,
                              columns_to_standardize=range(13),
                              Cs=[2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                                  100], penalty='l2')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.logistic_Pred(X_XLI_train, Y_XLI_train,
                                c, columns_to_standardize=range(13),
                                penalty='l2', X_test=X_XLI_test,
                                Y_test=Y_XLI_test)


# MODEL3: ANN: Multi-layer Perceptron classifier

accuracy_list, accuracy, hidden_layer_sizes, alpha = \
        functions.MLPClassifier_CV(X_XLI_train, Y_XLI_train, n_splits=10,
                            hidden_layer_sizes = [(9), (6), (3)],
                           columns_to_standardize=range(13),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100])


Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.MLPClassifier_Pred(X_XLI_train, Y_XLI_train,
                                    hidden_layer_sizes, alpha,
                             columns_to_standardize=range(13),
                             X_test=X_XLI_test, Y_test=Y_XLI_test)



###########################################
# prediction percentage (close-open)/open #
###########################################

# Direction_percentage as Y
Y_XLI_train = XLI_train.loc[:, 'Direction_percentage']
Y_XLI_test = XLI_test.loc[:, 'Direction_percentage']

# MODEL1: LASSO regression

accuracy_list, accuracy, alpha =\
        functions.lasso_CV(X_XLI_train, Y_XLI_train, n_splits=10,
                           columns_to_standardize=range(13),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='percentage')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.lasso_Pred(X_XLI_train, Y_XLI_train, alpha,
                            columns_to_standardize=range(13),
                            X_test=X_XLI_test, Y_test=Y_XLI_test,
                            Y_format='percentage')

# MODEL2: Ridge regression

accuracy_list, accuracy, alpha = \
        functions.ridge_CV(X_XLI_train, Y_XLI_train, n_splits=10,
                           columns_to_standardize=range(13),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='percentage')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.ridge_Pred(X_XLI_train, Y_XLI_train, alpha,
                             columns_to_standardize=range(13),
                             X_test=X_XLI_test, Y_test=Y_XLI_test,
                             Y_format='percentage')

# MODEL3: ANN: Multi-layer Perceptron regressor

accuracy_list, accuracy, hidden_layer_sizes, alpha = \
        functions.MLPRegressor_CV(X_XLI_train, Y_XLI_train, n_splits=10,
                            hidden_layer_sizes = [(9), (6), (3)],
                           columns_to_standardize=range(13),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='percentage')


Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.MLPRegressor_Pred(X_XLI_train, Y_XLI_train,
                                    hidden_layer_sizes, alpha,
                             columns_to_standardize=range(13),
                             X_test=X_XLI_test, Y_test=Y_XLI_test,
                             Y_format='percentage')



###############################
# prediction ratio close/open #
###############################

# Direction_ratio as Y
Y_XLI_train = XLI_train.loc[:, 'Direction_ratio']
Y_XLI_test = XLI_test.loc[:, 'Direction_ratio']

# MODEL1: LASSO regression

accuracy_list, accuracy, alpha =\
        functions.lasso_CV(X_XLI_train, Y_XLI_train, n_splits=10,
                           columns_to_standardize=range(13),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='ratio')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.lasso_Pred(X_XLI_train, Y_XLI_train, alpha,
                            columns_to_standardize=range(13),
                            X_test=X_XLI_test, Y_test=Y_XLI_test,
                            Y_format='ratio')

# MODEL2: Ridge regression

accuracy_list, accuracy, alpha = \
        functions.ridge_CV(X_XLI_train, Y_XLI_train, n_splits=10,
                           columns_to_standardize=range(13),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='ratio')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.ridge_Pred(X_XLI_train, Y_XLI_train, alpha,
                             columns_to_standardize=range(13),
                             X_test=X_XLI_test, Y_test=Y_XLI_test,
                             Y_format='ratio')

# MODEL3: ANN: Multi-layer Perceptron regressor

accuracy_list, accuracy, hidden_layer_sizes, alpha = \
        functions.MLPRegressor_CV(X_XLI_train, Y_XLI_train, n_splits=10,
                            hidden_layer_sizes = [(9), (6), (3)],
                           columns_to_standardize=range(13),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='ratio')


Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.MLPRegressor_Pred(X_XLI_train, Y_XLI_train,
                                    hidden_layer_sizes, alpha,
                             columns_to_standardize=range(13),
                             X_test=X_XLI_test, Y_test=Y_XLI_test,
                             Y_format='ratio')
























