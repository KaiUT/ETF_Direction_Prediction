import copy
import pandas_datareader.data as web
import pandas as pd
import datetime as dt
#  import matplotlib.pyplot as plt
import functions

# collect data
start = dt.datetime(2000, 1, 1)
end = dt.datetime(2017, 1, 1)
XLK = web.DataReader('XLK', 'yahoo', start, end).iloc[:, range(5)]


# -------------- XLK direction prediction --------------------------------

# plot raw data
functions.plot_raw_data(XLK, fig_path='/Users/kailiu/GitProjects/ETF_Direction_Prediction/results/XLK_raw.pdf')

###################
# data processing #
###################

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


# ====================== Data-driven models ================================

# choose features - OHLC and weekdays
indices = range(5) + range(32)[27:]
X_XLK_train = XLK_train.iloc[:, indices].values
X_XLK_test = XLK_test.iloc[:, indices].values


#################################
# prediction direction directly #
#################################

# Direction_Binary as Y
Y_XLK_train = XLK_train.loc[:, 'Direction_Binary']
Y_XLK_test = XLK_test.loc[:, 'Direction_Binary']


# MODEL1: Logistic regression with L1 penalty #

accuracy_list, accuracy, c = \
        functions.logistic_CV(X_XLK_train, Y_XLK_train, n_splits=10,
                              columns_to_standardize=range(5),
                              Cs=[2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                                  100], penalty='l1')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.logistic_Pred(X_XLK_train, Y_XLK_train,
                                c, columns_to_standardize=range(5),
                                penalty='l1', X_test=X_XLK_test,
                                Y_test=Y_XLK_test)


# MODEL2: Logistic regression with L2 penalty #

accuracy_list, accuracy, c = \
        functions.logistic_CV(X_XLK_train, Y_XLK_train, n_splits=10,
                              columns_to_standardize=range(5),
                              Cs=[2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                                  100], penalty='l2')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.logistic_Pred(X_XLK_train, Y_XLK_train,
                                c, columns_to_standardize=range(5),
                                penalty='l2', X_test=X_XLK_test,
                                Y_test=Y_XLK_test)


# MODEL3: ANN: Multi-layer Perceptron classifier

accuracy_list, accuracy, hidden_layer_sizes, alpha = \
        functions.MLPClassifier_CV(X_XLK_train, Y_XLK_train, n_splits=10,
                            hidden_layer_sizes = [(9), (6), (3)],
                           columns_to_standardize=range(5),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100])


Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.MLPClassifier_Pred(X_XLK_train, Y_XLK_train,
                                    hidden_layer_sizes, alpha,
                             columns_to_standardize=range(5),
                             X_test=X_XLK_test, Y_test=Y_XLK_test)



###########################################
# prediction percentage (close-open)/open #
###########################################

# Direction_percentage as Y
Y_XLK_train = XLK_train.loc[:, 'Direction_percentage']
Y_XLK_test = XLK_test.loc[:, 'Direction_percentage']

# MODEL1: LASSO regression

accuracy_list, accuracy, alpha =\
        functions.lasso_CV(X_XLK_train, Y_XLK_train, n_splits=10,
                           columns_to_standardize=range(5),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='percentage')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.lasso_Pred(X_XLK_train, Y_XLK_train, alpha,
                            columns_to_standardize=range(5),
                            X_test=X_XLK_test, Y_test=Y_XLK_test,
                            Y_format='percentage')

# MODEL2: Ridge regression

accuracy_list, accuracy, alpha = \
        functions.ridge_CV(X_XLK_train, Y_XLK_train, n_splits=10,
                           columns_to_standardize=range(5),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='percentage')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.ridge_Pred(X_XLK_train, Y_XLK_train, alpha,
                             columns_to_standardize=range(5),
                             X_test=X_XLK_test, Y_test=Y_XLK_test,
                             Y_format='percentage')

# MODEL3: ANN: Multi-layer Perceptron regressor

accuracy_list, accuracy, hidden_layer_sizes, alpha = \
        functions.MLPRegressor_CV(X_XLK_train, Y_XLK_train, n_splits=10,
                            hidden_layer_sizes = [(9), (6), (3)],
                           columns_to_standardize=range(5),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='percentage')


Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.MLPRegressor_Pred(X_XLK_train, Y_XLK_train,
                                    hidden_layer_sizes, alpha,
                             columns_to_standardize=range(5),
                             X_test=X_XLK_test, Y_test=Y_XLK_test,
                             Y_format='percentage')



###############################
# prediction ratio close/open #
###############################

# Direction_ratio as Y
Y_XLK_train = XLK_train.loc[:, 'Direction_ratio']
Y_XLK_test = XLK_test.loc[:, 'Direction_ratio']

# MODEL1: LASSO regression

accuracy_list, accuracy, alpha =\
        functions.lasso_CV(X_XLK_train, Y_XLK_train, n_splits=10,
                           columns_to_standardize=range(5),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='ratio')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.lasso_Pred(X_XLK_train, Y_XLK_train, alpha,
                            columns_to_standardize=range(5),
                            X_test=X_XLK_test, Y_test=Y_XLK_test,
                            Y_format='ratio')

# MODEL2: Ridge regression

accuracy_list, accuracy, alpha = \
        functions.ridge_CV(X_XLK_train, Y_XLK_train, n_splits=10,
                           columns_to_standardize=range(5),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='ratio')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.ridge_Pred(X_XLK_train, Y_XLK_train, alpha,
                             columns_to_standardize=range(5),
                             X_test=X_XLK_test, Y_test=Y_XLK_test,
                             Y_format='ratio')

# MODEL3: ANN: Multi-layer Perceptron regressor

accuracy_list, accuracy, hidden_layer_sizes, alpha = \
        functions.MLPRegressor_CV(X_XLK_train, Y_XLK_train, n_splits=10,
                            hidden_layer_sizes = [(9), (6), (3)],
                           columns_to_standardize=range(5),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='ratio')


Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.MLPRegressor_Pred(X_XLK_train, Y_XLK_train,
                                    hidden_layer_sizes, alpha,
                             columns_to_standardize=range(5),
                             X_test=X_XLK_test, Y_test=Y_XLK_test,
                             Y_format='ratio')



# ====================== Technical indicators models ==========================

# choose features - indicators and weekdays
indices = range(32)[8:]
X_XLK_train = XLK_train.iloc[:, indices].values
X_XLK_test = XLK_test.iloc[:, indices].values


#################################
# prediction direction directly #
#################################

# Direction_Binary as Y
Y_XLK_train = XLK_train.loc[:, 'Direction_Binary']
Y_XLK_test = XLK_test.loc[:, 'Direction_Binary']


# MODEL1: Logistic regression with L1 penalty #

accuracy_list, accuracy, c = \
        functions.logistic_CV(X_XLK_train, Y_XLK_train, n_splits=10,
                              columns_to_standardize=range(19),
                              Cs=[2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                                  100], penalty='l1')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.logistic_Pred(X_XLK_train, Y_XLK_train,
                                c, columns_to_standardize=range(19),
                                penalty='l1', X_test=X_XLK_test,
                                Y_test=Y_XLK_test)


# MODEL2: Logistic regression with L2 penalty #

accuracy_list, accuracy, c = \
        functions.logistic_CV(X_XLK_train, Y_XLK_train, n_splits=10,
                              columns_to_standardize=range(19),
                              Cs=[2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                                  100], penalty='l2')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.logistic_Pred(X_XLK_train, Y_XLK_train,
                                c, columns_to_standardize=range(19),
                                penalty='l2', X_test=X_XLK_test,
                                Y_test=Y_XLK_test)


# MODEL3: ANN: Multi-layer Perceptron classifier

accuracy_list, accuracy, hidden_layer_sizes, alpha = \
        functions.MLPClassifier_CV(X_XLK_train, Y_XLK_train, n_splits=10,
                            hidden_layer_sizes = [(9), (6), (3)],
                           columns_to_standardize=range(19),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100])


Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.MLPClassifier_Pred(X_XLK_train, Y_XLK_train,
                                    hidden_layer_sizes, alpha,
                             columns_to_standardize=range(19),
                             X_test=X_XLK_test, Y_test=Y_XLK_test)



###########################################
# prediction percentage (close-open)/open #
###########################################

# Direction_percentage as Y
Y_XLK_train = XLK_train.loc[:, 'Direction_percentage']
Y_XLK_test = XLK_test.loc[:, 'Direction_percentage']

# MODEL1: LASSO regression

accuracy_list, accuracy, alpha =\
        functions.lasso_CV(X_XLK_train, Y_XLK_train, n_splits=10,
                           columns_to_standardize=range(19),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='percentage')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.lasso_Pred(X_XLK_train, Y_XLK_train, alpha,
                            columns_to_standardize=range(19),
                            X_test=X_XLK_test, Y_test=Y_XLK_test,
                            Y_format='percentage')

# MODEL2: Ridge regression

accuracy_list, accuracy, alpha = \
        functions.ridge_CV(X_XLK_train, Y_XLK_train, n_splits=10,
                           columns_to_standardize=range(19),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='percentage')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.ridge_Pred(X_XLK_train, Y_XLK_train, alpha,
                             columns_to_standardize=range(19),
                             X_test=X_XLK_test, Y_test=Y_XLK_test,
                             Y_format='percentage')

# MODEL3: ANN: Multi-layer Perceptron regressor

accuracy_list, accuracy, hidden_layer_sizes, alpha = \
        functions.MLPRegressor_CV(X_XLK_train, Y_XLK_train, n_splits=10,
                            hidden_layer_sizes = [(9), (6), (3)],
                           columns_to_standardize=range(19),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='percentage')


Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.MLPRegressor_Pred(X_XLK_train, Y_XLK_train,
                                    hidden_layer_sizes, alpha,
                             columns_to_standardize=range(19),
                             X_test=X_XLK_test, Y_test=Y_XLK_test,
                             Y_format='percentage')



###############################
# prediction ratio close/open #
###############################

# Direction_ratio as Y
Y_XLK_train = XLK_train.loc[:, 'Direction_ratio']
Y_XLK_test = XLK_test.loc[:, 'Direction_ratio']

# MODEL1: LASSO regression

accuracy_list, accuracy, alpha =\
        functions.lasso_CV(X_XLK_train, Y_XLK_train, n_splits=10,
                           columns_to_standardize=range(19),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='ratio')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.lasso_Pred(X_XLK_train, Y_XLK_train, alpha,
                            columns_to_standardize=range(19),
                            X_test=X_XLK_test, Y_test=Y_XLK_test,
                            Y_format='ratio')

# MODEL2: Ridge regression

accuracy_list, accuracy, alpha = \
        functions.ridge_CV(X_XLK_train, Y_XLK_train, n_splits=10,
                           columns_to_standardize=range(19),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='ratio')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.ridge_Pred(X_XLK_train, Y_XLK_train, alpha,
                             columns_to_standardize=range(19),
                             X_test=X_XLK_test, Y_test=Y_XLK_test,
                             Y_format='ratio')

# MODEL3: ANN: Multi-layer Perceptron regressor

accuracy_list, accuracy, hidden_layer_sizes, alpha = \
        functions.MLPRegressor_CV(X_XLK_train, Y_XLK_train, n_splits=10,
                            hidden_layer_sizes = [(9), (6), (3)],
                           columns_to_standardize=range(19),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='ratio')


Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.MLPRegressor_Pred(X_XLK_train, Y_XLK_train,
                                    hidden_layer_sizes, alpha,
                             columns_to_standardize=range(19),
                             X_test=X_XLK_test, Y_test=Y_XLK_test,
                             Y_format='ratio')



# ====================== Subset of Technical indicators models ==========================

# choose features - indicators and weekdays
indices = range(32)[14:]
X_XLK_train = XLK_train.iloc[:, indices].values
X_XLK_test = XLK_test.iloc[:, indices].values


#################################
# prediction direction directly #
#################################

# Direction_Binary as Y
Y_XLK_train = XLK_train.loc[:, 'Direction_Binary']
Y_XLK_test = XLK_test.loc[:, 'Direction_Binary']


# MODEL1: Logistic regression with L1 penalty #

accuracy_list, accuracy, c = \
        functions.logistic_CV(X_XLK_train, Y_XLK_train, n_splits=10,
                              columns_to_standardize=range(13),
                              Cs=[2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                                  100], penalty='l1')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.logistic_Pred(X_XLK_train, Y_XLK_train,
                                c, columns_to_standardize=range(13),
                                penalty='l1', X_test=X_XLK_test,
                                Y_test=Y_XLK_test)


# MODEL2: Logistic regression with L2 penalty #

accuracy_list, accuracy, c = \
        functions.logistic_CV(X_XLK_train, Y_XLK_train, n_splits=10,
                              columns_to_standardize=range(13),
                              Cs=[2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                                  100], penalty='l2')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.logistic_Pred(X_XLK_train, Y_XLK_train,
                                c, columns_to_standardize=range(13),
                                penalty='l2', X_test=X_XLK_test,
                                Y_test=Y_XLK_test)


# MODEL3: ANN: Multi-layer Perceptron classifier

accuracy_list, accuracy, hidden_layer_sizes, alpha = \
        functions.MLPClassifier_CV(X_XLK_train, Y_XLK_train, n_splits=10,
                            hidden_layer_sizes = [(9), (6), (3)],
                           columns_to_standardize=range(13),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100])


Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.MLPClassifier_Pred(X_XLK_train, Y_XLK_train,
                                    hidden_layer_sizes, alpha,
                             columns_to_standardize=range(13),
                             X_test=X_XLK_test, Y_test=Y_XLK_test)



###########################################
# prediction percentage (close-open)/open #
###########################################

# Direction_percentage as Y
Y_XLK_train = XLK_train.loc[:, 'Direction_percentage']
Y_XLK_test = XLK_test.loc[:, 'Direction_percentage']

# MODEL1: LASSO regression

accuracy_list, accuracy, alpha =\
        functions.lasso_CV(X_XLK_train, Y_XLK_train, n_splits=10,
                           columns_to_standardize=range(13),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='percentage')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.lasso_Pred(X_XLK_train, Y_XLK_train, alpha,
                            columns_to_standardize=range(13),
                            X_test=X_XLK_test, Y_test=Y_XLK_test,
                            Y_format='percentage')

# MODEL2: Ridge regression

accuracy_list, accuracy, alpha = \
        functions.ridge_CV(X_XLK_train, Y_XLK_train, n_splits=10,
                           columns_to_standardize=range(13),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='percentage')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.ridge_Pred(X_XLK_train, Y_XLK_train, alpha,
                             columns_to_standardize=range(13),
                             X_test=X_XLK_test, Y_test=Y_XLK_test,
                             Y_format='percentage')

# MODEL3: ANN: Multi-layer Perceptron regressor

accuracy_list, accuracy, hidden_layer_sizes, alpha = \
        functions.MLPRegressor_CV(X_XLK_train, Y_XLK_train, n_splits=10,
                            hidden_layer_sizes = [(9), (6), (3)],
                           columns_to_standardize=range(13),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='percentage')


Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.MLPRegressor_Pred(X_XLK_train, Y_XLK_train,
                                    hidden_layer_sizes, alpha,
                             columns_to_standardize=range(13),
                             X_test=X_XLK_test, Y_test=Y_XLK_test,
                             Y_format='percentage')



###############################
# prediction ratio close/open #
###############################

# Direction_ratio as Y
Y_XLK_train = XLK_train.loc[:, 'Direction_ratio']
Y_XLK_test = XLK_test.loc[:, 'Direction_ratio']

# MODEL1: LASSO regression

accuracy_list, accuracy, alpha =\
        functions.lasso_CV(X_XLK_train, Y_XLK_train, n_splits=10,
                           columns_to_standardize=range(13),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='ratio')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.lasso_Pred(X_XLK_train, Y_XLK_train, alpha,
                            columns_to_standardize=range(13),
                            X_test=X_XLK_test, Y_test=Y_XLK_test,
                            Y_format='ratio')

# MODEL2: Ridge regression

accuracy_list, accuracy, alpha = \
        functions.ridge_CV(X_XLK_train, Y_XLK_train, n_splits=10,
                           columns_to_standardize=range(13),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='ratio')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.ridge_Pred(X_XLK_train, Y_XLK_train, alpha,
                             columns_to_standardize=range(13),
                             X_test=X_XLK_test, Y_test=Y_XLK_test,
                             Y_format='ratio')

# MODEL3: ANN: Multi-layer Perceptron regressor

accuracy_list, accuracy, hidden_layer_sizes, alpha = \
        functions.MLPRegressor_CV(X_XLK_train, Y_XLK_train, n_splits=10,
                            hidden_layer_sizes = [(9), (6), (3)],
                           columns_to_standardize=range(13),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='ratio')


Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.MLPRegressor_Pred(X_XLK_train, Y_XLK_train,
                                    hidden_layer_sizes, alpha,
                             columns_to_standardize=range(13),
                             X_test=X_XLK_test, Y_test=Y_XLK_test,
                             Y_format='ratio')
























