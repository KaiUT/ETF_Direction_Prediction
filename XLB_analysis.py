import copy
import pandas_datareader.data as web
import pandas as pd
import datetime as dt
#  import matplotlib.pyplot as plt
import functions

# collect data
start = dt.datetime(2000, 1, 1)
end = dt.datetime(2017, 1, 1)
XLB = web.DataReader('XLB', 'yahoo', start, end).iloc[:, range(5)]


# -------------- XLB direction prediction --------------------------------

# plot raw data
functions.plot_raw_data(XLB, fig_path='/Users/kailiu/GitProjects/ETF_Direction_Prediction/results/XLB_raw.pdf')

###################
# data processing #
###################

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


# ====================== Data-driven models ================================

# choose features - OHLC and weekdays
indices = range(5) + range(32)[27:]
X_XLB_train = XLB_train.iloc[:, indices].values
X_XLB_test = XLB_test.iloc[:, indices].values


#################################
# prediction direction directly #
#################################

# Direction_Binary as Y
Y_XLB_train = XLB_train.loc[:, 'Direction_Binary']
Y_XLB_test = XLB_test.loc[:, 'Direction_Binary']


# MODEL1: Logistic regression with L1 penalty #

accuracy_list, accuracy, c = \
        functions.logistic_CV(X_XLB_train, Y_XLB_train, n_splits=10,
                              columns_to_standardize=range(5),
                              Cs=[2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                                  100], penalty='l1')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.logistic_Pred(X_XLB_train, Y_XLB_train,
                                c, columns_to_standardize=range(5),
                                penalty='l1', X_test=X_XLB_test,
                                Y_test=Y_XLB_test)


# MODEL2: Logistic regression with L2 penalty #

accuracy_list, accuracy, c = \
        functions.logistic_CV(X_XLB_train, Y_XLB_train, n_splits=10,
                              columns_to_standardize=range(5),
                              Cs=[2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                                  100], penalty='l2')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.logistic_Pred(X_XLB_train, Y_XLB_train,
                                c, columns_to_standardize=range(5),
                                penalty='l2', X_test=X_XLB_test,
                                Y_test=Y_XLB_test)


# MODEL3: ANN: Multi-layer Perceptron classifier

accuracy_list, accuracy, hidden_layer_sizes, alpha = \
        functions.MLPClassifier_CV(X_XLB_train, Y_XLB_train, n_splits=10,
                            hidden_layer_sizes = [(9), (6), (3)],
                           columns_to_standardize=range(5),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100])


Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.MLPClassifier_Pred(X_XLB_train, Y_XLB_train,
                                    hidden_layer_sizes, alpha,
                             columns_to_standardize=range(5),
                             X_test=X_XLB_test, Y_test=Y_XLB_test)



###########################################
# prediction percentage (close-open)/open #
###########################################

# Direction_percentage as Y
Y_XLB_train = XLB_train.loc[:, 'Direction_percentage']
Y_XLB_test = XLB_test.loc[:, 'Direction_percentage']

# MODEL1: LASSO regression

accuracy_list, accuracy, alpha =\
        functions.lasso_CV(X_XLB_train, Y_XLB_train, n_splits=10,
                           columns_to_standardize=range(5),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='percentage')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.lasso_Pred(X_XLB_train, Y_XLB_train, alpha,
                            columns_to_standardize=range(5),
                            X_test=X_XLB_test, Y_test=Y_XLB_test,
                            Y_format='percentage')

# MODEL2: Ridge regression

accuracy_list, accuracy, alpha = \
        functions.ridge_CV(X_XLB_train, Y_XLB_train, n_splits=10,
                           columns_to_standardize=range(5),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='percentage')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.ridge_Pred(X_XLB_train, Y_XLB_train, alpha,
                             columns_to_standardize=range(5),
                             X_test=X_XLB_test, Y_test=Y_XLB_test,
                             Y_format='percentage')

# MODEL3: ANN: Multi-layer Perceptron regressor

accuracy_list, accuracy, hidden_layer_sizes, alpha = \
        functions.MLPRegressor_CV(X_XLB_train, Y_XLB_train, n_splits=10,
                            hidden_layer_sizes = [(9), (6), (3)],
                           columns_to_standardize=range(5),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='percentage')


Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.MLPRegressor_Pred(X_XLB_train, Y_XLB_train,
                                    hidden_layer_sizes, alpha,
                             columns_to_standardize=range(5),
                             X_test=X_XLB_test, Y_test=Y_XLB_test,
                             Y_format='percentage')



###############################
# prediction ratio close/open #
###############################

# Direction_ratio as Y
Y_XLB_train = XLB_train.loc[:, 'Direction_ratio']
Y_XLB_test = XLB_test.loc[:, 'Direction_ratio']

# MODEL1: LASSO regression

accuracy_list, accuracy, alpha =\
        functions.lasso_CV(X_XLB_train, Y_XLB_train, n_splits=10,
                           columns_to_standardize=range(5),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='ratio')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.lasso_Pred(X_XLB_train, Y_XLB_train, alpha,
                            columns_to_standardize=range(5),
                            X_test=X_XLB_test, Y_test=Y_XLB_test,
                            Y_format='ratio')

# MODEL2: Ridge regression

accuracy_list, accuracy, alpha = \
        functions.ridge_CV(X_XLB_train, Y_XLB_train, n_splits=10,
                           columns_to_standardize=range(5),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='ratio')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.ridge_Pred(X_XLB_train, Y_XLB_train, alpha,
                             columns_to_standardize=range(5),
                             X_test=X_XLB_test, Y_test=Y_XLB_test,
                             Y_format='ratio')

# MODEL3: ANN: Multi-layer Perceptron regressor

accuracy_list, accuracy, hidden_layer_sizes, alpha = \
        functions.MLPRegressor_CV(X_XLB_train, Y_XLB_train, n_splits=10,
                            hidden_layer_sizes = [(9), (6), (3)],
                           columns_to_standardize=range(5),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='ratio')


Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.MLPRegressor_Pred(X_XLB_train, Y_XLB_train,
                                    hidden_layer_sizes, alpha,
                             columns_to_standardize=range(5),
                             X_test=X_XLB_test, Y_test=Y_XLB_test,
                             Y_format='ratio')



# ====================== Technical indicators models ==========================

# choose features - indicators and weekdays
indices = range(32)[8:]
X_XLB_train = XLB_train.iloc[:, indices].values
X_XLB_test = XLB_test.iloc[:, indices].values


#################################
# prediction direction directly #
#################################

# Direction_Binary as Y
Y_XLB_train = XLB_train.loc[:, 'Direction_Binary']
Y_XLB_test = XLB_test.loc[:, 'Direction_Binary']


# MODEL1: Logistic regression with L1 penalty #

accuracy_list, accuracy, c = \
        functions.logistic_CV(X_XLB_train, Y_XLB_train, n_splits=10,
                              columns_to_standardize=range(19),
                              Cs=[2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                                  100], penalty='l1')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.logistic_Pred(X_XLB_train, Y_XLB_train,
                                c, columns_to_standardize=range(19),
                                penalty='l1', X_test=X_XLB_test,
                                Y_test=Y_XLB_test)


# MODEL2: Logistic regression with L2 penalty #

accuracy_list, accuracy, c = \
        functions.logistic_CV(X_XLB_train, Y_XLB_train, n_splits=10,
                              columns_to_standardize=range(19),
                              Cs=[2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                                  100], penalty='l2')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.logistic_Pred(X_XLB_train, Y_XLB_train,
                                c, columns_to_standardize=range(19),
                                penalty='l2', X_test=X_XLB_test,
                                Y_test=Y_XLB_test)


# MODEL3: ANN: Multi-layer Perceptron classifier

accuracy_list, accuracy, hidden_layer_sizes, alpha = \
        functions.MLPClassifier_CV(X_XLB_train, Y_XLB_train, n_splits=10,
                            hidden_layer_sizes = [(9), (6), (3)],
                           columns_to_standardize=range(19),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100])


Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.MLPClassifier_Pred(X_XLB_train, Y_XLB_train,
                                    hidden_layer_sizes, alpha,
                             columns_to_standardize=range(19),
                             X_test=X_XLB_test, Y_test=Y_XLB_test)



###########################################
# prediction percentage (close-open)/open #
###########################################

# Direction_percentage as Y
Y_XLB_train = XLB_train.loc[:, 'Direction_percentage']
Y_XLB_test = XLB_test.loc[:, 'Direction_percentage']

# MODEL1: LASSO regression

accuracy_list, accuracy, alpha =\
        functions.lasso_CV(X_XLB_train, Y_XLB_train, n_splits=10,
                           columns_to_standardize=range(19),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='percentage')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.lasso_Pred(X_XLB_train, Y_XLB_train, alpha,
                            columns_to_standardize=range(19),
                            X_test=X_XLB_test, Y_test=Y_XLB_test,
                            Y_format='percentage')

# MODEL2: Ridge regression

accuracy_list, accuracy, alpha = \
        functions.ridge_CV(X_XLB_train, Y_XLB_train, n_splits=10,
                           columns_to_standardize=range(19),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='percentage')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.ridge_Pred(X_XLB_train, Y_XLB_train, alpha,
                             columns_to_standardize=range(19),
                             X_test=X_XLB_test, Y_test=Y_XLB_test,
                             Y_format='percentage')

# MODEL3: ANN: Multi-layer Perceptron regressor

accuracy_list, accuracy, hidden_layer_sizes, alpha = \
        functions.MLPRegressor_CV(X_XLB_train, Y_XLB_train, n_splits=10,
                            hidden_layer_sizes = [(9), (6), (3)],
                           columns_to_standardize=range(19),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='percentage')


Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.MLPRegressor_Pred(X_XLB_train, Y_XLB_train,
                                    hidden_layer_sizes, alpha,
                             columns_to_standardize=range(19),
                             X_test=X_XLB_test, Y_test=Y_XLB_test,
                             Y_format='percentage')



###############################
# prediction ratio close/open #
###############################

# Direction_ratio as Y
Y_XLB_train = XLB_train.loc[:, 'Direction_ratio']
Y_XLB_test = XLB_test.loc[:, 'Direction_ratio']

# MODEL1: LASSO regression

accuracy_list, accuracy, alpha =\
        functions.lasso_CV(X_XLB_train, Y_XLB_train, n_splits=10,
                           columns_to_standardize=range(19),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='ratio')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.lasso_Pred(X_XLB_train, Y_XLB_train, alpha,
                            columns_to_standardize=range(19),
                            X_test=X_XLB_test, Y_test=Y_XLB_test,
                            Y_format='ratio')

# MODEL2: Ridge regression

accuracy_list, accuracy, alpha = \
        functions.ridge_CV(X_XLB_train, Y_XLB_train, n_splits=10,
                           columns_to_standardize=range(19),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='ratio')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.ridge_Pred(X_XLB_train, Y_XLB_train, alpha,
                             columns_to_standardize=range(19),
                             X_test=X_XLB_test, Y_test=Y_XLB_test,
                             Y_format='ratio')

# MODEL3: ANN: Multi-layer Perceptron regressor

accuracy_list, accuracy, hidden_layer_sizes, alpha = \
        functions.MLPRegressor_CV(X_XLB_train, Y_XLB_train, n_splits=10,
                            hidden_layer_sizes = [(9), (6), (3)],
                           columns_to_standardize=range(19),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='ratio')


Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.MLPRegressor_Pred(X_XLB_train, Y_XLB_train,
                                    hidden_layer_sizes, alpha,
                             columns_to_standardize=range(19),
                             X_test=X_XLB_test, Y_test=Y_XLB_test,
                             Y_format='ratio')



# ====================== Subset of Technical indicators models ==========================

# choose features - indicators and weekdays
indices = range(32)[14:]
X_XLB_train = XLB_train.iloc[:, indices].values
X_XLB_test = XLB_test.iloc[:, indices].values


#################################
# prediction direction directly #
#################################

# Direction_Binary as Y
Y_XLB_train = XLB_train.loc[:, 'Direction_Binary']
Y_XLB_test = XLB_test.loc[:, 'Direction_Binary']


# MODEL1: Logistic regression with L1 penalty #

accuracy_list, accuracy, c = \
        functions.logistic_CV(X_XLB_train, Y_XLB_train, n_splits=10,
                              columns_to_standardize=range(13),
                              Cs=[2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                                  100], penalty='l1')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.logistic_Pred(X_XLB_train, Y_XLB_train,
                                c, columns_to_standardize=range(13),
                                penalty='l1', X_test=X_XLB_test,
                                Y_test=Y_XLB_test)


# MODEL2: Logistic regression with L2 penalty #

accuracy_list, accuracy, c = \
        functions.logistic_CV(X_XLB_train, Y_XLB_train, n_splits=10,
                              columns_to_standardize=range(13),
                              Cs=[2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                                  100], penalty='l2')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.logistic_Pred(X_XLB_train, Y_XLB_train,
                                c, columns_to_standardize=range(13),
                                penalty='l2', X_test=X_XLB_test,
                                Y_test=Y_XLB_test)


# MODEL3: ANN: Multi-layer Perceptron classifier

accuracy_list, accuracy, hidden_layer_sizes, alpha = \
        functions.MLPClassifier_CV(X_XLB_train, Y_XLB_train, n_splits=10,
                            hidden_layer_sizes = [(9), (6), (3)],
                           columns_to_standardize=range(13),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100])


Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.MLPClassifier_Pred(X_XLB_train, Y_XLB_train,
                                    hidden_layer_sizes, alpha,
                             columns_to_standardize=range(13),
                             X_test=X_XLB_test, Y_test=Y_XLB_test)



###########################################
# prediction percentage (close-open)/open #
###########################################

# Direction_percentage as Y
Y_XLB_train = XLB_train.loc[:, 'Direction_percentage']
Y_XLB_test = XLB_test.loc[:, 'Direction_percentage']

# MODEL1: LASSO regression

accuracy_list, accuracy, alpha =\
        functions.lasso_CV(X_XLB_train, Y_XLB_train, n_splits=10,
                           columns_to_standardize=range(13),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='percentage')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.lasso_Pred(X_XLB_train, Y_XLB_train, alpha,
                            columns_to_standardize=range(13),
                            X_test=X_XLB_test, Y_test=Y_XLB_test,
                            Y_format='percentage')

# MODEL2: Ridge regression

accuracy_list, accuracy, alpha = \
        functions.ridge_CV(X_XLB_train, Y_XLB_train, n_splits=10,
                           columns_to_standardize=range(13),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='percentage')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.ridge_Pred(X_XLB_train, Y_XLB_train, alpha,
                             columns_to_standardize=range(13),
                             X_test=X_XLB_test, Y_test=Y_XLB_test,
                             Y_format='percentage')

# MODEL3: ANN: Multi-layer Perceptron regressor

accuracy_list, accuracy, hidden_layer_sizes, alpha = \
        functions.MLPRegressor_CV(X_XLB_train, Y_XLB_train, n_splits=10,
                            hidden_layer_sizes = [(9), (6), (3)],
                           columns_to_standardize=range(13),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='percentage')


Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.MLPRegressor_Pred(X_XLB_train, Y_XLB_train,
                                    hidden_layer_sizes, alpha,
                             columns_to_standardize=range(13),
                             X_test=X_XLB_test, Y_test=Y_XLB_test,
                             Y_format='percentage')



###############################
# prediction ratio close/open #
###############################

# Direction_ratio as Y
Y_XLB_train = XLB_train.loc[:, 'Direction_ratio']
Y_XLB_test = XLB_test.loc[:, 'Direction_ratio']

# MODEL1: LASSO regression

accuracy_list, accuracy, alpha =\
        functions.lasso_CV(X_XLB_train, Y_XLB_train, n_splits=10,
                           columns_to_standardize=range(13),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='ratio')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.lasso_Pred(X_XLB_train, Y_XLB_train, alpha,
                            columns_to_standardize=range(13),
                            X_test=X_XLB_test, Y_test=Y_XLB_test,
                            Y_format='ratio')

# MODEL2: Ridge regression

accuracy_list, accuracy, alpha = \
        functions.ridge_CV(X_XLB_train, Y_XLB_train, n_splits=10,
                           columns_to_standardize=range(13),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='ratio')

Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.ridge_Pred(X_XLB_train, Y_XLB_train, alpha,
                             columns_to_standardize=range(13),
                             X_test=X_XLB_test, Y_test=Y_XLB_test,
                             Y_format='ratio')

# MODEL3: ANN: Multi-layer Perceptron regressor

accuracy_list, accuracy, hidden_layer_sizes, alpha = \
        functions.MLPRegressor_CV(X_XLB_train, Y_XLB_train, n_splits=10,
                            hidden_layer_sizes = [(9), (6), (3)],
                           columns_to_standardize=range(13),
                           Alphas=[0.1, 1, 5, 10, 30, 50, 100],
                           Y_format='ratio')


Y_test_pred, out_accuracy, Y_train_pred, in_accuracy =\
        functions.MLPRegressor_Pred(X_XLB_train, Y_XLB_train,
                                    hidden_layer_sizes, alpha,
                             columns_to_standardize=range(13),
                             X_test=X_XLB_test, Y_test=Y_XLB_test,
                             Y_format='ratio')
























