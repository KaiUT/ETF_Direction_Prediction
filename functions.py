
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from matplotlib.ticker import MaxNLocator
from matplotlib.finance import candlestick_ohlc
import pandas as pd


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

    if min_period should be considered??

    """
    ewma = data[column].ewm(span=period, min_periods=min_periods, **kwargs).mean()
    return data.join(ewma.to_frame(add_column))


def MACD(data, min_periods=[0,0], column='Close', add_column='MACD', **kwargs):
    r"""Calculate Moving Average Convergence/Divergence (MACD)

    The MACD line is the 12-day EWMA less the 26-day EWMA.

    Using Smoothing Factor 2/(1+N), where N is periods.

    if min_period should be considered??

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

    if min_period should be considered??

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
    _K = 100 * (data[columns[0]] - data[columns[2]].rolling(period).min()) / (data[columns[1]].rolling(period).max() - data[columns[2]].rolling(period).min())
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
                                                columns=columns)
    _K = fast_D
    _D = _K.rolling(smoothing).mean()
    return data.join(_K.to_frame(add_columns[0]).join\
                     (_D.to_frame(add_columns[1])))


def momentum(data, period=4, column='Close', add_column='Momentum'):
    r"""

    """
    M = data[column] / data[column].shift(period) * 100
    return data.join(M.to_frame(add_column))


def acceleration(data, periods=[5, 34, 5], columns=['Close', 'High', 'Low'],\
                 add_column='Acceleration'):
    r"""

    """
    median = (data[columns[1]] + data[columns[2]]) / 2
    AO = median.rolling(periods[1]).mean() - median.rolling(periods[2]).mean()
    AC = AO - AO.rolling(periods[0]).mean()
    return data.join(AC.to_frame(add_column))


def williams_R(data, period=14, columns=['Close', 'High', 'Low'],
               add_column='Williams_%R'):
    r"""

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
    r"""

    """
    _, accdist_series = AccDist(data, columns=columns)
    ewma1 = accdist_series.ewm(span=periods[0], min_periods=min_periods[0],
                              **kwargs).mean()
    ewma2 = accdist_series.ewm(span=periods[1], min_periods=min_periods[1],
                              **kwargs).mean()
    chaikin = ewma1 - ewma2
    return data.join(chaikin.to_frame(add_column))


def william_AccDist(data, columns=['Close', 'High', 'Low'],
                    add_column='William_AccDist'):
    r"""

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























