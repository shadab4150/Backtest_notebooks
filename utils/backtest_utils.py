# IMPORTS
import numpy as np
import pandas as pd
import bt
import scipy
import stockstats
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.graph_objs as go
from prettytable import PrettyTable
from IPython.display import display


# PRELIMINARY FUNCTIONS
def LONG(close, low, signals, open, SL, dates, weekend_entry, slippage):
    '''
    arguments:
    close: close price
    low : low price
    signals : signals (long or short)
    open : open price
    weekend_entry : whether to enter the market on Fridays or not

    returns:
    array of long
    '''
    long = []
    for x in range(len(close)-1):
        # If slippage, change entry price
        if slippage:
            mu = -3.98e-6
            std = 0.000957
            slippage_percent = np.random.normal(loc=mu, scale=std, size=1)[0]
            entry_price = close[x]*(1 + slippage_percent)
        else:
            entry_price = close[x]

        # If Friday, modified calculations
        if dates.iloc[x].strftime('%A') == 'Friday':
            if not weekend_entry:
                # No entry
                long.append(0)
            else:
                # Entry with no stoploss
                ans = (close[x+1]-entry_price)*signals[x]
                long.append(ans)
        # If not Friday, calculate as usual with the stop loss
        else:
            stop_loss = SL
            if signals[x] >= 1 and (low[x+1] < entry_price*(1-stop_loss)):
                ans = -stop_loss*entry_price
                long.append(ans*signals[x])
            else:
                ans = (close[x+1]-entry_price)*signals[x]
                long.append(ans)
    return long


def SHORT(close, low, signals, open, high, SL, dates, weekend_entry, slippage):
    '''
    arguments:
    close: close price
    low : low price
    signals : signals (long or short)
    open : open price
    weekend_entry : whether to enter the market on Fridays or not

    returns:
    array of short
    '''
    short = []
    for x in range(len(close)-1):
        # If slippage, change entry price
        if slippage:
            mu = -3.98e-6
            std = 0.000957
            slippage_percent = np.random.normal(loc=mu, scale=std, size=1)[0]
            entry_price = close[x]*(1 - slippage_percent)
        else:
            entry_price = close[x]

        # If Friday, modified calculations
        if dates.iloc[x].strftime('%A') == 'Friday':
            if not weekend_entry:
                # No entry
                short.append(0)
            else:
                # Entry with no stoploss
                ans = (close[x+1]-entry_price)*signals[x]
                short.append(ans)
        # If not Friday, calculate as usual with the stop loss
        else:
            stop_loss = SL
            if signals[x] <= -1 and (high[x+1] > entry_price*(1+stop_loss)):
                ans = -stop_loss*entry_price
                short.append(ans*abs(signals[x]))
            else:
                ans = (close[x+1]-entry_price)*signals[x]
                short.append(ans)
    return short


def ret_with_sl(longg, short, signals):
    '''
    arguments:
    longg: takes long retruns
    short : takes short returns
    signals : signals (long or short)

    returns:
    array of returns with stop loss applied
    '''
    res = []
    for x in range(len(signals)-1):
        if signals[x] >= 1:
            res.append(longg[x])
        else:
            res.append(short[x])
    return res


def norm_ret(val):
    '''
    takes a values and returns noramlized returns
    '''
    nmrsl = [100]
    for x in range(1, len(val)+1):
        ans = nmrsl[x-1]+val[x-1]*0.01*nmrsl[x-1]
        nmrsl.append(ans)
    return nmrsl


def max_drawdown(X):
    '''
    Calculates max drawdown
    '''
    mdd = 0
    peak = X[0]
    for x in X:
        if x > peak:
            peak = x
        dd = (peak - x) / peak
        if dd > mdd:
            mdd = dd
    return -mdd*100


def sharpe_ratio(X):
    '''
    Calculate sharpe ratio from an array of values
    '''
    sr = pd.Series(X).pct_change().mean()/pd.Series(X).pct_change().std()
    sr = (252**0.5)*sr
    return sr


def total_return(X):
    '''
    calculate total returns
    '''
    return 100 * (pd.Series(X).iloc[-1] / pd.Series(X).iloc[0]-1)


def daily_vol(X):
    '''
    calculates daily volatility
    '''
    return pd.Series(X).pct_change().std()*(252**0.5)*100


def calculate_bt(X):
    return total_return(X), sharpe_ratio(X), max_drawdown(X), daily_vol(X)


def adjusted_sharpe(X, Y):
    """
    calculate adjusted_sharpe
    Y corresponds to spx futures
    """
    r_spx = pd.Series(Y).pct_change().mean()
    asr = (pd.Series(X).pct_change().mean() - r_spx)/(pd.Series(X).pct_change().std())
    asr = (252**0.5)*asr
    return asr


def omega(X, Y):
    '''
    calculates omega
    Y corresponds to spx futures
    '''
    return total_return(X)/(total_return(X)-total_return(Y))


def sl_hit_ratio_alt(sigs, Close, High, Low, SL=0.0075):
    SL_hit = []
    for i in range(len(Close)-1):
        if sigs[i] == 1 and ((Close[i]-Low[i+1]) > SL*Close[i]):
            SL_hit.append(1)
        elif sigs[i] == -1 and ((High[i+1]-Close[i]) > SL*Close[i]):
            SL_hit.append(1)
        else:
            SL_hit.append(0)
    SL_hit.append(0)
    sl_num = sum(SL_hit)*100/len(SL_hit)
    return sl_num, SL_hit


def consec_wrong_or_sl_alt(sl_df, sl_hit_list):
    df_copy = sl_df.copy()
    df_copy["sl_hit"] = sl_hit_list
    df_copy = df_copy[df_copy["prediction"] != 0]
    temp = df_copy["sl_hit"].values
    df_copy["req"] = temp
    df_copy['counter'] = df_copy["req"].diff().ne(0).cumsum()
    df2 = df_copy.groupby('counter')['req'].min().to_frame(name='value').join(
        df_copy.groupby('counter')['req'].count().rename('number')
    )
    max_consec1 = df2[df2['value'] == 1]['number'].tolist()

    def count_elements(seq) -> dict:
        hist = {}
        for i in seq:
            hist[i] = hist.get(i, 0) + 1
        return hist
    counted = count_elements(max_consec1)
    return counted


# BACKTEST FUNCTIONS
def plot_table(tb_df, row_lbls):
    tb_df = tb_df.round(4)
    plt.figure(figsize=(14, len(row_lbls)))
    plt.axis('tight')
    plt.axis('off')
    rcolors = plt.cm.BuPu(np.full(len(row_lbls), 0.1))
    ccolors = plt.cm.BuPu(np.full(tb_df.shape[1], 0.1))
    the_table = plt.table(
        cellText=tb_df.values,
        colLabels=tb_df.columns.tolist(),
        colWidths=[0.2 for x in tb_df.columns.tolist()],
        rowLabels=row_lbls,
        rowColours=rcolors,
        colColours=ccolors,
        loc="center"
    )
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(18)
    the_table.scale(2, 2)
    plt.tight_layout()
    plt.show()


def BackTest(data_preds, plot=False):
    '''
    data_preds : a dataframe contain Index_Close prices and prediction
    It should contain 3 columns names namely:
    column_name = ['Index_Close','prediction']
    plot = True or False for Equity Progression
    '''
    clos = ['Index_Close']
    clo = clos[0]
    name = ['Under lying Asset']
    nm = name[0]
    bk_data = data_preds.copy()
    bk_data.reset_index(inplace=True)
    bk_data = bk_data.loc[:, ('Date', clo)].copy()
    bk_data['Predictions'] = pd.Series(data_preds['prediction'].values)
    bk_data.set_index('Date', inplace=True)
    bk_data[clo] = pd.to_numeric(bk_data[clo])
    bk_df_ = bk_data[[clo]].copy()
    bk_df_.columns = ['Under lying Asset']
    bk_sig = bk_data[['Predictions']].copy()
    bk_sig.columns = ['Under lying Asset']
    tw = bk_sig.copy()

    tw[bk_sig == 1.0] = 1.0
    tw[bk_sig == 0.0] = 0.0
    tw[bk_sig == -1.0] = -1.0

    s1 = bt.Strategy(
              "Model",
              [
                  bt.algos.RunDaily(),
                  bt.algos.WeighTarget(tw),  # Triggers weights for long/short strategy
                 bt.algos.Rebalance(),
               ],)
    s2 = bt.Strategy(
          nm,
            [
                  bt.algos.RunDaily(),
                bt.algos.SelectAll(),
                bt.algos.WeighEqually(),
                bt.algos.Rebalance(),
            ],)
    test = bt.Backtest(s1, bk_df_)
    test_sp500 = bt.Backtest(s2, bk_df_)
    res1 = test
    res2 = test_sp500
    a = bt.run(res1, res2)
    print("---"*20)
    a_res = a.prices
    a.display()
    if plot:
        a.plot()
    return a_res


def rollingMetrics(norm_ret_with_lev, date, rolling_analysis_window):
    rolling_sharpe = []
    rolling_returns = []
    rolling_drawdown = []
    rolling_volatility = []
    for i in range(rolling_analysis_window, len(norm_ret_with_lev)):
        # Y = pct_chng_model_sl[i-rolling_analysis_window:i+1]
        # X = norm_ret(Y)
        X = norm_ret_with_lev[i-rolling_analysis_window:i+1]
        TR, SR, MDD, DV = calculate_bt(X)
        rolling_sharpe.append(SR)
        rolling_returns.append(TR)
        rolling_drawdown.append(MDD)
        rolling_volatility.append(DV)

    plt.figure(figsize=(18, 15))
    plt.suptitle('Rolling Metrics')
    idx = pd.to_datetime(date[-len(rolling_returns)-rolling_analysis_window:-rolling_analysis_window])
    ax1 = plt.subplot(411)
    ax1.hlines(0, idx[0], idx[-1], 'r', 'dotted')
    ax1.plot(pd.Series(index=idx, data=rolling_returns), label='Rolling Returns', linewidth=2.0)
    ax1.legend()
    ax1.set(xlabel="Date the model goes live")
    ax2 = plt.subplot(412)
    ax2.hlines(0, idx[0], idx[-1], 'r', 'dotted')
    ax2.plot(pd.Series(index=idx, data=rolling_sharpe), label='Rolling Sharpe', linewidth=2.0)
    ax2.legend()
    ax2.set(xlabel="Date the model goes live")
    ax3 = plt.subplot(413)
    ax3.hlines(0, idx[0], idx[-1], 'r', 'dotted')
    ax3.plot(pd.Series(index=idx, data=rolling_drawdown), label='Rolling Max Drawdown', linewidth=2.0)
    ax3.legend()
    ax3.set(xlabel="Date the model goes live")
    ax4 = plt.subplot(414)
    ax4.hlines(0, idx[0], idx[-1], 'r', 'dotted')
    ax4.plot(pd.Series(index=idx, data=rolling_volatility), label='Rolling Volatility', linewidth=2.0)
    ax4.legend()
    ax4.set(xlabel="Date the model goes live")
    plt.show()

    x = PrettyTable();
    x.field_names = ["Stats", "Average", "Max", "Min"]
    x.add_row(["Total Return", sum(rolling_returns)/len(rolling_returns), max(rolling_returns), min(rolling_returns)])
    x.add_row(["Daily Sharpe", sum(rolling_sharpe)/len(rolling_sharpe), max(rolling_sharpe), min(rolling_sharpe)])
    x.add_row(["Max Drawdown", sum(rolling_drawdown)/len(rolling_drawdown), max(rolling_drawdown), min(rolling_drawdown)])
    x.add_row(["Daily Volatility", sum(rolling_volatility)/len(rolling_volatility), max(rolling_volatility), min(rolling_volatility)])
    print('Rolling Metrics [{} days window]'.format(rolling_analysis_window))
    print(x)


def rollingStopLossAnalysis(sl_hit_list, sl_df, n_days=7, plot=True):
    sl_df["sl_hit"] = sl_hit_list

    def count_sl(sl_df):
        df_copy = sl_df.copy()
        df_copy = df_copy[(df_copy["prediction"] != 0) & (df_copy["sl_hit"] == 1)]
        return len(df_copy)

    sl_count_list = []
    for i in range(n_days, len(sl_df)):
        sl_count_list.append(count_sl(sl_df[i-n_days:i]))
    sl_avg, sl_std = np.mean(sl_count_list), np.std(sl_count_list)

    if plot:
        print('\n\nRolling SL Analysis [{} day window]'.format(n_days))
        x = PrettyTable()
        x.field_names = ["Stats", "Average", "STD", "Max", "Min"]
        x.add_row(["SL Hit Counts", np.around(sl_avg, 2), np.around(sl_std, 2), max(sl_count_list), min(sl_count_list)])
        print(x)
        plt.figure(figsize=(18, 4))
        plt.title('Number of SL hits [{} day window]'.format(n_days))
        plt.plot(sl_count_list)
        plt.hlines(sl_avg, 0, len(sl_count_list), 'g', label="average")
        plt.hlines(sl_avg+sl_std, 0, len(sl_count_list), 'r', "dashed", label="average+1STD")
        plt.hlines(sl_avg+sl_std*2, 0, len(sl_count_list), 'r', "dashed", label="average+2STD")
        plt.legend()
        plt.show()
        plt.figure(figsize=(18, 4))
        plt.title('Histogram of SL hits [{} day window]'.format(n_days))
        plt.hist(sl_count_list)
        plt.show()
    return sl_count_list


def getVolatility(ohlc, how='adx'):
    sdf = ohlc.copy()
    sdf = stockstats.StockDataFrame(sdf)
    if how == 'adx':
        return sdf.get('adx').values


def volatilityAnalysis(norm_ret_with_lev, ohlc, sl_hit_list, sl_count_list, rolling_analysis_window):
    # Get volatility
    vol = getVolatility(ohlc, how='adx')

    df = pd.DataFrame({'volatility': vol, 'sl_hit': sl_hit_list})
    bins = [i*5 for i in range(21)]
    g = pd.cut(
        df['volatility'],
        bins=bins,
        labels=['{}-{}'.format(bins[i], bins[i+1]) for i in range(len(bins[:-1]))]
    )
    df = df.groupby(g, observed=True)['sl_hit'].agg(['count', 'sum']).reset_index()
    df['percentage_hits'] = 100*df['sum']/df['count']
    df.set_index('volatility', inplace=True)

    # Calculate rolling metrics
    rolling_sharpe = []
    rolling_returns = []
    rolling_drawdown = []
    rolling_volatility = []
    rolling_underlying_volatility = []
    for i in range(rolling_analysis_window, len(norm_ret_with_lev)):
        X = norm_ret_with_lev[i-rolling_analysis_window:i+1]
        TR, SR, MDD, DV = calculate_bt(X)
        rolling_sharpe.append(SR)
        rolling_returns.append(TR)
        rolling_drawdown.append(MDD)
        rolling_volatility.append(DV)

        V = vol[i-rolling_analysis_window:i+1]
        rolling_underlying_volatility.append(np.nanmean(V))

    plt.figure(figsize=(18, 15))
    plt.suptitle('Volatility Analysis')
    idx = pd.to_datetime(ohlc.Date.values[-len(rolling_returns)-rolling_analysis_window:-rolling_analysis_window])

    ax1 = plt.subplot(411)
    color = 'tab:blue'
    ax1.hlines(0, idx[0], idx[-1], color, 'dotted')
    ax1.set_ylabel('Rolling Returns', color=color)
    ax1.plot(pd.Series(index=idx, data=rolling_returns), label='Rolling Returns', linewidth=2.0, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    #ax1.set(xlabel="Date the model goes live")
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.hlines(0, idx[0], idx[-1], color, 'dotted')
    ax2.set_ylabel('mean ADX', color=color)
    ax2.plot(pd.Series(index=idx, data=rolling_underlying_volatility), label='mean ADX', linewidth=2.0, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    ax3 = plt.subplot(412)
    ax3.axhline(0, color='r')
    ax3.set_xlabel('mean ADX')
    ax3.set_ylabel('Rolling returns')
    ax3.scatter(rolling_underlying_volatility, rolling_returns)

    ax4 = plt.subplot(413)
    color = 'tab:blue'
    ax4.hlines(0, idx[0], idx[-1], color, 'dotted')
    ax4.set_ylabel('Rolling SL hits', color=color)
    ax4.plot(pd.Series(index=idx, data=sl_count_list), label='Rolling SL hits', linewidth=2.0, color=color)
    ax4.tick_params(axis='y', labelcolor=color)
    #ax4.set(xlabel="Date the model goes live")
    ax5 = ax4.twinx()
    color = 'tab:red'
    ax5.hlines(0, idx[0], idx[-1], color, 'dotted')
    ax5.set_ylabel('mean ADX', color=color)
    ax5.plot(pd.Series(index=idx, data=rolling_underlying_volatility), label='mean ADX', linewidth=2.0, color=color)
    ax5.tick_params(axis='y', labelcolor=color)



def BackTestAnalysis(
    pred_df,
    start_capital=10000,
    max_loss_per_day=2,
    stop_loss_long=0.75/100,
    stop_loss_short=0.75/100,
    contract_size=5,
    excess_liquidity_factor=1200,
    transaction_charge_per_lot=2.57,
    weekend_entry=True,
    slippage=False,
    rolling_analysis_window=66,
    volatility_analysis_window=22,
    save_sheet=True,
    interactive=False,
    stats=True,
    model_name='Model',
    asset_name='Under_Lying_asset',
):
    '''
    pred_df : a dataframe containing 6 columns, namely - Date, High, Low, Close, Open, prediction
    start_capital: starting capital
    max_loss_per_day: maximum allowed loss per day
    stop_loss_long: stop loss for long signals
    stop_loss_short: stop loss for short signals
    contract size: Futures contract is how many multiples of the SPX (MES = $5 x SPX)
    weekend_entry: if False, will not take any positions on Friday irrespective of signal;
                   if True, it will take a position without any stop loss.
    '''
    all_prd = pred_df.copy()
    all_prd['Date'] = pd.to_datetime(all_prd['Date'])
    dates = all_prd['Date']
    signals = all_prd['prediction'].values  # signals
    high = all_prd['High'].values  # high price
    low = all_prd['Low'].values  # low price
    close = all_prd['Close'].values  # close price
    open = all_prd['Open'].values  # open price
    ret = [(close[x+1]-close[x])*signals[x] for x in range(len(close)-1)]  # gives return
    longg = LONG(close, low, signals, open, stop_loss_long, dates, weekend_entry, slippage)  # long returns
    short = SHORT(close, low, signals, open, high, stop_loss_short, dates, weekend_entry, slippage)  # short returns
    return_with_sl = ret_with_sl(longg, short, signals)  # reurns with stop loss
    pct_chng_sp500 = [(close[x+1]-close[x])/close[x]*100 for x in range(len(close)-1)]  # pct_chng S&P50
    pct_chng_model_sl = [(return_with_sl[x]/close[x])*100 for x in range(len(return_with_sl))]  # pct_change on model with stop loss
    pct_chng_model = [(ret[x]/close[x])*100 for x in range(len(ret))]  # pct_chnage on model
    norm_model_return_sl = norm_ret(pct_chng_model_sl)  # noramlized return with stop loss
    norm_sp500_return = norm_ret(pct_chng_sp500)  # normalized S&P500 return
    norm_model_ret = norm_ret(pct_chng_model)  # Normalized model return

    portfolio_value = [start_capital]
    model_returns = list(return_with_sl)
    norm_ret_with_lev = [100]
    SLs = []
    lots = []
    transaction_charge = []

    mlpd = max_loss_per_day/100
    for i in range(len(signals[:-1])):
        stop_loss = stop_loss_long if signals[0] >= 1 else stop_loss_short
        n_lots = min(
            np.floor(mlpd * portfolio_value[i] / (contract_size * stop_loss * close[i])),
            np.floor(portfolio_value[i] / excess_liquidity_factor)
        )
        trans_ch = abs(n_lots / contract_size * transaction_charge_per_lot * 2 * signals[i])
        trans_ch = trans_ch if model_returns[i] != 0 else 0  # Set zero transaction charges when model is out on the weekend
        res = (portfolio_value[i] + (model_returns[i] * contract_size * n_lots)) - trans_ch
        res_norm = norm_ret_with_lev[i] * (res / portfolio_value[i])

        portfolio_value.append(res)
        norm_ret_with_lev.append(res_norm)
        lots.append(n_lots)
        transaction_charge.append(trans_ch)
        SLs.append(stop_loss)

    wins_after_sl_short = [0]
    for i in range(1, len(signals)):
        if pct_chng_model_sl[i-1] > 0 and signals[i-1] == -1:
            wins_after_sl_short.append(wins_after_sl_short[i-1]+1)
        else:
            wins_after_sl_short.append(wins_after_sl_short[i-1])
    wins_after_sl_long = [0]
    for i in range(1, len(signals)):
        if pct_chng_model_sl[i-1] > 0 and signals[i-1] == 1:
            wins_after_sl_long.append(wins_after_sl_long[i-1]+1)
        else:
            wins_after_sl_long.append(wins_after_sl_long[i-1])
    num_short_signals = [0]
    for i in range(1, len(signals)):
        if signals[i-1] == -1:
            num_short_signals.append(num_short_signals[i-1]+1)
        else:
            num_short_signals.append(num_short_signals[i-1])
    num_long_signals = [0]
    for i in range(1, len(signals)):
        if signals[i-1] == 1:
            num_long_signals.append(num_long_signals[i-1]+1)
        else:
            num_long_signals.append(num_long_signals[i-1])
    No_of_long_signals = num_long_signals[-1]
    No_of_short_signals = num_short_signals[-1]
    Win_ratio_long = wins_after_sl_long[-1]/num_long_signals[-1]
    try:
        Win_ratio_short = wins_after_sl_short[-1]/num_short_signals[-1]
    except Exception as err:
        print(err)
        Win_ratio_short = 0

    # creating data frame:
    x = all_prd.Date.tolist()[:]
    all_prd.set_index('Date', inplace=True)
    all_prd['long'] = list(longg) + [None]
    all_prd['short'] = list(short) + [None]
    all_prd['return with SL'] = list(return_with_sl) + [None]
    all_prd['% change SP500'] = list(pct_chng_sp500) + [None]
    all_prd['% change Model'] = list(pct_chng_model) + [None]
    all_prd['% change Model_SL'] = list(pct_chng_model_sl) + [None]
    all_prd['Normalised Model return'] = list(norm_model_ret)
    all_prd['Normalised Model_SL return'] = list(norm_model_return_sl)
    all_prd['Normalised SP500 return'] = list(norm_sp500_return)
    all_prd['Normalised Leverage Model SL'] = norm_ret_with_lev
    all_prd['Wins after SL-Short'] = wins_after_sl_short
    all_prd['Wins after SL-Long'] = wins_after_sl_long
    all_prd['No of Short Signals'] = num_short_signals
    all_prd['No of Long Signals'] = num_long_signals
    all_prd['Portfolio Value'] = portfolio_value
    all_prd['Lots'] = lots + [None]
    all_prd['Transaction Charge'] = transaction_charge + [None]
    all_prd['Leverage'] = [(close[i]*lots[i]*contract_size)/portfolio_value[i] for i in range(len(close[:-1]))] + [None]
    all_prd['Net Liquidity'] = portfolio_value
    all_prd['Excess Liquidity'] = [portfolio_value[i]-lots[i]*1200 for i in range(len(close[:-1]))] + [None]
    all_prd['Maximum loss in $'] = [lots[i]*contract_size*close[i]*SLs[i] for i in range(len(close[:-1]))] + [None]
    all_prd['Margin Call Error'] = all_prd['Excess Liquidity']-all_prd['Maximum loss in $']
    try:
        all_prd.drop(['y_true'], axis=1, inplace=True)
    except:
        pass
    print('Start Date : ', str(x[0])[:10], ' End Date : ', str(x[-1])[:10])
    if interactive:
        trace1 = go.Scatter(x=x, y=norm_sp500_return, mode='lines', line=dict(color='darkorange', width=2), name='Under Lying Asset')
        trace2 = go.Scatter(x=x, y=norm_model_return_sl, mode='lines', line=dict(color='navy', width=2), name='Model_with_stoploss', showlegend=True)
        trace3 = go.Scatter(x=x, y=norm_model_ret, mode='lines', line=dict(color='green', width=2), name='Model', showlegend=True)
        trace4 = go.Scatter(x=x, y=norm_ret_with_lev, mode='lines', line=dict(color='red', width=2), name='Model_with_leverage', showlegend=True)
        layout = go.Layout(title='Equity Progression', xaxis=dict(title='Dates'), yaxis=dict(title='Normalized Return'))
        fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout)
        fig.show()
    else:
        plt.figure(figsize=(14, 5))
        all_prd['Normalised Leverage Model SL'].plot()
        all_prd['Normalised Model return'].plot()
        all_prd['Normalised Model_SL return'].plot()
        all_prd['Normalised SP500 return'].plot()
        plt.legend()
        plt.title('Equity Progression')
        plt.show()
    if save_sheet:
        all_prd.to_excel('performance_sheet_.xlsx', index=False)
    if stats:
        rt, sr, mdd, dv = calculate_bt(norm_model_ret)
        rt1, sr1, mdd1, dv1 = calculate_bt(norm_model_return_sl)
        rt2, sr2, mdd2, dv2 = calculate_bt(norm_sp500_return)
        rt3, sr3, mdd3, dv3 = calculate_bt(norm_ret_with_lev)
        arr, arr1, arr2, arr3 = [rt, sr, mdd, dv], [rt1, sr1, mdd1, dv1], [rt2, sr2, mdd2, dv2], [rt3, sr3, mdd3, dv3]
        sts = ['Total Return', "Daily Sharpe", 'Max Drawdown', 'Daily volatility']
        st_df = pd.DataFrame(np.column_stack((arr, arr1, arr2, arr3)),
                             columns=[f'{model_name}', f'{model_name}_SL', f'{asset_name}', f'{model_name}_LEV_SL'])
        plot_table(st_df, sts)

        r1 = [omega(norm_model_ret, norm_sp500_return), adjusted_sharpe(norm_model_ret, norm_sp500_return)]
        r2 = [omega(norm_model_return_sl, norm_sp500_return), adjusted_sharpe(norm_model_return_sl, norm_sp500_return)]
        # r3 = [omega(norm_sp500_return, norm_sp500_return), adjusted_sharpe(norm_sp500_return, norm_sp500_return)]
        r4 = [omega(norm_ret_with_lev, norm_sp500_return), adjusted_sharpe(norm_ret_with_lev, norm_sp500_return)]
        labels = ['Omega', 'Adjusted Sharpe']
        vals = pd.DataFrame(np.column_stack((r1, r2, r4)),
                             columns=[f'{model_name}', f'{model_name}_SL', f'{model_name}_LEV_SL'])
        plot_table(vals, labels)

        sts1 = [
            'Maximum % loss per day', 'Starting Capital', 'Stoploss of Strategy Long', 'Stoploss of Strategy Short',
            'No of Long Signals', 'No of Short Signals', 'Win Ratio Long', 'Win Ratio Short']
        val1 = [
            max_loss_per_day, start_capital, stop_loss_long, stop_loss_short,
            No_of_long_signals, No_of_short_signals, Win_ratio_long, Win_ratio_short]
        st_df1 = pd.DataFrame(val1, columns=['Parameters Value'])
        plot_table(st_df1, sts1)

    RET = all_prd['Portfolio Value'].pct_change().fillna(0).values

    plt.figure(1, figsize=(18, 5))
    plt.subplot(121)
    sns.histplot(RET, bins=20)
    plt.title('Histogram of returns with SL')


    # display percentage of OUT LONG SHORT days of total
    display((pred_df['prediction'].value_counts()/pred_df.shape[0]).to_frame().style.background_gradient(cmap='Blues').format('{:.2%}'))

    # Rolling Metrics
    if rolling_analysis_window:
        rollingMetrics(norm_ret_with_lev, x, rolling_analysis_window)

    sl_hit_num, sl_hit_list = sl_hit_ratio_alt(signals, close, high, low, SL=0.0075)
    sl_df = pred_df[['Close', 'prediction']].copy()
    cons_sl_hit = consec_wrong_or_sl_alt(sl_df, sl_hit_list)
    print("\n\nSTOP LOSS STATISTICS")
    print(f"Stop loss hit percent: {sl_hit_num:.1f} %")
    CN = 'Consecutive SL Hits'
    cons_sl_hit_df = pd.DataFrame(
        cons_sl_hit.items(),
        columns=[CN, 'Count']
    ).sort_values(CN).set_index(CN)
    display(cons_sl_hit_df)

    # Rolling SL analysis
    _ = rollingStopLossAnalysis(sl_hit_list, sl_df, n_days=7)

    # Volatility analysis
    if volatility_analysis_window:
        ohlc = pd.DataFrame(zip(dates, open, high, low, close), columns=['Date', 'open', 'high', 'low', 'close'])
        ohlc['Date'] = pd.to_datetime(ohlc['Date'])
        sl_count_list = rollingStopLossAnalysis(sl_hit_list, sl_df, n_days=volatility_analysis_window, plot=False)
        volatilityAnalysis(norm_ret_with_lev, ohlc, sl_hit_list, sl_count_list, volatility_analysis_window)

    return all_prd
