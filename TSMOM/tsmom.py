import pandas as pd
import numpy as np

# Configuration
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Tickers
d_tickers = {
    'Equities': ['ES1 Index', 'NQ1 Index', 'Z 1 Index', 'VG1 Index', 'XU1 Index', 'NK1 Index'],
    'Commodities': ['CL1 Comdty', 'CO1 Comdty', 'NG1 Comdty', 'XB1 Comdty',
                    'HO1 Comdty', 'HG1 Comdty', 'LA1 Comdty', 'LX1 Comdty',
                    'C 1 Comdty', 'S 1 Comdty', 'BO1 Comdty', 'LC1 Comdty',
                    'W 1 Comdty', 'GC1 Comdty', 'PL1 Comdty'],
    'Rates': ['FV1 Comdty', 'TY1 Comdty', 'RX1 Comdty', 'JB1 Comdty'],
    'FX': ['ANT1 Curncy', 'IUS1 Curncy', 'CD1 Curncy', 'LAS1 Curncy', 'AD1 Curncy']
}

def ex_ante_vol(data_series, com=60, freq='D'):
    """
    Calculate the ex-ante volatility of a time series.
    
    Parameters:
    data_series (pd.Series): Price data series.
    com (int): Center of mass for the exponential weighted moving average.
    freq (str): Frequency of data ('D' for daily, 'M' for monthly).
    
    Returns:
    pd.Series: Annualized volatility series.
    """
    ret_series = data_series.pct_change(periods=1)  # Calculate daily returns
    vol = ret_series.ewm(com=com).std()  # Exponentially weighted standard deviation
    ann_vol = vol * np.sqrt(253)  # Annualize the volatility
    if freq == 'M':
        ann_vol = ann_vol.resample('BM').last().ffill()  # Resample to monthly if needed
    return ann_vol

def cum_returns(data_series):
    """
    Calculate cumulative returns from a price series.
    
    Parameters:
    data_series (pd.Series): Price data series.
    
    Returns:
    pd.Series: Cumulative returns series.
    """
    ret_series = data_series.pct_change(periods=1)  # Calculate daily returns
    cum_rets = (1 + ret_series).cumprod()  # Calculate cumulative returns
    cum_rets.iloc[0] = 1  # Set the first value to 1
    return cum_rets

def tsmom_signal(asset_name, cum_returns, vol, lookback, vol_target=0.1):
    """
    Calculate time series momentum signals.
    
    Parameters:
    asset_name (str): Name of the asset.
    cum_returns (pd.Series): Cumulative returns series.
    vol (pd.Series): Volatility series.
    lookback (int): Lookback period for momentum calculation.
    vol_target (float): Target volatility for position sizing.
    
    Returns:
    pd.DataFrame: Position sizes.
    pd.DataFrame: PnL series.
    """
    df = pd.concat([cum_returns, vol, cum_returns.pct_change(lookback)], axis=1, keys=['cum_returns', 'vol', 'lookback'])
    col_name = f"{asset_name} {lookback} D"
    pnl, size_dict = {df.index[lookback]: 0}, {df.index[lookback]: 1}  # Initialize PnL and size dictionaries

    # Loop over the data to calculate PnL and position sizes
    for k in range(lookback + 1, len(df)):
        leverage = vol_target / df['vol'].iloc[k-1]  # Calculate leverage
        if df['lookback'].iloc[k-1] > 0:  # If momentum is positive
            pnl[df.index[k]] = ((df['cum_returns'].iloc[k] / df['cum_returns'].iloc[k - 1]) - 1) * leverage
            size_dict[df.index[k]] = leverage
        elif df['lookback'].iloc[k-1] < 0:  # If momentum is negative
            pnl[df.index[k]] = ((df['cum_returns'].iloc[k] / df['cum_returns'].iloc[k - 1]) - 1) * leverage * -1
            size_dict[df.index[k]] = leverage * -1

    # Convert dictionaries to DataFrames
    new_size = pd.DataFrame.from_dict(size_dict, orient='index', columns=[col_name])
    new_pnl = pd.DataFrame.from_dict(pnl, orient='index', columns=[col_name])
    return new_size, new_pnl

def tsmom_gearing(signal_returns):
    """
    Calculate the gearing for a set of signal returns.
    
    Parameters:
    signal_returns (pd.DataFrame): DataFrame of signal returns.
    
    Returns:
    pd.DataFrame: Gearing values.
    """
    corr_m = signal_returns.ewm(com=60).corr().reset_index().dropna()  # Calculate exponentially weighted correlations
    corr_grouped = corr_m.groupby('level_0').sum(numeric_only=True)  # Group by date
    gearing_sum = corr_grouped.sum(axis=1).replace(0, 1e-10)  # Sum correlations and replace zeros
    corr_grouped['Gearing'] = 1 / np.sqrt(gearing_sum)  # Calculate gearing
    return corr_grouped[['Gearing']]

def comb_pnl(pnl, asset_weights, gearing):
    """
    Combine PnL and weights with gearing.
    
    Parameters:
    pnl (pd.DataFrame): PnL series.
    asset_weights (pd.DataFrame): Asset weights.
    gearing (pd.Series): Gearing values.
    
    Returns:
    pd.DataFrame: Combined PnL and weight.
    """
    df_comb_pnl = pnl.mul(gearing, axis=0)  # Apply gearing to PnL
    df_comb_pnl['PnL'] = df_comb_pnl.sum(axis=1)  # Sum PnL across all assets
    df_comb_pnl['Weight'] = asset_weights.sum(axis=1)  # Sum weights across all assets
    df_comb_pnl.dropna(inplace=True)  # Drop NaN values
    return df_comb_pnl[['PnL', 'Weight']]

def total_return(asset_returns, exposure_size, period):
    """
    Calculate total returns from asset returns and exposure sizes.
    
    Parameters:
    asset_returns (pd.DataFrame): Asset returns.
    exposure_size (pd.DataFrame): Exposure sizes.
    period (str): Frequency period for returns ('weekly', 'daily').
    
    Returns:
    pd.DataFrame: Total returns.
    """
    asset_returns = asset_returns.pivot(index='date', columns='security', values='PX_Last')
    if period == 'weekly':
        asset_returns = asset_returns.resample('W-WED').last()  # Resample to weekly if needed
    asset_returns = asset_returns.pct_change().shift(-1).fillna(0)  # Calculate returns and shift

    df_return = exposure_size.copy()
    df_return['PnL'] = exposure_size.mul(asset_returns).round(2).dropna().sum(axis=1)  # Calculate PnL
    df_return['Pct_PnL'] = df_return['PnL'] / 100000000  # Scale PnL
    return df_return

def tsmom_port(source, lookback_days, d_tickers, period=None):
    """
    Run the time series momentum strategy for multiple assets.
    
    Parameters:
    source (pd.DataFrame): Data source dataframe.
    lookback_days (list): List of lookback periods.
    d_tickers (dict): Dictionary of tickers by asset class.
    period (str): Frequency period for returns ('weekly', 'daily').
    
    Returns:
    dict: Dictionary of results by asset class.
    """
    data = source.pivot(index='date', columns='security', values='PX_Last').ffill().reset_index()
    data['date'] = pd.to_datetime(data['date'])
    data = pd.melt(data, id_vars=['date'], var_name='security', value_name='PX_Last')  # Reshape data

    fx_rate = source.pivot(index='date', columns='security', values='FX_Rate').ffill().reset_index()
    fx_rate['date'] = pd.to_datetime(fx_rate['date'])
    fx_rate = pd.melt(fx_rate, id_vars=['date'], var_name='security', value_name='FX_Rate')
    fx_rate['FX_Rate'] = 1 / fx_rate['FX_Rate']  # Invert FX rates

    tickers = [ticker for asset in ['Equities', 'Commodities', 'Rates'] for ticker in d_tickers[asset]]
    cont_size = source[['security', 'FUT_CONT_SIZE']].drop_duplicates()
    cont_value = data.merge(cont_size, on='security').merge(fx_rate, on=['date', 'security'])
    cont_value['cont_value'] = cont_value['PX_Last'] * cont_value['FUT_CONT_SIZE'] * cont_value['FX_Rate']
    cont_value = cont_value.pivot(index='date', columns='security', values='cont_value')[tickers]

    d_asset_weight, d_asset_pnl = {}, {}
    for ticker in tickers:
        df_data_ticker = data[data.security == ticker][['date', 'PX_Last']].set_index('date')
        ticker_cum_ret = cum_returns(df_data_ticker['PX_Last'])
        ticker_ex_ante_vol = ex_ante_vol(df_data_ticker['PX_Last'])

        dfs_pnl, dfs_size = [], []
        for lookback in lookback_days:
            ticker_size, ticker_pnl = tsmom_signal(ticker, ticker_cum_ret, ticker_ex_ante_vol, lookback)
            dfs_pnl.append(ticker_pnl)
            dfs_size.append(ticker_size)

        df_pnl = pd.concat(dfs_pnl, axis=1).dropna()  # Combine PnL DataFrames
        gearing = tsmom_gearing(df_pnl)  # Calculate gearing
        df_size = pd.concat(dfs_size, axis=1).join(gearing).dropna()  # Combine size DataFrames
        df_weighted = df_size.iloc[:, :-1].mul(df_size['Gearing'], axis=0)  # Apply gearing to sizes
        df_comb_pnl = comb_pnl(df_pnl, df_weighted, df_size['Gearing'])  # Combine PnL and weights

        d_asset_weight[ticker] = df_comb_pnl['Weight']
        d_asset_pnl[ticker] = df_comb_pnl['PnL']

    df_pnl = pd.DataFrame(d_asset_pnl)
    df_size = pd.DataFrame(d_asset_weight)

    d_classes, d_ind_weight, d_class_weight, d_class_pnl = {}, {}, {}, {}
    for key in ['Equities', 'Commodities', 'Rates']:
        l_tickers = d_tickers[key]
        pnl_class = df_pnl[l_tickers].dropna()  # Filter PnL DataFrame for the asset class
        gearing = tsmom_gearing(pnl_class)  # Calculate gearing
        size_class = df_size[l_tickers].join(gearing).dropna()  # Combine size DataFrame with gearing
        weighted_class = size_class.iloc[:, :-1].mul(size_class['Gearing'], axis=0)  # Apply gearing to sizes
        class_pnl = comb_pnl(pnl_class, weighted_class, size_class['Gearing'])  # Combine PnL and weights

        df_class = weighted_class.copy()
        df_class['PnL'] = class_pnl['PnL']
        df_class['CumPnL'] = (df_class['PnL'] + 1).cumprod()  # Calculate cumulative PnL

        d_class_weight[key] = class_pnl['Weight']
        d_ind_weight[key] = weighted_class
        d_class_pnl[key] = class_pnl['PnL']
        d_classes[key] = df_class

    df_class_pnl = pd.DataFrame(d_class_pnl)
    gearing = tsmom_gearing(df_class_pnl)  # Calculate gearing for classes
    df_class_size = pd.DataFrame(d_class_weight).join(gearing).dropna()  # Combine class sizes with gearing
    df_weighted = df_class_size.iloc[:, :-1].mul(df_class_size['Gearing'], axis=0)  # Apply gearing to sizes
    df_comb_pnl = comb_pnl(df_class_pnl, df_weighted, df_class_size['Gearing'])  # Combine PnL and weights

    # Combine individual weights across asset classes
    dfs_weights = [d_ind_weight[key].multiply(df_class_size['Gearing'], axis='index').dropna() for key in ['Equities', 'Commodities', 'Rates']]
    df_port = pd.concat(dfs_weights, axis=1).mul(100000000).div(cont_value).round(0).dropna()  # Scale and combine weights
    df_port.to_csv('contracts.csv')
    df_port = df_port.mul(cont_value).dropna()

    if period == 'weekly':
        df_port = df_port.resample('W-WED').last()  # Resample to weekly if needed

    df_output = total_return(data, df_port, period)  # Calculate total returns
    df_output['CumPnL'] = (df_output['Pct_PnL'] + 1).cumprod()  # Calculate cumulative PnL
    df_test = df_output[['Pct_PnL']].tail(1000).std() * np.sqrt(253)  # Calculate annualized volatility

    print(df_test)
    d_classes['Equities'].to_csv('equities_results.csv')
    d_classes['Commodities'].to_csv('commodities_results.csv')
    d_classes['Rates'].to_csv('rates_results.csv')
    df_output.to_csv('cta_results.csv')

    return d_classes

def tsmom_breakpoints(lookback_days, d_tickers):
    """
    Calculate breakpoints for time series momentum strategy.
    
    Parameters:
    lookback_days (list): List of lookback periods.
    d_tickers (dict): Dictionary of tickers by asset class.
    
    Returns:
    pd.DataFrame: DataFrame of breakpoints.
    """
    source = pd.read_csv('data_pull_updated.csv')
    data = source.pivot(index='date', columns='security', values='PX_Last').ffill().reset_index()
    data['date'] = pd.to_datetime(data['date'])
    data = pd.melt(data, id_vars=['date'], var_name='security', value_name='PX_Last')  # Reshape data

    tickers = [ticker for asset in ['Equities', 'Commodities', 'Rates'] for ticker in d_tickers[asset]]
    dfs = []

    for ticker in tickers:
        ticker_data = data[data['security'] == ticker]
        # Select specific breakpoints
        df_filtered = pd.concat([ticker_data.iloc[-i].to_frame().transpose() for i in [1, 22, 85, 253]])
        dfs.append(df_filtered)

    df_concat = pd.concat(dfs).sort_values(by=['date']).pivot(index='date', columns='security', values='PX_Last')[tickers]
    df_div = df_concat.div(df_concat.iloc[-1]) - 1  # Calculate percentage changes
    df_div.drop(df_div.tail(1).index, inplace=True)

    df_concat.to_csv('breakpoints.csv')
    df_div.to_csv('breakpoints_pct.csv')

    return df_concat

if __name__ == "__main__":
    source = pd.read_csv('data_pull_updated.csv')
    test = tsmom_port(source, [21, 84, 252], d_tickers)
    test2 = tsmom_breakpoints([21, 84, 252], d_tickers)
