import math
import numpy as np
import pandas as pd


def multi_shift(df, shift=3):
    """Get last N rows relative to another row in pandas"""
    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)

    dfs = [df.shift(i) for i in np.arange(shift)]
    for ix, dfi in enumerate(dfs[1:]):
        dfs[ix + 1].columns = [str(col) for col in dfi.columns + str(ix + 1)]
    return pd.concat(dfs, axis=1, sort=True)


def get_trading_periods(periods_per_year=252):
    half_year = math.ceil(periods_per_year / 2)
    return periods_per_year, half_year


def match_dates(returns, benchmark):
    if isinstance(returns, pd.DataFrame):
        loc = max(returns[returns.columns[0]].ne(0).idxmax(), benchmark.ne(0).idxmax())
    else:
        loc = max(returns.ne(0).idxmax(), benchmark.ne(0).idxmax())
    returns = returns.loc[loc:]
    benchmark = benchmark.loc[loc:]

    return returns, benchmark
