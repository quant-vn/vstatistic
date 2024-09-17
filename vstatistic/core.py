import math
import inspect
import numpy as np
import pandas as pd
from scipy.stats import norm
from pandas.core.base import PandasObject

from . import utils


def pct_rank(prices, window=60):
    """Rank prices by window"""
    rank = utils.multi_shift(prices, window).T.rank(pct=True).T
    return rank.iloc[:, 0] * 100.0


def compsum(returns):
    """Calculates rolling compounded returns"""
    return returns.add(1).cumprod() - 1


def comp(returns):
    """Calculates total compounded returns"""
    return returns.add(1).prod() - 1


def to_excess_returns(returns, rf, nperiods=None):
    """
    Calculates excess returns by subtracting
    risk-free returns from total returns

    Args:
        * returns (Series, DataFrame): Returns
        * rf (float, Series, DataFrame): Risk-Free rate(s)
        * nperiods (int): Optional. If provided, will convert rf to different
            frequency using deannualize
    Returns:
        * excess_returns (Series, DataFrame): Returns - rf
    """
    if isinstance(rf, int):
        rf = float(rf)

    if not isinstance(rf, float):
        rf = rf[rf.index.isin(returns.index)]

    if nperiods is not None:
        # deannualize
        rf = np.power(1 + rf, 1.0 / nperiods) - 1.0

    return returns - rf


def prepare_returns(data, rf=0.0, nperiods=None):
    """Converts price data into returns + cleanup"""
    data = data.copy()
    function = inspect.stack()[1][3]
    if isinstance(data, pd.DataFrame):
        for col in data.columns:
            if data[col].dropna().min() >= 0 and data[col].dropna().max() > 1:
                data[col] = data[col].pct_change()
    elif data.min() >= 0 and data.max() > 1:
        data = data.pct_change()

    # cleanup data
    data = data.replace([np.inf, -np.inf], float("NaN"))

    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.fillna(0).replace([np.inf, -np.inf], float("NaN"))
    unnecessary_function_calls = [
        "_prepare_benchmark",
        "cagr",
        "gain_to_pain_ratio",
        "rolling_volatility",
    ]

    if function not in unnecessary_function_calls:
        if rf > 0:
            return to_excess_returns(data, rf, nperiods)
    return data


def group_returns(returns, groupby, compounded=False):
    """Summarize returns
    group_returns(df, df.index.year)
    group_returns(df, [df.index.year, df.index.month])
    """
    if compounded:
        return returns.groupby(groupby).apply(comp)
    return returns.groupby(groupby).sum()


def aggregate_returns(returns, period=None, compounded=True):
    """Aggregates returns based on date periods"""
    if period is None or "day" in period:
        return returns
    index = returns.index

    if "month" in period:
        return group_returns(returns, index.month, compounded=compounded)

    if "quarter" in period:
        return group_returns(returns, index.quarter, compounded=compounded)

    if period == "A" or any(x in period for x in ["year", "eoy", "yoy"]):
        return group_returns(returns, index.year, compounded=compounded)

    if "week" in period:
        return group_returns(returns, index.week, compounded=compounded)

    if "eow" in period or period == "W":
        return group_returns(returns, [index.year, index.week], compounded=compounded)

    if "eom" in period or period == "M":
        return group_returns(returns, [index.year, index.month], compounded=compounded)

    if "eoq" in period or period == "Q":
        return group_returns(
            returns, [index.year, index.quarter], compounded=compounded
        )

    if not isinstance(period, str):
        return group_returns(returns, period, compounded)

    return returns


def expected_return(returns, aggregate=None, compounded=True, is_prepare_returns=True):
    """
    Returns the expected return for a given period
    by calculating the geometric holding period return
    """
    if is_prepare_returns:
        returns = prepare_returns(returns)
    returns = aggregate_returns(returns, aggregate, compounded)
    return np.prod(1 + returns, axis=0) ** (1 / len(returns)) - 1


def distribution(returns, compounded=True, is_prepare_returns=True):
    def get_outliers(data):
        # https://datascience.stackexchange.com/a/57199
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1  # IQR is interquartile range.
        filtered = (data >= Q1 - 1.5 * IQR) & (data <= Q3 + 1.5 * IQR)
        return {
            "values": data.loc[filtered].tolist(),
            "outliers": data.loc[~filtered].tolist(),
        }

    if isinstance(returns, pd.DataFrame):
        returns = returns.copy()
        returns.columns = map(str.lower, returns.columns)
        if len(returns.columns) > 1 and "close" in returns.columns:
            returns = returns["close"]
        else:
            returns = returns[returns.columns[0]]

    apply_fnc = comp if compounded else np.sum
    daily = returns.dropna()

    if is_prepare_returns:
        daily = prepare_returns(daily)

    return {
        "Daily": get_outliers(daily),
        "Weekly": get_outliers(daily.resample("W-MON").apply(apply_fnc)),
        "Monthly": get_outliers(daily.resample("M").apply(apply_fnc)),
        "Quarterly": get_outliers(daily.resample("Q").apply(apply_fnc)),
        "Yearly": get_outliers(daily.resample("A").apply(apply_fnc)),
    }


def geometric_mean(retruns, aggregate=None, compounded=True):
    """Shorthand for expected_return()"""
    return expected_return(retruns, aggregate, compounded)


def ghpr(retruns, aggregate=None, compounded=True):
    """Shorthand for expected_return()"""
    return expected_return(retruns, aggregate, compounded)


def outliers(returns, quantile=0.95):
    """Returns series of outliers"""
    return returns[returns > returns.quantile(quantile)].dropna(how="all")


def remove_outliers(returns, quantile=0.95):
    """Returns series of returns without the outliers"""
    return returns[returns < returns.quantile(quantile)]


def best(returns, aggregate=None, compounded=True, is_prepare_returns=True):
    """Returns the best day/month/week/quarter/year's return"""
    if is_prepare_returns:
        returns = prepare_returns(returns)
    return aggregate_returns(returns, aggregate, compounded).max()


def worst(returns, aggregate=None, compounded=True, is_prepare_returns=True):
    """Returns the worst day/month/week/quarter/year's return"""
    if is_prepare_returns:
        returns = prepare_returns(returns)
    return aggregate_returns(returns, aggregate, compounded).min()


def count_consecutive(data):
    """Counts consecutive data (like cumsum() with reset on zeroes)"""

    def _count(data):
        return data * (data.groupby((data != data.shift(1)).cumsum()).cumcount() + 1)

    if isinstance(data, pd.DataFrame):
        for col in data.columns:
            data[col] = _count(data[col])
        return data
    return _count(data)


def consecutive_wins(returns, aggregate=None, compounded=True, is_prepare_returns=True):
    """Returns the maximum consecutive wins by day/month/week/quarter/year"""
    if is_prepare_returns:
        returns = prepare_returns(returns)
    returns = aggregate_returns(returns, aggregate, compounded) > 0
    return count_consecutive(returns).max()


def consecutive_losses(returns, aggregate=None, compounded=True, is_prepare_returns=True):
    """
    Returns the maximum consecutive losses by
    day/month/week/quarter/year
    """
    if is_prepare_returns:
        returns = prepare_returns(returns)
    returns = aggregate_returns(returns, aggregate, compounded) < 0
    return count_consecutive(returns).max()


def exposure(returns, is_prepare_returns=True):
    """Returns the market exposure time (returns != 0)"""
    if is_prepare_returns:
        returns = prepare_returns(returns)

    def _exposure(ret):
        ex = len(ret[(~np.isnan(ret)) & (ret != 0)]) / len(ret)
        return math.ceil(ex * 100) / 100

    if isinstance(returns, pd.DataFrame):
        _df = {}
        for col in returns.columns:
            _df[col] = _exposure(returns[col])
        return pd.Series(_df)
    return _exposure(returns)


def win_rate(returns, aggregate=None, compounded=True, is_prepare_returns=True):
    """Calculates the win ratio for a period"""

    def _win_rate(series):
        try:
            return len(series[series > 0]) / len(series[series != 0])
        except Exception:
            return 0.0

    if is_prepare_returns:
        returns = prepare_returns(returns)
    if aggregate:
        returns = aggregate_returns(returns, aggregate, compounded)

    if isinstance(returns, pd.DataFrame):
        _df = {}
        for col in returns.columns:
            _df[col] = _win_rate(returns[col])

        return pd.Series(_df)

    return _win_rate(returns)


def avg_return(returns, aggregate=None, compounded=True, is_prepare_returns=True):
    """Calculates the average return/trade return for a period"""
    if is_prepare_returns:
        returns = prepare_returns(returns)
    if aggregate:
        returns = aggregate_returns(returns, aggregate, compounded)
    return returns[returns != 0].dropna().mean()


def avg_win(returns, aggregate=None, compounded=True, is_prepare_returns=True):
    """
    Calculates the average winning
    return/trade return for a period
    """
    if is_prepare_returns:
        returns = prepare_returns(returns)
    if aggregate:
        returns = aggregate_returns(returns, aggregate, compounded)
    return returns[returns > 0].dropna().mean()


def avg_loss(returns, aggregate=None, compounded=True, is_prepare_returns=True):
    """
    Calculates the average low if
    return/trade return for a period
    """
    if is_prepare_returns:
        returns = prepare_returns(returns)
    if aggregate:
        returns = aggregate_returns(returns, aggregate, compounded)
    return returns[returns < 0].dropna().mean()


def volatility(returns, periods=252, annualize=True, is_prepare_returns=True):
    """Calculates the volatility of returns for a period"""
    if is_prepare_returns:
        returns = prepare_returns(returns)
    std = returns.std()
    if annualize:
        return std * np.sqrt(periods)

    return std


def rolling_volatility(returns, rolling_period=126, periods_per_year=252, is_prepare_returns=True):
    if is_prepare_returns:
        returns = prepare_returns(returns, rolling_period)

    return returns.rolling(rolling_period).std() * np.sqrt(periods_per_year)


def to_log_returns(returns, rf=0.0, nperiods=None):
    """Converts returns series to log returns"""
    returns = prepare_returns(returns, rf, nperiods)
    try:
        return np.log(returns + 1).replace([np.inf, -np.inf], float("NaN"))
    except Exception:
        return 0.0


def log_returns(returns, rf=0.0, nperiods=None):
    """Shorthand for to_log_returns"""
    return to_log_returns(returns, rf, nperiods)


def implied_volatility(returns, periods=252, annualize=True):
    """Calculates the implied volatility of returns for a period"""
    logret = log_returns(returns)
    if annualize:
        return logret.rolling(periods).std() * np.sqrt(periods)
    return logret.std()


def autocorr_penalty(returns, is_prepare_returns=False):
    """Metric to account for auto correlation"""
    if is_prepare_returns:
        returns = prepare_returns(returns)

    if isinstance(returns, pd.DataFrame):
        returns = returns[returns.columns[0]]

    num = len(returns)
    coef = np.abs(np.corrcoef(returns[:-1], returns[1:])[0, 1])
    corr = [((num - x) / num) * coef**x for x in range(1, num)]
    return np.sqrt(1 + 2 * np.sum(corr))


def sharpe(returns, rf=0.0, periods=252, annualize=True, smart=False):
    """
    Calculates the sharpe ratio of access returns

    If rf is non-zero, you must specify periods.
    In this case, rf is assumed to be expressed in yearly (annualized) terms

    Args:
        * returns (Series, DataFrame): Input return series
        * rf (float): Risk-free rate expressed as a yearly (annualized) return
        * periods (int): Freq. of returns (252/365 for daily, 12 for monthly)
        * annualize: return annualize sharpe?
        * smart: return smart sharpe ratio
    """
    if rf != 0 and periods is None:
        raise Exception("Must provide periods if rf != 0")

    returns = prepare_returns(returns, rf, periods)
    divisor = returns.std(ddof=1)
    if smart:
        # penalize sharpe with auto correlation
        divisor = divisor * autocorr_penalty(returns)
    res = returns.mean() / divisor

    if annualize:
        return res * np.sqrt(1 if periods is None else periods)

    return res


def smart_sharpe(returns, rf=0.0, periods=252, annualize=True):
    return sharpe(returns, rf, periods, annualize, True)


def rolling_sharpe(
    returns,
    rf=0.0,
    rolling_period=126,
    annualize=True,
    periods_per_year=252,
    is_prepare_returns=True,
):

    if rf != 0 and rolling_period is None:
        raise Exception("Must provide periods if rf != 0")

    if is_prepare_returns:
        returns = prepare_returns(returns, rf, rolling_period)

    res = returns.rolling(rolling_period).mean() / returns.rolling(rolling_period).std()

    if annualize:
        res = res * np.sqrt(1 if periods_per_year is None else periods_per_year)
    return res


def sortino(returns, rf=0, periods=252, annualize=True, smart=False):
    """
    Calculates the sortino ratio of access returns

    If rf is non-zero, you must specify periods.
    In this case, rf is assumed to be expressed in yearly (annualized) terms

    Calculation is based on this paper by Red Rock Capital
    http://www.redrockcapital.com/Sortino__A__Sharper__Ratio_Red_Rock_Capital.pdf
    """
    if rf != 0 and periods is None:
        raise Exception("Must provide periods if rf != 0")

    returns = prepare_returns(returns, rf, periods)

    downside = np.sqrt((returns[returns < 0] ** 2).sum() / len(returns))

    if smart:
        # penalize sortino with auto correlation
        downside = downside * autocorr_penalty(returns)

    res = returns.mean() / downside

    if annualize:
        return res * np.sqrt(1 if periods is None else periods)

    return res


def smart_sortino(returns, rf=0, periods=252, annualize=True):
    return sortino(returns, rf, periods, annualize, True)


def rolling_sortino(
    returns, rf=0, rolling_period=126, annualize=True, periods_per_year=252, **kwargs
):
    if rf != 0 and rolling_period is None:
        raise Exception("Must provide periods if rf != 0")

    if kwargs.get("prepare_returns", True):
        returns = prepare_returns(returns, rf, rolling_period)

    downside = (
        returns.rolling(rolling_period).apply(
            lambda x: (x.values[x.values < 0] ** 2).sum()
        )
        / rolling_period
    )

    res = returns.rolling(rolling_period).mean() / np.sqrt(downside)
    if annualize:
        res = res * np.sqrt(1 if periods_per_year is None else periods_per_year)
    return res


def adjusted_sortino(returns, rf=0, periods=252, annualize=True, smart=False):
    """
    Jack Schwager's version of the Sortino ratio allows for
    direct comparisons to the Sharpe. See here for more info:
    https://archive.is/wip/2rwFW
    """
    data = sortino(returns, rf, periods=periods, annualize=annualize, smart=smart)
    return data / math.sqrt(2)


def skew(returns, is_prepare_returns=True):
    """
    Calculates returns' skewness
    (the degree of asymmetry of a distribution around its mean)
    """
    if is_prepare_returns:
        returns = prepare_returns(returns)
    return returns.skew()


def kurtosis(returns, is_prepare_returns=True):
    """
    Calculates returns' kurtosis
    (the degree to which a distribution peak compared to a normal distribution)
    """
    if is_prepare_returns:
        returns = prepare_returns(returns)
    return returns.kurtosis()


def probabilistic_ratio(series, rf=0.0, base="sharpe", periods=252, annualize=False, smart=False):
    if base.lower() == "sharpe":
        base = sharpe(series, periods=periods, annualize=False, smart=smart)
    elif base.lower() == "sortino":
        base = sortino(series, periods=periods, annualize=False, smart=smart)
    elif base.lower() == "adjusted_sortino":
        base = adjusted_sortino(series, periods=periods, annualize=False, smart=smart)
    else:
        raise Exception(
            "`metric` must be either `sharpe`, `sortino`, or `adjusted_sortino`"
        )
    skew_no = skew(series, is_prepare_returns=False)
    kurtosis_no = kurtosis(series, is_prepare_returns=False)

    n = len(series)

    sigma_sr = np.sqrt(
        (
            1
            + (0.5 * base**2)
            - (skew_no * base)
            + (((kurtosis_no - 3) / 4) * base**2)
        )
        / (n - 1)
    )

    ratio = (base - rf) / sigma_sr
    psr = norm.cdf(ratio)

    if annualize:
        return psr * (252**0.5)
    return psr


def probabilistic_sharpe_ratio(
    series, rf=0.0, periods=252, annualize=False, smart=False
):
    return probabilistic_ratio(
        series, rf, base="sharpe", periods=periods, annualize=annualize, smart=smart
    )


def probabilistic_sortino_ratio(
    series, rf=0.0, periods=252, annualize=False, smart=False
):
    return probabilistic_ratio(
        series, rf, base="sortino", periods=periods, annualize=annualize, smart=smart
    )


def probabilistic_adjusted_sortino_ratio(
    series, rf=0.0, periods=252, annualize=False, smart=False
):
    return probabilistic_ratio(
        series,
        rf,
        base="adjusted_sortino",
        periods=periods,
        annualize=annualize,
        smart=smart,
    )


def omega(returns, rf=0.0, required_return=0.0, periods=252):
    """
    Determines the Omega ratio of a strategy.
    See https://en.wikipedia.org/wiki/Omega_ratio for more details.
    """
    if len(returns) < 2:
        return np.nan

    if required_return <= -1:
        return np.nan

    returns = prepare_returns(returns, rf, periods)

    if periods == 1:
        return_threshold = required_return
    else:
        return_threshold = (1 + required_return) ** (1.0 / periods) - 1

    returns_less_thresh = returns - return_threshold
    # numer = returns_less_thresh[returns_less_thresh > 0.0].sum()
    numer = returns_less_thresh[returns_less_thresh > 0.0].sum().values[0]
    # denom = -1.0 * returns_less_thresh[returns_less_thresh < 0.0].sum()
    denom = -1.0 * returns_less_thresh[returns_less_thresh < 0.0].sum().values[0]

    if denom > 0.0:
        return numer / denom

    return np.nan


def gain_to_pain_ratio(returns, rf=0, resolution="D"):
    """
    Jack Schwager's GPR. See here for more info:
    https://archive.is/wip/2rwFW
    """
    returns = prepare_returns(returns, rf).resample(resolution).sum()
    downside = abs(returns[returns < 0].sum())
    return returns.sum() / downside


def cagr(returns, rf=0.0, compounded=True, periods=252):
    """
    Calculates the communicative annualized growth return
    (CAGR%) of access returns

    If rf is non-zero, you must specify periods.
    In this case, rf is assumed to be expressed in yearly (annualized) terms
    """
    total = prepare_returns(returns, rf)
    if compounded:
        total = comp(total)
    else:
        total = np.sum(total)

    years = (returns.index[-1] - returns.index[0]).days / periods

    res = abs(total + 1.0) ** (1.0 / years) - 1

    if isinstance(returns, pd.DataFrame):
        res = pd.Series(res)
        res.index = returns.columns

    return res


def rar(returns, rf=0.0):
    """
    Calculates the risk-adjusted return of access returns
    (CAGR / exposure. takes time into account.)

    If rf is non-zero, you must specify periods.
    In this case, rf is assumed to be expressed in yearly (annualized) terms
    """
    returns = prepare_returns(returns, rf)
    return cagr(returns) / exposure(returns)


def to_prices(returns, base=1e5):
    """Converts returns series to price data"""
    returns = returns.copy().fillna(0).replace([np.inf, -np.inf], float("NaN"))

    return base + base * compsum(returns)


def prepare_prices(data, base=1.0):
    """Converts return data into prices + cleanup"""
    data = data.copy()
    if isinstance(data, pd.DataFrame):
        for col in data.columns:
            if data[col].dropna().min() <= 0 or data[col].dropna().max() < 1:
                data[col] = to_prices(data[col], base)

    # is it returns?
    # elif data.min() < 0 and data.max() < 1:
    elif data.min() < 0 or data.max() < 1:
        data = to_prices(data, base)

    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.fillna(0).replace([np.inf, -np.inf], float("NaN"))

    return data


def max_drawdown(prices):
    """Calculates the maximum drawdown"""
    prices = prepare_prices(prices)
    return (prices / prices.expanding(min_periods=0).max()).min() - 1


def to_drawdown_series(returns):
    """Convert returns series to drawdown series"""
    prices = prepare_prices(returns)
    dd = prices / np.maximum.accumulate(prices) - 1.0
    return dd.replace([np.inf, -np.inf, -0], 0)


def calmar(returns, is_prepare_returns=True):
    """Calculates the calmar ratio (CAGR% / MaxDD%)"""
    if is_prepare_returns:
        returns = prepare_returns(returns)
    cagr_ratio = cagr(returns)
    max_dd = max_drawdown(returns)
    return cagr_ratio / abs(max_dd)


def ulcer_index(returns):
    """Calculates the ulcer index score (downside risk measurment)"""
    dd = to_drawdown_series(returns)
    return np.sqrt(np.divide((dd**2).sum(), returns.shape[0] - 1))


def ulcer_performance_index(returns, rf=0):
    """
    Calculates the ulcer index score
    (downside risk measurment)
    """
    return (comp(returns) - rf) / ulcer_index(returns)


def upi(returns, rf=0):
    """Shorthand for ulcer_performance_index()"""
    return ulcer_performance_index(returns, rf)


def value_at_risk(returns, sigma=1, confidence=0.95, is_prepare_returns=True):
    """
    Calculats the daily value-at-risk
    (variance-covariance calculation with confidence n)
    """
    if is_prepare_returns:
        returns = prepare_returns(returns)
    mu = returns.mean()
    sigma *= returns.std()

    if confidence > 1:
        confidence = confidence / 100

    return norm.ppf(1 - confidence, mu, sigma)


def var(returns, sigma=1, confidence=0.95, is_prepare_returns=True):
    """Shorthand for value_at_risk()"""
    return value_at_risk(returns, sigma, confidence, is_prepare_returns)


def conditional_value_at_risk(returns, sigma=1, confidence=0.95, is_prepare_returns=True):
    """
    Calculats the conditional daily value-at-risk (aka expected shortfall)
    quantifies the amount of tail risk an investment
    """
    if is_prepare_returns:
        returns = prepare_returns(returns)
    var = value_at_risk(returns, sigma, confidence)
    c_var = returns[returns < var].values.mean()
    return c_var if ~np.isnan(c_var) else var


def cvar(returns, sigma=1, confidence=0.95, is_prepare_returns=True):
    """Shorthand for conditional_value_at_risk()"""
    return conditional_value_at_risk(returns, sigma, confidence, is_prepare_returns)


def serenity_index(returns, rf=0):
    """
    Calculates the serenity index score
    (https://www.keyquant.com/Download/GetFile?Filename=%5CPublications%5CKeyQuant_WhitePaper_APT_Part1.pdf)
    """
    dd = to_drawdown_series(returns)
    pitfall = -cvar(dd) / returns.std()
    return (returns.sum() - rf) / (ulcer_index(returns) * pitfall)


def risk_of_ruin(returns, is_prepare_returns=True):
    """
    Calculates the risk of ruin
    (the likelihood of losing all one's investment capital)
    """
    if is_prepare_returns:
        returns = prepare_returns(returns)
    wins = win_rate(returns)
    return ((1 - wins) / (1 + wins)) ** len(returns)


def ror(returns):
    """Shorthand for risk_of_ruin()"""
    return risk_of_ruin(returns)


def expected_shortfall(returns, sigma=1, confidence=0.95):
    """Shorthand for conditional_value_at_risk()"""
    return conditional_value_at_risk(returns, sigma, confidence)


def tail_ratio(returns, cutoff=0.95, is_prepare_returns=True):
    """
    Measures the ratio between the right
    (95%) and left tail (5%).
    """
    if is_prepare_returns:
        returns = prepare_returns(returns)
    return abs(returns.quantile(cutoff) / returns.quantile(1 - cutoff))


def payoff_ratio(returns, is_prepare_returns=True):
    """Measures the payoff ratio (average win/average loss)"""
    if is_prepare_returns:
        returns = prepare_returns(returns)
    return avg_win(returns) / abs(avg_loss(returns))


def win_loss_ratio(returns, is_prepare_returns=True):
    """Shorthand for payoff_ratio()"""
    return payoff_ratio(returns, is_prepare_returns)


def profit_ratio(returns, is_prepare_returns=True):
    """Measures the profit ratio (win ratio / loss ratio)"""
    if is_prepare_returns:
        returns = prepare_returns(returns)
    wins = returns[returns >= 0]
    loss = returns[returns < 0]

    win_ratio = abs(wins.mean() / wins.count())
    loss_ratio = abs(loss.mean() / loss.count())
    try:
        return win_ratio / loss_ratio
    except Exception:
        return 0.0


def profit_factor(returns, is_prepare_returns=True):
    """Measures the profit ratio (wins/loss)"""
    if is_prepare_returns:
        returns = prepare_returns(returns)
    return abs(returns[returns >= 0].sum() / returns[returns < 0].sum())


def cpc_index(returns, is_prepare_returns=True):
    """
    Measures the cpc ratio
    (profit factor * win % * win loss ratio)
    """
    if is_prepare_returns:
        returns = prepare_returns(returns)
    return profit_factor(returns) * win_rate(returns) * win_loss_ratio(returns)


def common_sense_ratio(returns, is_prepare_returns=True):
    """Measures the common sense ratio (profit factor * tail ratio)"""
    if is_prepare_returns:
        returns = prepare_returns(returns)
    return profit_factor(returns) * tail_ratio(returns)


def outlier_win_ratio(returns, quantile=0.99, is_prepare_returns=True):
    """
    Calculates the outlier winners ratio
    99th percentile of returns / mean positive return
    """
    if is_prepare_returns:
        returns = prepare_returns(returns)
    return returns.quantile(quantile).mean() / returns[returns >= 0].mean()


def outlier_loss_ratio(returns, quantile=0.01, is_prepare_returns=True):
    """
    Calculates the outlier losers ratio
    1st percentile of returns / mean negative return
    """
    if is_prepare_returns:
        returns = prepare_returns(returns)
    return returns.quantile(quantile).mean() / returns[returns < 0].mean()


def recovery_factor(returns, rf=0., is_prepare_returns=True):
    """Measures how fast the strategy recovers from drawdowns"""
    if is_prepare_returns:
        returns = prepare_returns(returns)
    total_returns = returns.sum() - rf
    max_dd = max_drawdown(returns)
    return abs(total_returns) / abs(max_dd)


def risk_return_ratio(returns, is_prepare_returns=True):
    """
    Calculates the return / risk ratio
    (sharpe ratio without factoring in the risk-free rate)
    """
    if is_prepare_returns:
        returns = prepare_returns(returns)
    return returns.mean() / returns.std()


def drawdown_details(drawdown):
    """
    Calculates drawdown details, including start/end/valley dates,
    duration, max drawdown and max dd for 99% of the dd period
    for every drawdown period
    """

    def _drawdown_details(drawdown):
        # mark no drawdown
        no_dd = drawdown == 0

        # extract dd start dates, first date of the drawdown
        starts = ~no_dd & no_dd.shift(1)
        starts = list(starts[starts.values].index)

        # extract end dates, last date of the drawdown
        ends = no_dd & (~no_dd).shift(1)
        ends = ends.shift(-1, fill_value=False)
        ends = list(ends[ends.values].index)

        # no drawdown :)
        if not starts:
            return pd.DataFrame(
                index=[],
                columns=(
                    "start",
                    "valley",
                    "end",
                    "days",
                    "max drawdown",
                    "99% max drawdown",
                ),
            )

        # drawdown series begins in a drawdown
        if ends and starts[0] > ends[0]:
            starts.insert(0, drawdown.index[0])

        # series ends in a drawdown fill with last date
        if not ends or starts[-1] > ends[-1]:
            ends.append(drawdown.index[-1])

        # build dataframe from results
        data = []
        for i, _ in enumerate(starts):
            dd = drawdown[starts[i]: ends[i]]
            clean_dd = -remove_outliers(-dd, 0.99)
            data.append(
                (
                    starts[i],
                    dd.idxmin(),
                    ends[i],
                    (ends[i] - starts[i]).days + 1,
                    dd.min() * 100,
                    clean_dd.min() * 100,
                )
            )

        df = pd.DataFrame(
            data=data,
            columns=(
                "start",
                "valley",
                "end",
                "days",
                "max drawdown",
                "99% max drawdown",
            ),
        )
        df["days"] = df["days"].astype(int)
        df["max drawdown"] = df["max drawdown"].astype(float)
        df["99% max drawdown"] = df["99% max drawdown"].astype(float)

        df["start"] = df["start"].dt.strftime("%Y-%m-%d")
        df["end"] = df["end"].dt.strftime("%Y-%m-%d")
        df["valley"] = df["valley"].dt.strftime("%Y-%m-%d")

        return df

    if isinstance(drawdown, pd.DataFrame):
        _dfs = {}
        for col in drawdown.columns:
            _dfs[col] = _drawdown_details(drawdown[col])
        return pd.concat(_dfs, axis=1)

    return _drawdown_details(drawdown)


def kelly_criterion(returns, is_prepare_returns=True):
    """
    Calculates the recommended maximum amount of capital that
    should be allocated to the given strategy, based on the
    Kelly Criterion (http://en.wikipedia.org/wiki/Kelly_criterion)
    """
    if is_prepare_returns:
        returns = prepare_returns(returns)
    win_loss_ratio = payoff_ratio(returns)
    win_prob = win_rate(returns)
    lose_prob = 1 - win_prob

    return ((win_loss_ratio * win_prob) - lose_prob) / win_loss_ratio


def extend_pandas():
    PandasObject.compsum = compsum
    PandasObject.comp = comp
    PandasObject.expected_return = expected_return
    PandasObject.geometric_mean = geometric_mean
    PandasObject.ghpr = ghpr
    PandasObject.outliers = outliers
    PandasObject.remove_outliers = remove_outliers
    PandasObject.best = best
    PandasObject.worst = worst
    PandasObject.consecutive_wins = consecutive_wins
    PandasObject.consecutive_losses = consecutive_losses
    PandasObject.exposure = exposure
    PandasObject.win_rate = win_rate
    PandasObject.avg_return = avg_return
    PandasObject.avg_win = avg_win
    PandasObject.avg_loss = avg_loss
    PandasObject.volatility = volatility
    PandasObject.rolling_volatility = rolling_volatility
    PandasObject.implied_volatility = implied_volatility
    # PandasObject.autocorr_penalty = autocorr_penalty
    PandasObject.sharpe = sharpe
    PandasObject.smart_sharpe = smart_sharpe
    PandasObject.rolling_sharpe = rolling_sharpe
    PandasObject.sortino = sortino
    PandasObject.smart_sortino = smart_sortino
    PandasObject.rolling_sortino = rolling_sortino
    PandasObject.adjusted_sortino = adjusted_sortino
    PandasObject.omega = omega
    PandasObject.cagr = cagr
    PandasObject.rar = rar
    PandasObject.skew = skew
    PandasObject.kurtosis = kurtosis
    PandasObject.calmar = calmar
    PandasObject.ulcer_index = ulcer_index
    PandasObject.ulcer_performance_index = ulcer_performance_index
    PandasObject.upi = upi
    PandasObject.serenity_index = serenity_index
    PandasObject.risk_of_ruin = risk_of_ruin
    PandasObject.ror = ror
    PandasObject.value_at_risk = value_at_risk
    PandasObject.var = var
    PandasObject.conditional_value_at_risk = conditional_value_at_risk
    PandasObject.cvar = cvar
    PandasObject.expected_shortfall = expected_shortfall
    PandasObject.tail_ratio = tail_ratio
    PandasObject.payoff_ratio = payoff_ratio
    PandasObject.win_loss_ratio = win_loss_ratio
    PandasObject.profit_ratio = profit_ratio
    PandasObject.profit_factor = profit_factor
    PandasObject.gain_to_pain_ratio = gain_to_pain_ratio
    PandasObject.cpc_index = cpc_index
    PandasObject.common_sense_ratio = common_sense_ratio
    PandasObject.outlier_win_ratio = outlier_win_ratio
    PandasObject.outlier_loss_ratio = outlier_loss_ratio
    PandasObject.recovery_factor = recovery_factor
    PandasObject.risk_return_ratio = risk_return_ratio
    PandasObject.max_drawdown = max_drawdown
    PandasObject.to_drawdown_series = to_drawdown_series
    PandasObject.drawdown_details = drawdown_details
    PandasObject.kelly_criterion = kelly_criterion
    PandasObject.pct_rank = pct_rank
    PandasObject.probabilistic_sharpe_ratio = probabilistic_sharpe_ratio
    PandasObject.probabilistic_sortino_ratio = probabilistic_sortino_ratio
    PandasObject.probabilistic_adjusted_sortino_ratio = (
        probabilistic_adjusted_sortino_ratio
    )

    PandasObject.to_prices = to_prices
    PandasObject.to_log_returns = to_log_returns
    PandasObject.log_returns = log_returns
    PandasObject.aggregate_returns = aggregate_returns
    PandasObject.to_excess_returns = to_excess_returns
    PandasObject.multi_shift = utils.multi_shift
