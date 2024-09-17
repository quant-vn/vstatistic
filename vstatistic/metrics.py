import math
import numpy as np
import pandas as pd
from datetime import datetime
from packaging.version import Version
from dateutil.relativedelta import relativedelta

from . import core, utils

PANDAS_VERSION_2 = Version(pd.__version__) >= Version("2")
PANDAS_VERSION_2_2 = Version(pd.__version__) >= Version("2.2")

if PANDAS_VERSION_2_2:
    MONTH_END = "ME"
    YEAR_END = "YE"
else:
    MONTH_END = "M"
    YEAR_END = "Y"


def _calc_dd(df, display=True, as_pct=False):
    dd = core.to_drawdown_series(df)
    dd_info = core.drawdown_details(dd)

    if dd_info.empty:
        return pd.DataFrame()

    if "returns" in dd_info:
        ret_dd = dd_info["returns"]
    # to match multiple columns like returns_1, returns_2, ...
    elif (
        any(dd_info.columns.get_level_values(0).str.contains("returns"))
        and dd_info.columns.get_level_values(0).nunique() > 1
    ):
        ret_dd = dd_info.loc[
            :, dd_info.columns.get_level_values(0).str.contains("returns")
        ]
    else:
        ret_dd = dd_info

    if (
        any(ret_dd.columns.get_level_values(0).str.contains("returns"))
        and ret_dd.columns.get_level_values(0).nunique() > 1
    ):
        dd_stats = {
            col: {
                "Max Drawdown %": ret_dd[col]
                .sort_values(by="max drawdown", ascending=True)["max drawdown"]
                .values[0]
                / 100,
                "Longest DD Days": str(
                    np.round(
                        ret_dd[col]
                        .sort_values(by="days", ascending=False)["days"]
                        .values[0]
                    )
                ),
                "Avg. Drawdown %": ret_dd[col]["max drawdown"].mean() / 100,
                "Avg. Drawdown Days": str(np.round(ret_dd[col]["days"].mean())),
            }
            for col in ret_dd.columns.get_level_values(0)
        }
    else:
        dd_stats = {
            "returns": {
                "Max Drawdown %": ret_dd.sort_values(by="max drawdown", ascending=True)[
                    "max drawdown"
                ].values[0]
                / 100,
                "Longest DD Days": str(
                    np.round(
                        ret_dd.sort_values(by="days", ascending=False)["days"].values[0]
                    )
                ),
                "Avg. Drawdown %": ret_dd["max drawdown"].mean() / 100,
                "Avg. Drawdown Days": str(np.round(ret_dd["days"].mean())),
            }
        }
    if "benchmark" in df and (dd_info.columns, pd.MultiIndex):
        bench_dd = dd_info["benchmark"].sort_values(by="max drawdown")
        dd_stats["benchmark"] = {
            "Max Drawdown %": bench_dd.sort_values(by="max drawdown", ascending=True)[
                "max drawdown"
            ].values[0]
            / 100,
            "Longest DD Days": str(
                np.round(
                    bench_dd.sort_values(by="days", ascending=False)["days"].values[0]
                )
            ),
            "Avg. Drawdown %": bench_dd["max drawdown"].mean() / 100,
            "Avg. Drawdown Days": str(np.round(bench_dd["days"].mean())),
        }

    # pct multiplier
    pct = 100 if display or as_pct else 1

    dd_stats = pd.DataFrame(dd_stats).T
    dd_stats["Max Drawdown %"] = dd_stats["Max Drawdown %"].astype(float) * pct
    dd_stats["Avg. Drawdown %"] = dd_stats["Avg. Drawdown %"].astype(float) * pct

    return dd_stats.T


def metrics(
    returns,
    benchmark=None,
    rf=0.0,
    display=True,
    mode="basic",
    sep=False,
    compounded=True,
    periods_per_year=252,
    is_prepare_returns=True,
    match_dates=True,
    **kwargs,
):
    if match_dates:
        returns = returns.dropna()
    returns.index = returns.index.tz_localize(None)
    win_year, _ = utils.get_trading_periods(periods_per_year)
    benchmark_colname = kwargs.get("benchmark_title", "Benchmark")
    strategy_colname = kwargs.get("strategy_title", "Strategy")
    if benchmark is not None:
        if isinstance(benchmark, str):
            benchmark_colname = f"Benchmark ({benchmark.upper()})"
        elif isinstance(benchmark, pd.DataFrame) and len(benchmark.columns) > 1:
            raise ValueError(
                "`benchmark` must be a pandas Series, "
                "but a multi-column DataFrame was passed"
            )
    if isinstance(returns, pd.DataFrame):
        if len(returns.columns) > 1:
            blank = [""] * len(returns.columns)
            if isinstance(strategy_colname, str):
                strategy_colname = list(returns.columns)
    else:
        blank = [""]
    if is_prepare_returns:
        df = core.prepare_returns(returns)
    if isinstance(returns, pd.Series):
        df = pd.DataFrame({"returns": returns})
    elif isinstance(returns, pd.DataFrame):
        df = pd.DataFrame(
            {
                "returns_" + str(i + 1): returns[strategy_col]
                for i, strategy_col in enumerate(returns.columns)
            }
        )
    if benchmark is not None:
        benchmark = core.prepare_benchmark(benchmark, returns.index, rf)
        if match_dates is True:
            returns, benchmark = utils.match_dates(returns, benchmark)
        df["benchmark"] = benchmark
        if isinstance(returns, pd.Series):
            blank = ["", ""]
            df["returns"] = returns
        elif isinstance(returns, pd.DataFrame):
            blank = [""] * len(returns.columns) + [""]
            for i, strategy_col in enumerate(returns.columns):
                df["returns_" + str(i + 1)] = returns[strategy_col]
    if isinstance(returns, pd.Series):
        s_start = {"returns": df["returns"].index.strftime("%Y-%m-%d")[0]}
        s_end = {"returns": df["returns"].index.strftime("%Y-%m-%d")[-1]}
        s_rf = {"returns": rf}
    elif isinstance(returns, pd.DataFrame):
        df_strategy_columns = [col for col in df.columns if col != "benchmark"]
        s_start = {
            strategy_col: df[strategy_col].dropna().index.strftime("%Y-%m-%d")[0]
            for strategy_col in df_strategy_columns
        }
        s_end = {
            strategy_col: df[strategy_col].dropna().index.strftime("%Y-%m-%d")[-1]
            for strategy_col in df_strategy_columns
        }
        s_rf = {strategy_col: rf for strategy_col in df_strategy_columns}

    if "benchmark" in df:
        s_start["benchmark"] = df["benchmark"].index.strftime("%Y-%m-%d")[0]
        s_end["benchmark"] = df["benchmark"].index.strftime("%Y-%m-%d")[-1]
        s_rf["benchmark"] = rf

    df = df.fillna(0)

    # pct multiplier
    pct = 100 if display or "internal" in kwargs else 1
    if kwargs.get("as_pct", False):
        pct = 100

    # return df
    dd = _calc_dd(
        df,
        display=(display or "internal" in kwargs),
        as_pct=kwargs.get("as_pct", False),
    )

    metrics = pd.DataFrame()
    metrics["Start Period"] = pd.Series(s_start)
    metrics["End Period"] = pd.Series(s_end)
    metrics["Risk-Free Rate %"] = pd.Series(s_rf) * 100
    metrics["Time in Market %"] = core.exposure(df, is_prepare_returns=False) * pct

    metrics["~"] = blank

    if compounded:
        metrics["Cumulative Return %"] = (core.comp(df) * pct).map("{:,.2f}".format)
    else:
        metrics["Total Return %"] = (df.sum() * pct).map("{:,.2f}".format)

    metrics["CAGR﹪%"] = core.cagr(df, rf, compounded) * pct

    metrics["~~~~~~~~~~~~~~"] = blank

    metrics["Sharpe"] = core.sharpe(df, rf, win_year, True)
    metrics["Prob. Sharpe Ratio %"] = (
        core.probabilistic_sharpe_ratio(df, rf, win_year, False) * pct
    )
    if mode.lower() == "full":
        metrics["Smart Sharpe"] = core.smart_sharpe(df, rf, win_year, True)

    metrics["Sortino"] = core.sortino(df, rf, win_year, True)
    if mode.lower() == "full":
        metrics["Smart Sortino"] = core.smart_sortino(df, rf, win_year, True)
    metrics["Sortino/√2"] = metrics["Sortino"] / math.sqrt(2)
    if mode.lower() == "full":
        metrics["Smart Sortino/√2"] = metrics["Smart Sortino"] / math.sqrt(2)
    metrics["Omega"] = core.omega(df, rf, 0.0, win_year)

    metrics["~~~~~~~~"] = blank
    metrics["Max Drawdown %"] = blank
    metrics["Longest DD Days"] = blank

    if mode.lower() == "full":
        if isinstance(returns, pd.Series):
            ret_vol = (
                core.volatility(df["returns"], win_year, True, is_prepare_returns=False)
                * pct
            )
        elif isinstance(returns, pd.DataFrame):
            ret_vol = [
                core.volatility(
                    df[strategy_col], win_year, True, is_prepare_returns=False
                )
                * pct
                for strategy_col in df_strategy_columns
            ]
        if "benchmark" in df:
            bench_vol = (
                core.volatility(
                    df["benchmark"], win_year, True, is_prepare_returns=False
                )
                * pct
            )

            vol_ = [ret_vol, bench_vol]
            if isinstance(ret_vol, list):
                metrics["Volatility (ann.) %"] = list(pd.core.common.flatten(vol_))
            else:
                metrics["Volatility (ann.) %"] = vol_

            if isinstance(returns, pd.Series):
                metrics["R^2"] = core.r_squared(
                    df["returns"], df["benchmark"], is_prepare_returns=False
                )
                metrics["Information Ratio"] = core.information_ratio(
                    df["returns"], df["benchmark"], is_prepare_returns=False
                )
            elif isinstance(returns, pd.DataFrame):
                metrics["R^2"] = (
                    [
                        core.r_squared(
                            df[strategy_col], df["benchmark"], is_prepare_returns=False
                        ).round(2)
                        for strategy_col in df_strategy_columns
                    ]
                ) + ["-"]
                metrics["Information Ratio"] = (
                    [
                        core.information_ratio(
                            df[strategy_col], df["benchmark"], is_prepare_returns=False
                        ).round(2)
                        for strategy_col in df_strategy_columns
                    ]
                ) + ["-"]
        else:
            if isinstance(returns, pd.Series):
                metrics["Volatility (ann.) %"] = [ret_vol]
            elif isinstance(returns, pd.DataFrame):
                metrics["Volatility (ann.) %"] = ret_vol

        metrics["Calmar"] = core.calmar(df, is_prepare_returns=False)
        metrics["Skew"] = core.skew(df, is_prepare_returns=False)
        metrics["Kurtosis"] = core.kurtosis(df, is_prepare_returns=False)

        metrics["~~~~~~~~~~"] = blank

        metrics["Expected Daily %%"] = (
            core.expected_return(df, compounded=compounded, is_prepare_returns=False) * pct
        )
        metrics["Expected Monthly %%"] = (
            core.expected_return(
                df, compounded=compounded, aggregate="M", is_prepare_returns=False
            ) * pct
        )
        metrics["Expected Yearly %%"] = (
            core.expected_return(
                df, compounded=compounded, aggregate="A", is_prepare_returns=False
            ) * pct
        )
        metrics["Kelly Criterion %"] = (core.kelly_criterion(df, is_prepare_returns=False) * pct)
        metrics["Risk of Ruin %"] = core.risk_of_ruin(df, is_prepare_returns=False)
        metrics["Daily Value-at-Risk %"] = -abs(core.var(df, is_prepare_returns=False) * pct)
        metrics["Expected Shortfall (cVaR) %"] = -abs(core.cvar(df, is_prepare_returns=False) * pct)

    metrics["~~~~~~"] = blank

    if mode.lower() == "full":
        metrics["Max Consecutive Wins *int"] = core.consecutive_wins(df)
        metrics["Max Consecutive Losses *int"] = core.consecutive_losses(df)

    metrics["Gain/Pain Ratio"] = core.gain_to_pain_ratio(df, rf)
    metrics["Gain/Pain (1M)"] = core.gain_to_pain_ratio(df, rf, MONTH_END)
    # if mode.lower() == 'full':
    #     metrics['GPR (3M)'] = _stats.gain_to_pain_ratio(df, rf, "Q")
    #     metrics['GPR (6M)'] = _stats.gain_to_pain_ratio(df, rf, "2Q")
    #     metrics['GPR (1Y)'] = _stats.gain_to_pain_ratio(df, rf, "A")
    metrics["~~~~~~~"] = blank

    metrics["Payoff Ratio"] = core.payoff_ratio(df, is_prepare_returns=False)
    metrics["Profit Factor"] = core.profit_factor(df, is_prepare_returns=False)
    metrics["Common Sense Ratio"] = core.common_sense_ratio(df, is_prepare_returns=False)
    metrics["CPC Index"] = core.cpc_index(df, is_prepare_returns=False)
    metrics["Tail Ratio"] = core.tail_ratio(df, is_prepare_returns=False)
    metrics["Outlier Win Ratio"] = core.outlier_win_ratio(df, is_prepare_returns=False)
    metrics["Outlier Loss Ratio"] = core.outlier_loss_ratio(df, is_prepare_returns=False)

    # returns
    metrics["~~"] = blank
    comp_func = core.comp if compounded else np.sum

    today = df.index[-1]
    metrics["MTD %"] = comp_func(df[df.index >= datetime(today.year, today.month, 1)]) * pct

    d = today - relativedelta(months=3)
    metrics["3M %"] = comp_func(df[df.index >= d]) * pct
    d = today - relativedelta(months=6)
    metrics["6M %"] = comp_func(df[df.index >= d]) * pct
    metrics["YTD %"] = comp_func(df[df.index >= datetime(today.year, 1, 1)]) * pct
    d = today - relativedelta(years=1)
    metrics["1Y %"] = comp_func(df[df.index >= d]) * pct
    d = today - relativedelta(months=35)
    metrics["3Y (ann.) %"] = core.cagr(df[df.index >= d], 0.0, compounded) * pct
    d = today - relativedelta(months=59)
    metrics["5Y (ann.) %"] = core.cagr(df[df.index >= d], 0.0, compounded) * pct
    d = today - relativedelta(years=10)
    metrics["10Y (ann.) %"] = core.cagr(df[df.index >= d], 0.0, compounded) * pct
    metrics["All-time (ann.) %"] = core.cagr(df, 0.0, compounded) * pct

    # best/worst
    if mode.lower() == "full":
        metrics["~~~"] = blank
        metrics["Best Day %"] = core.best(df, compounded=compounded, is_prepare_returns=False) * pct
        metrics["Worst Day %"] = core.worst(df, is_prepare_returns=False) * pct
        metrics["Best Month %"] = (
            core.best(df, compounded=compounded, aggregate="M", is_prepare_returns=False) * pct
        )
        metrics["Worst Month %"] = (
            core.worst(df, aggregate="M", is_prepare_returns=False) * pct
        )
        metrics["Best Year %"] = (
            core.best(df, compounded=compounded, aggregate="A", is_prepare_returns=False) * pct
        )
        metrics["Worst Year %"] = (
            core.worst(df, compounded=compounded, aggregate="A", is_prepare_returns=False) * pct
        )

    # dd
    metrics["~~~~"] = blank
    for ix, row in dd.iterrows():
        metrics[ix] = row
    metrics["Recovery Factor"] = core.recovery_factor(df)
    metrics["Ulcer Index"] = core.ulcer_index(df)
    metrics["Serenity Index"] = core.serenity_index(df, rf)

    # win rate
    if mode.lower() == "full":
        metrics["~~~~~"] = blank
        metrics["Avg. Up Month %"] = (
            core.avg_win(df, compounded=compounded, aggregate="M", is_prepare_returns=False) * pct
        )
        metrics["Avg. Down Month %"] = (
            core.avg_loss(df, compounded=compounded, aggregate="M", is_prepare_returns=False) * pct
        )
        metrics["Win Days %%"] = core.win_rate(df, is_prepare_returns=False) * pct
        metrics["Win Month %%"] = (
            core.win_rate(df, compounded=compounded, aggregate="M", is_prepare_returns=False) * pct
        )
        metrics["Win Quarter %%"] = (
            core.win_rate(df, compounded=compounded, aggregate="Q", is_prepare_returns=False) * pct
        )
        metrics["Win Year %%"] = (
            core.win_rate(df, compounded=compounded, aggregate="A", is_prepare_returns=False) * pct
        )

        if "benchmark" in df:
            metrics["~~~~~~~~~~~~"] = blank
            if isinstance(returns, pd.Series):
                greeks = core.greeks(
                    df["returns"], df["benchmark"], win_year, is_prepare_returns=False
                )
                metrics["Beta"] = [str(round(greeks["beta"], 2)), "-"]
                metrics["Alpha"] = [str(round(greeks["alpha"], 2)), "-"]
                metrics["Correlation"] = [
                    str(round(df["benchmark"].corr(df["returns"]) * pct, 2)) + "%",
                    "-",
                ]
                metrics["Treynor Ratio"] = [
                    str(
                        round(
                            core.treynor_ratio(
                                df["returns"], df["benchmark"], win_year, rf
                            )
                            * pct,
                            2,
                        )
                    )
                    + "%",
                    "-",
                ]
            elif isinstance(returns, pd.DataFrame):
                greeks = [
                    core.greeks(
                        df[strategy_col],
                        df["benchmark"],
                        win_year,
                        is_prepare_returns=False,
                    )
                    for strategy_col in df_strategy_columns
                ]
                metrics["Beta"] = [str(round(g["beta"], 2)) for g in greeks] + ["-"]
                metrics["Alpha"] = [str(round(g["alpha"], 2)) for g in greeks] + ["-"]
                metrics["Correlation"] = (
                    [
                        str(round(df["benchmark"].corr(df[strategy_col]) * pct, 2))
                        + "%"
                        for strategy_col in df_strategy_columns
                    ]
                ) + ["-"]
                metrics["Treynor Ratio"] = (
                    [
                        str(
                            round(
                                core.treynor_ratio(
                                    df[strategy_col], df["benchmark"], win_year, rf
                                )
                                * pct,
                                2,
                            )
                        )
                        + "%"
                        for strategy_col in df_strategy_columns
                    ]
                ) + ["-"]

    # prepare for display
    for col in metrics.columns:
        try:
            metrics[col] = metrics[col].astype(float).round(2)
            if display or "internal" in kwargs:
                metrics[col] = metrics[col].astype(str)
        except Exception:
            pass
        if (display or "internal" in kwargs) and "*int" in col:
            metrics[col] = metrics[col].str.replace(".0", "", regex=False)
            metrics.rename({col: col.replace("*int", "")}, axis=1, inplace=True)
        if (display or "internal" in kwargs) and "%" in col:
            metrics[col] = metrics[col] + "%"

    try:
        metrics["Longest DD Days"] = pd.to_numeric(metrics["Longest DD Days"]).astype("int")
        metrics["Avg. Drawdown Days"] = pd.to_numeric(metrics["Avg. Drawdown Days"]).astype("int")

        if display or "internal" in kwargs:
            metrics["Longest DD Days"] = metrics["Longest DD Days"].astype(str)
            metrics["Avg. Drawdown Days"] = metrics["Avg. Drawdown Days"].astype(str)
    except Exception:
        metrics["Longest DD Days"] = "-"
        metrics["Avg. Drawdown Days"] = "-"
        if display or "internal" in kwargs:
            metrics["Longest DD Days"] = "-"
            metrics["Avg. Drawdown Days"] = "-"

    metrics.columns = [col if "~" not in col else "" for col in metrics.columns]
    metrics.columns = [col[:-1] if "%" in col else col for col in metrics.columns]
    metrics = metrics.T

    if "benchmark" in df:
        column_names = [strategy_colname, benchmark_colname]
        if isinstance(strategy_colname, list):
            metrics.columns = list(pd.core.common.flatten(column_names))
        else:
            metrics.columns = column_names
    else:
        if isinstance(strategy_colname, list):
            metrics.columns = strategy_colname
        else:
            metrics.columns = [strategy_colname]

    # cleanups
    metrics.replace([-0, "-0"], 0, inplace=True)
    metrics.replace(
        [
            np.nan,
            -np.nan,
            np.inf,
            -np.inf,
            "-nan%",
            "nan%",
            "-nan",
            "nan",
            "-inf%",
            "inf%",
            "-inf",
            "inf",
        ],
        "-",
        inplace=True,
    )

    # move benchmark to be the first column always if present
    if "benchmark" in df:
        metrics = metrics[
            [benchmark_colname]
            + [col for col in metrics.columns if col != benchmark_colname]
        ]
    if not sep:
        metrics = metrics[metrics.index != ""]
    # remove spaces from column names
    metrics = metrics.T
    metrics.columns = [c.replace(" %", "").replace(" *int", "").strip() for c in metrics.columns]
    metrics = metrics.T
    return metrics
