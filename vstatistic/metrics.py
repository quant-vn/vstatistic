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
                "max_drawdown %": ret_dd[col]
                .sort_values(by="max drawdown", ascending=True)["max drawdown"]
                .values[0]
                / 100,
                "longest_dd_days": str(
                    np.round(
                        ret_dd[col]
                        .sort_values(by="days", ascending=False)["days"]
                        .values[0]
                    )
                ),
                "avg_drawdown %": ret_dd[col]["max drawdown"].mean() / 100,
                "avg_drawdown_days": str(np.round(ret_dd[col]["days"].mean())),
            }
            for col in ret_dd.columns.get_level_values(0)
        }
    else:
        dd_stats = {
            "returns": {
                "max_drawdown %": ret_dd.sort_values(by="max drawdown", ascending=True)[
                    "max drawdown"
                ].values[0]
                / 100,
                "longest_dd_days": str(
                    np.round(
                        ret_dd.sort_values(by="days", ascending=False)["days"].values[0]
                    )
                ),
                "avg_drawdown %": ret_dd["max drawdown"].mean() / 100,
                "avg_drawdown_days": str(np.round(ret_dd["days"].mean())),
            }
        }
    if "benchmark" in df and (dd_info.columns, pd.MultiIndex):
        bench_dd = dd_info["benchmark"].sort_values(by="max drawdown")
        dd_stats["benchmark"] = {
            "max_drawdown %": bench_dd.sort_values(by="max drawdown", ascending=True)[
                "max drawdown"
            ].values[0]
            / 100,
            "longest_dd_days": str(
                np.round(
                    bench_dd.sort_values(by="days", ascending=False)["days"].values[0]
                )
            ),
            "avg_drawdown %": bench_dd["max drawdown"].mean() / 100,
            "avg_drawdown_days": str(np.round(bench_dd["days"].mean())),
        }

    # pct multiplier
    pct = 100 if display or as_pct else 1

    dd_stats = pd.DataFrame(dd_stats).T
    dd_stats["max_drawdown %"] = dd_stats["max_drawdown %"].astype(float) * pct
    dd_stats["avg_drawdown %"] = dd_stats["avg_drawdown %"].astype(float) * pct

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
    metrics["start_period"] = pd.Series(s_start)
    metrics["end_period"] = pd.Series(s_end)
    metrics["risk_free_rate %%"] = pd.Series(s_rf) * 100
    metrics["time_in_market %%"] = core.exposure(df, is_prepare_returns=False) * pct

    metrics["~"] = blank

    if compounded:
        metrics["cumulative_return %%"] = (core.comp(df) * pct).map("{:,.2f}".format)
    else:
        metrics["total_return %%"] = (df.sum() * pct).map("{:,.2f}".format)

    metrics["cagr %%"] = core.cagr(df, rf, compounded) * pct

    metrics["~~~~~~~~~~~~~~"] = blank

    metrics["sharpe"] = core.sharpe(df, rf, win_year, True)
    metrics["prob_sharpe_ratio %%"] = (
        core.probabilistic_sharpe_ratio(df, rf, win_year, False) * pct
    )
    if mode.lower() == "full":
        metrics["smart_sharpe"] = core.smart_sharpe(df, rf, win_year, True)

    metrics["sortino"] = core.sortino(df, rf, win_year, True)
    if mode.lower() == "full":
        metrics["smart_sortino"] = core.smart_sortino(df, rf, win_year, True)
    metrics["sortino_sqrt2"] = metrics["sortino"] / math.sqrt(2)
    if mode.lower() == "full":
        metrics["smart_sortino_sqrt2"] = metrics["smart_sortino"] / math.sqrt(2)
    metrics["omega"] = core.omega(df, rf, 0.0, win_year)

    metrics["~~~~~~~~"] = blank
    metrics["max_drawdown %%"] = blank
    metrics["longest_dd_days"] = blank

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
                metrics["volatility %%"] = list(pd.core.common.flatten(vol_))
            else:
                metrics["volatility %%"] = vol_

            if isinstance(returns, pd.Series):
                metrics["r2"] = core.r_squared(
                    df["returns"], df["benchmark"], is_prepare_returns=False
                )
                metrics["information_ratio"] = core.information_ratio(
                    df["returns"], df["benchmark"], is_prepare_returns=False
                )
            elif isinstance(returns, pd.DataFrame):
                metrics["r2"] = (
                    [
                        core.r_squared(
                            df[strategy_col], df["benchmark"], is_prepare_returns=False
                        ).round(2)
                        for strategy_col in df_strategy_columns
                    ]
                ) + ["-"]
                metrics["information_ratio"] = (
                    [
                        core.information_ratio(
                            df[strategy_col], df["benchmark"], is_prepare_returns=False
                        ).round(2)
                        for strategy_col in df_strategy_columns
                    ]
                ) + ["-"]
        else:
            if isinstance(returns, pd.Series):
                metrics["volatility %%"] = [ret_vol]
            elif isinstance(returns, pd.DataFrame):
                metrics["volatility %%"] = ret_vol

        metrics["calmar"] = core.calmar(df, is_prepare_returns=False)
        metrics["skew"] = core.skew(df, is_prepare_returns=False)
        metrics["kurtosis"] = core.kurtosis(df, is_prepare_returns=False)

        metrics["~~~~~~~~~~"] = blank

        metrics["expected_daily %%"] = (
            core.expected_return(df, compounded=compounded, is_prepare_returns=False) * pct
        )
        metrics["expected_monthly %%"] = (
            core.expected_return(
                df, compounded=compounded, aggregate="M", is_prepare_returns=False
            ) * pct
        )
        metrics["expected_yearly %%"] = (
            core.expected_return(
                df, compounded=compounded, aggregate="A", is_prepare_returns=False
            ) * pct
        )
        metrics["kelly_criterion %%"] = (core.kelly_criterion(df, is_prepare_returns=False) * pct)
        metrics["risk_of_ruin %%"] = core.risk_of_ruin(df, is_prepare_returns=False)
        metrics["daily_value_at_risk %%"] = -abs(core.var(df, is_prepare_returns=False) * pct)
        metrics["expected_shortfall %%"] = -abs(core.cvar(df, is_prepare_returns=False) * pct)

    metrics["~~~~~~"] = blank

    if mode.lower() == "full":
        metrics["max_consecutive_wins *int"] = core.consecutive_wins(df)
        metrics["max_consecutive_losses *int"] = core.consecutive_losses(df)

    metrics["gain_pain_ratio"] = core.gain_to_pain_ratio(df, rf)
    metrics["gain_pain_1m"] = core.gain_to_pain_ratio(df, rf, MONTH_END)
    # if mode.lower() == 'full':
    #     metrics['GPR (3M)'] = _stats.gain_to_pain_ratio(df, rf, "Q")
    #     metrics['GPR (6M)'] = _stats.gain_to_pain_ratio(df, rf, "2Q")
    #     metrics['GPR (1Y)'] = _stats.gain_to_pain_ratio(df, rf, "A")
    metrics["~~~~~~~"] = blank

    metrics["payoff_ratio"] = core.payoff_ratio(df, is_prepare_returns=False)
    metrics["profit_factor"] = core.profit_factor(df, is_prepare_returns=False)
    metrics["common_sense_ratio"] = core.common_sense_ratio(df, is_prepare_returns=False)
    metrics["cpc_index"] = core.cpc_index(df, is_prepare_returns=False)
    metrics["tail_ratio"] = core.tail_ratio(df, is_prepare_returns=False)
    metrics["outlier_win_ratio"] = core.outlier_win_ratio(df, is_prepare_returns=False)
    metrics["outlier_loss_ratio"] = core.outlier_loss_ratio(df, is_prepare_returns=False)

    # returns
    metrics["~~"] = blank
    comp_func = core.comp if compounded else np.sum

    today = df.index[-1]
    metrics["mtd %%"] = comp_func(df[df.index >= datetime(today.year, today.month, 1)]) * pct

    d = today - relativedelta(months=3)
    metrics["3m %%"] = comp_func(df[df.index >= d]) * pct
    d = today - relativedelta(months=6)
    metrics["6m %%"] = comp_func(df[df.index >= d]) * pct
    metrics["ytd %%"] = comp_func(df[df.index >= datetime(today.year, 1, 1)]) * pct
    d = today - relativedelta(years=1)
    metrics["1y %%"] = comp_func(df[df.index >= d]) * pct
    d = today - relativedelta(months=35)
    metrics["3y %%"] = core.cagr(df[df.index >= d], 0.0, compounded) * pct
    d = today - relativedelta(months=59)
    metrics["5y %%"] = core.cagr(df[df.index >= d], 0.0, compounded) * pct
    d = today - relativedelta(years=10)
    metrics["10y %%"] = core.cagr(df[df.index >= d], 0.0, compounded) * pct
    metrics["all_time %%"] = core.cagr(df, 0.0, compounded) * pct

    # best/worst
    if mode.lower() == "full":
        metrics["~~~"] = blank
        metrics["best_day %%"] = core.best(
            df, compounded=compounded, is_prepare_returns=False
        ) * pct
        metrics["worst_day %%"] = core.worst(df, is_prepare_returns=False) * pct
        metrics["best_month %%"] = (
            core.best(df, compounded=compounded, aggregate="M", is_prepare_returns=False) * pct
        )
        metrics["worst_month %%"] = (
            core.worst(df, aggregate="M", is_prepare_returns=False) * pct
        )
        metrics["best_year %%"] = (
            core.best(df, compounded=compounded, aggregate="A", is_prepare_returns=False) * pct
        )
        metrics["worst_year %%"] = (
            core.worst(df, compounded=compounded, aggregate="A", is_prepare_returns=False) * pct
        )

    # dd
    metrics["~~~~"] = blank
    for ix, row in dd.iterrows():
        metrics[ix] = row
    metrics["recovery_factor"] = core.recovery_factor(df)
    metrics["ulcer_index"] = core.ulcer_index(df)
    metrics["serenity_index"] = core.serenity_index(df, rf)

    # win rate
    if mode.lower() == "full":
        metrics["~~~~~"] = blank
        metrics["avg_up_month %%"] = (
            core.avg_win(df, compounded=compounded, aggregate="M", is_prepare_returns=False) * pct
        )
        metrics["avg_down_month %%"] = (
            core.avg_loss(df, compounded=compounded, aggregate="M", is_prepare_returns=False) * pct
        )
        metrics["win_days"] = core.win_rate(df, is_prepare_returns=False) * pct
        metrics["win_month"] = (
            core.win_rate(df, compounded=compounded, aggregate="M", is_prepare_returns=False) * pct
        )
        metrics["win_quarter %%"] = (
            core.win_rate(df, compounded=compounded, aggregate="Q", is_prepare_returns=False) * pct
        )
        metrics["win_year %%"] = (
            core.win_rate(df, compounded=compounded, aggregate="A", is_prepare_returns=False) * pct
        )

        if "benchmark" in df:
            metrics["~~~~~~~~~~~~"] = blank
            if isinstance(returns, pd.Series):
                greeks = core.greeks(
                    df["returns"], df["benchmark"], win_year, is_prepare_returns=False
                )
                metrics["beta"] = [str(round(greeks["beta"], 2)), "-"]
                metrics["alpha"] = [str(round(greeks["alpha"], 2)), "-"]
                metrics["correlation"] = [
                    str(round(df["benchmark"].corr(df["returns"]) * pct, 2)) + "%",
                    "-",
                ]
                metrics["treynor_ratio"] = [
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
                metrics["beta"] = [str(round(g["beta"], 2)) for g in greeks] + ["-"]
                metrics["alpha"] = [str(round(g["alpha"], 2)) for g in greeks] + ["-"]
                metrics["correlation"] = (
                    [
                        str(round(df["benchmark"].corr(df[strategy_col]) * pct, 2))
                        + "%"
                        for strategy_col in df_strategy_columns
                    ]
                ) + ["-"]
                metrics["treynor_ratio"] = (
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
        metrics["longest_dd_days"] = pd.to_numeric(metrics["longest_dd_days"]).astype("int")
        metrics["avg_drawdown_days"] = pd.to_numeric(metrics["avg_drawdown_days"]).astype("int")

        if display or "internal" in kwargs:
            metrics["longest_dd_days"] = metrics["longest_dd_days"].astype(str)
            metrics["avg_drawdown_days"] = metrics["avg_drawdown_days"].astype(str)
    except Exception:
        metrics["longest_dd_days"] = "-"
        metrics["avg_drawdown_days"] = "-"
        if display or "internal" in kwargs:
            metrics["longest_dd_days"] = "-"
            metrics["avg_drawdown_days"] = "-"

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
