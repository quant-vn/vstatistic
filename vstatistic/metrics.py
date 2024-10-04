import os
import re
import math
import numpy as np
import pandas as pd
from datetime import datetime
from tabulate import tabulate
from packaging.version import Version
from dateutil.relativedelta import relativedelta

from . import core, utils, plot

PANDAS_VERSION_2 = Version(pd.__version__) >= Version("2")
PANDAS_VERSION_2_2 = Version(pd.__version__) >= Version("2.2")

if PANDAS_VERSION_2_2:
    MONTH_END = "ME"
    QUARTER_END = "QE"
    YEAR_END = "YE"
else:
    MONTH_END = "M"
    QUARTER_END = "Q"
    YEAR_END = "A"


class Metrics:
    def __init__(self) -> None:
        self.__returns: pd.Series = None
        self.__benchmark: dict = {}
        self.__orderbook: pd.DataFrame = None
        self.__transaction: pd.DataFrame = None
        self.__trigger: pd.DataFrame = None

    @property
    def returns(self) -> pd.Series:
        return self.__returns

    @returns.setter
    def returns(self, value: pd.Series):
        value = value.dropna()
        value.index = value.index.tz_localize(None)
        self.__returns = value

    def add_benchmark(self, benchmark: pd.Series, name: str):
        self.__benchmark[name] = benchmark

    @property
    def orderbook(self) -> pd.DataFrame:
        return self.__orderbook

    @orderbook.setter
    def orderbook(self, value: pd.DataFrame):
        self.__orderbook = value

    @property
    def transaction(self) -> pd.DataFrame:
        return self.__transaction

    @transaction.setter
    def transaction(self, value: pd.DataFrame):
        self.__transaction = value

    @property
    def trigger(self) -> pd.DataFrame:
        return self.__trigger

    @trigger.setter
    def trigger(self, value: pd.DataFrame):
        self.__trigger = value

    def calc_dd(self, df, display=True, as_pct=False):
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
        self,
        rf=0.0,
        compounded=True,
        periods_per_year=252
    ):
        returns = self.__returns
        win_year, _ = utils.get_trading_periods(periods_per_year)
        benchmark_colname = "Benchmark"
        strategy_colname = "Returns"
        if isinstance(returns, pd.DataFrame):
            if len(returns.columns) > 1:
                blank = [""] * len(returns.columns)
                if isinstance(strategy_colname, str):
                    strategy_colname = list(returns.columns)
        else:
            blank = [""]
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
        if self.__benchmark:
            _benchmarks = list(self.__benchmark.keys())
            benchmark_colname = _benchmarks[0]
            benchmark = self.__benchmark.get(benchmark_colname)
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
        pct = 1

        # return df
        dd = self.calc_dd(df, display=False)

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
        metrics["smart_sharpe"] = core.smart_sharpe(df, rf, win_year, True)

        metrics["sortino"] = core.sortino(df, rf, win_year, True)
        metrics["smart_sortino"] = core.smart_sortino(df, rf, win_year, True)
        metrics["sortino_sqrt2"] = metrics["sortino"] / math.sqrt(2)
        metrics["smart_sortino_sqrt2"] = metrics["smart_sortino"] / math.sqrt(2)
        metrics["omega"] = core.omega(df, rf, 0.0, win_year)

        metrics["~~~~~~~~"] = blank
        metrics["max_drawdown %%"] = blank
        metrics["longest_dd_days"] = blank

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
            except Exception:
                pass

        try:
            metrics["longest_dd_days"] = pd.to_numeric(metrics["longest_dd_days"]).astype("int")
            metrics["avg_drawdown_days"] = pd.to_numeric(metrics["avg_drawdown_days"]).astype("int")
        except Exception:
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
        metrics = metrics[metrics.index != ""]
        # remove spaces from column names
        metrics = metrics.T
        metrics.columns = [
            c.replace(" %", "").replace(" *int", "").strip() for c in metrics.columns
        ]
        metrics = metrics.T
        return metrics

    def html_table(self, obj, showindex="default"):
        obj = tabulate(
            obj, headers="keys", tablefmt="html", floatfmt=".2f", showindex=showindex
        )
        obj = obj.replace(' style="text-align: right;"', "")
        obj = obj.replace(' style="text-align: left;"', "")
        obj = obj.replace(' style="text-align: center;"', "")
        obj = re.sub("<td> +", "<td>", obj)
        obj = re.sub(" +</td>", "</td>", obj)
        obj = re.sub("<th> +", "<th>", obj)
        obj = re.sub(" +</th>", "</th>", obj)
        return obj

    def report(self, output: str = "output.html", type: str = "static"):
        win_year, win_half_year = utils.get_trading_periods(252)
        tpl = ""
        with open(os.path.dirname(os.path.realpath(__file__)) + "/report.html") as f:
            tpl = f.read()
            f.close()

        date_range = self.__returns.index.strftime("%e %b, %Y")
        tpl = tpl.replace("{{date_range}}", date_range[0] + " - " + date_range[-1])
        tpl = tpl.replace("{{title}}", "Statistic")
        _m = self.metrics()
        print(_m)
        _m.index.name = "Metric"
        tpl = tpl.replace("{{metrics}}", self.html_table(_m))

        if isinstance(self.__returns, pd.DataFrame):
            num_cols = len(self.__returns.columns)
            for i in reversed(range(num_cols + 1, num_cols + 3)):
                str_td = "<td></td>" * i
                tpl = tpl.replace(
                    f"<tr>{str_td}</tr>", '<tr><td colspan="{}"><hr></td></tr>'.format(i)
                )

        tpl = tpl.replace(
            "<tr><td></td><td></td><td></td></tr>", '<tr><td colspan="3"><hr></td></tr>'
        )
        tpl = tpl.replace(
            "<tr><td></td><td></td></tr>", '<tr><td colspan="2"><hr></td></tr>'
        )

        dd = core.to_drawdown_series(self.__returns)
        dd_info = core.drawdown_details(dd).sort_values(
            by="max drawdown", ascending=True
        )[:10]
        dd_info = dd_info[["start", "end", "max drawdown", "days"]]
        dd_info.columns = ["Started", "Recovered", "Drawdown", "Days"]
        tpl = tpl.replace("{{dd_info}}", self.html_table(dd_info, False))
        # orderbook
        tpl = tpl.replace("{{orderbook}}", self.html_table(self.__orderbook, False))
        # transaction
        tpl = tpl.replace("{{transaction}}", self.html_table(self.__transaction, False))
        # trigger
        tpl = tpl.replace("{{trigger}}", self.html_table(self.__trigger, False))
        # plots
        figfile = utils.file_stream()
        plot.returns(
            self.__returns,
            self.__benchmark.copy(),
            figsize=(8, 5),
            subtitle=False,
            savefig={"fname": figfile, "format": "svg"},
            show=False,
            ylabel=False,
            cumulative=True,
            is_prepare_returns=False
        )
        tpl = tpl.replace("{{returns}}", utils.embed_figure(figfile, "svg"))
        # log return
        figfile = utils.file_stream()
        plot.log_returns(
            self.__returns,
            self.__benchmark.copy(),
            figsize=(8, 4),
            subtitle=False,
            savefig={"fname": figfile, "format": "svg"},
            show=False,
            ylabel=False,
            cumulative=True,
            is_prepare_returns=False,
        )
        tpl = tpl.replace("{{log_returns}}", utils.embed_figure(figfile, "svg"))
        # Vol return
        figfile = utils.file_stream()
        plot.returns(
            self.__returns,
            self.__benchmark.copy(),
            match_volatility=True,
            figsize=(8, 4),
            subtitle=False,
            savefig={"fname": figfile, "format": "svg"},
            show=False,
            ylabel=False,
            cumulative=True,
            is_prepare_returns=False,
        )
        tpl = tpl.replace("{{vol_returns}}", utils.embed_figure(figfile, "svg"))
        # yearly return
        figfile = utils.file_stream()
        plot.yearly_returns(
            self.__returns,
            self.__benchmark.copy(),
            figsize=(8, 4),
            subtitle=False,
            savefig={"fname": figfile, "format": "svg"},
            show=False,
            ylabel=False,
            compounded=True,
            is_prepare_returns=False
        )
        tpl = tpl.replace("{{eoy_returns}}", utils.embed_figure(figfile, "svg"))
        # # histogram
        # # figfile = utils.file_stream()
        # # plot.histogram(
        # #     self.__returns,
        # #     self.__benchmark.get("SPY"),
        # #     figsize=(7, 4),
        # #     subtitle=False,
        # #     savefig={"fname": figfile, "format": "svg"},
        # #     show=False,
        # #     ylabel=False,
        # #     compounded=True,
        # #     is_prepare_returns=False,
        # # )
        # # tpl = tpl.replace("{{monthly_dist}}", utils.embed_figure(figfile, "svg"))
        # # daily return
        # figfile = utils.file_stream()
        # plot.daily_returns(
        #     self.__returns,
        #     self.__benchmark.copy(),
        #     figsize=(8, 3),
        #     subtitle=False,
        #     savefig={"fname": figfile, "format": "svg"},
        #     show=False,
        #     ylabel=False,
        #     is_prepare_returns=False,
        #     active=False
        # )
        # tpl = tpl.replace("{{daily_returns}}", utils.embed_figure(figfile, "svg"))
        # # rolling beta
        # figfile = utils.file_stream()
        # plot.rolling_beta(
        #     self.__returns,
        #     self.__benchmark.get("SPY"),
        #     figsize=(8, 3),
        #     subtitle=False,
        #     window1=win_half_year,
        #     window2=win_year,
        #     savefig={"fname": figfile, "format": "svg"},
        #     show=False,
        #     ylabel=False,
        #     is_prepare_returns=False,
        # )
        # tpl = tpl.replace("{{rolling_beta}}", utils.embed_figure(figfile, "svg"))
        # # rolling vol
        # figfile = utils.file_stream()
        # plot.rolling_volatility(
        #     self.__returns,
        #     self.__benchmark.get("SPY"),
        #     figsize=(8, 3),
        #     subtitle=False,
        #     savefig={"fname": figfile, "format": "svg"},
        #     show=False,
        #     ylabel=False,
        #     period=win_half_year,
        #     periods_per_year=win_year,
        # )
        # tpl = tpl.replace("{{rolling_vol}}", utils.embed_figure(figfile, "svg"))
        # rolling sharpe
        figfile = utils.file_stream()
        plot.rolling_sharpe(
            self.__returns,
            figsize=(8, 3),
            subtitle=False,
            savefig={"fname": figfile, "format": "svg"},
            show=False,
            ylabel=False,
            period=win_half_year,
            periods_per_year=win_year,
        )
        tpl = tpl.replace("{{rolling_sharpe}}", utils.embed_figure(figfile, "svg"))
        # sortino
        figfile = utils.file_stream()
        plot.rolling_sortino(
            self.__returns,
            figsize=(8, 3),
            subtitle=False,
            savefig={"fname": figfile, "format": "svg"},
            show=False,
            ylabel=False,
            period=win_half_year,
            periods_per_year=win_year,
        )
        tpl = tpl.replace("{{rolling_sortino}}", utils.embed_figure(figfile, "svg"))
        # drawdown period
        figfile = utils.file_stream()
        plot.drawdowns_periods(
            self.__returns,
            figsize=(8, 4),
            subtitle=False,
            title=self.__returns.name,
            savefig={"fname": figfile, "format": "svg"},
            show=False,
            ylabel=False,
            compounded=True,
            is_prepare_returns=False,
        )
        tpl = tpl.replace("{{dd_periods}}", utils.embed_figure(figfile, "svg"))

        figfile = utils.file_stream()
        plot.drawdown(
            self.__returns,
            figsize=(8, 3),
            subtitle=False,
            savefig={"fname": figfile, "format": "svg"},
            show=False,
            ylabel=False,
        )
        tpl = tpl.replace("{{dd_plot}}", utils.embed_figure(figfile, "svg"))

        # figfile = utils.file_stream()
        # plot.monthly_heatmap(
        #     self.__returns,
        #     self.__benchmark.get("SPY"),
        #     figsize=(8, 4),
        #     cbar=False,
        #     returns_label=self.__returns.name,
        #     savefig={"fname": figfile, "format": "svg"},
        #     show=False,
        #     ylabel=False,
        #     compounded=True,
        #     active=False,
        # )
        # tpl = tpl.replace("{{monthly_heatmap}}", utils.embed_figure(figfile, "svg"))

        figfile = utils.file_stream()
        plot.distribution(
            self.__returns,
            figsize=(8, 4),
            subtitle=False,
            title=self.__returns.name,
            savefig={"fname": figfile, "format": "svg"},
            show=False,
            ylabel=False,
            compounded=True,
            is_prepare_returns=False,
        )
        tpl = tpl.replace("{{returns_dist}}", utils.embed_figure(figfile, "svg"))

        # END
        tpl = re.sub(r"\{\{(.*?)\}\}", "", tpl)
        tpl = tpl.replace("white-space:pre;", "")
        with open(output, "w", encoding="utf-8") as f:
            f.write(tpl)
