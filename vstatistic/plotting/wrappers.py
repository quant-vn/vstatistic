import warnings
import matplotlib.pyplot as _plt
from matplotlib.ticker import (
    StrMethodFormatter as _StrMethodFormatter,
    FuncFormatter as _FuncFormatter,
)

import numpy as _np
from pandas import DataFrame as _df
import pandas as _pd
import seaborn as _sns
from packaging.version import Version

from .. import core, utils

from . import core as _core

PANDAS_VERSION_2 = Version(_pd.__version__) >= Version("2")
PANDAS_VERSION_2_2 = Version(_pd.__version__) >= Version("2.2")

if PANDAS_VERSION_2_2:
    MONTH_END = "ME"
    QUARTER_END = "QE"
    YEAR_END = "YE"
else:
    MONTH_END = "M"
    QUARTER_END = "Q"
    YEAR_END = "A"


_FLATUI_COLORS = ["#ec692d", "#4ca456", "#d9302c", "#4a154b", "#eaa23f", "#004883"]
_GRAYSCALE_COLORS = (len(_FLATUI_COLORS) * ["black"]) + ["white"]

_HAS_PLOTLY = False
try:
    import plotly

    _HAS_PLOTLY = True
except ImportError:
    pass


def to_plotly(fig):
    if not _HAS_PLOTLY:
        return fig
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = plotly.tools.mpl_to_plotly(fig)
        return plotly.plotly.iplot(fig, filename="vstatistic-plot", overwrite=True)


def snapshot(
    returns,
    grayscale=False,
    figsize=(10, 8),
    title="Portfolio Summary",
    fontname="Arial",
    lw=1.5,
    mode="comp",
    subtitle=True,
    savefig=None,
    show=True,
    log_scale=False,
    **kwargs,
):

    strategy_colname = kwargs.get("strategy_col", "Strategy")

    multi_column = False
    if isinstance(returns, _pd.Series):
        returns.name = strategy_colname
    elif isinstance(returns, _pd.DataFrame):
        if len(returns.columns) > 1:
            if strategy_colname in returns.columns:
                returns = returns[strategy_colname]
            else:
                multi_column = True
                returns = returns.mean(axis=1)
                title = title + " (daily equal-weighted*)"
        returns.columns = strategy_colname

    colors = _GRAYSCALE_COLORS if grayscale else _FLATUI_COLORS
    returns = core.make_portfolio(returns.dropna(), 1, mode).pct_change().fillna(0)

    if figsize is None:
        size = list(_plt.gcf().get_size_inches())
        figsize = (size[0], size[0] * 0.75)

    fig, axes = _plt.subplots(
        3, 1, sharex=True, figsize=figsize, gridspec_kw={"height_ratios": [3, 1, 1]}
    )

    if multi_column:
        _plt.figtext(
            0,
            -0.05,
            "            * When a multi-column DataFrame is passed, the mean of all columns will be used as returns.\n"  # noqa
            "              To change this behavior, use a pandas Series or pass the column name in the `strategy_col` parameter.", # noqa
            ha="left",
            fontsize=11,
            color="black",
            alpha=0.6,
            linespacing=1.5,
        )

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    fig.suptitle(
        title, fontsize=14, y=0.97, fontname=fontname, fontweight="bold", color="black"
    )

    fig.set_facecolor("white")

    if subtitle:
        if isinstance(returns, _pd.Series):
            axes[0].set_title(
                "%s - %s ;  Sharpe: %.2f                      \n"
                % (
                    returns.index.date[:1][0].strftime("%e %b '%y"),
                    returns.index.date[-1:][0].strftime("%e %b '%y"),
                    core.sharpe(returns),
                ),
                fontsize=12,
                color="gray",
            )
        elif isinstance(returns, _pd.DataFrame):
            axes[0].set_title(
                "\n%s - %s ;  "
                % (
                    returns.index.date[:1][0].strftime("%e %b '%y"),
                    returns.index.date[-1:][0].strftime("%e %b '%y"),
                ),
                fontsize=12,
                color="gray",
            )

    axes[0].set_ylabel("Cumulative Return", fontname=fontname, fontweight="bold", fontsize=12)
    if isinstance(returns, _pd.Series):
        axes[0].plot(
            core.compsum(returns) * 100,
            color=colors[1],
            lw=1 if grayscale else lw,
            zorder=1,
        )
    elif isinstance(returns, _pd.DataFrame):
        for col in returns.columns:
            axes[0].plot(
                core.compsum(returns[col]) * 100,
                label=col,
                lw=1 if grayscale else lw,
                zorder=1,
            )
    axes[0].axhline(0, color="silver", lw=1, zorder=0)

    axes[0].set_yscale("symlog" if log_scale else "linear")
    # axes[0].legend(fontsize=12)

    dd = core.to_drawdown_series(returns) * 100
    ddmin = utils.round_to_closest(abs(dd.min()), 5)
    ddmin_ticks = 5
    if ddmin > 50:
        ddmin_ticks = ddmin / 4
    elif ddmin > 20:
        ddmin_ticks = ddmin / 3
    ddmin_ticks = int(utils._round_to_closest(ddmin_ticks, 5))

    axes[1].set_ylabel("Drawdown", fontname=fontname, fontweight="bold", fontsize=12)
    axes[1].set_yticks(_np.arange(-ddmin, 0, step=ddmin_ticks))
    if isinstance(dd, _pd.Series):
        axes[1].plot(dd, color=colors[2], lw=1 if grayscale else lw, zorder=1)
    elif isinstance(dd, _pd.DataFrame):
        for col in dd.columns:
            axes[1].plot(dd[col], label=col, lw=1 if grayscale else lw, zorder=1)
    axes[1].axhline(0, color="silver", lw=1, zorder=0)
    if not grayscale:
        if isinstance(dd, _pd.Series):
            axes[1].fill_between(dd.index, 0, dd, color=colors[2], alpha=0.25)
        elif isinstance(dd, _pd.DataFrame):
            for i, col in enumerate(dd.columns):
                axes[1].fill_between(
                    dd[col].index, 0, dd[col], color=colors[i + 1], alpha=0.25
                )

    axes[1].set_yscale("symlog" if log_scale else "linear")

    axes[2].set_ylabel(
        "Daily Return", fontname=fontname, fontweight="bold", fontsize=12
    )
    if isinstance(returns, _pd.Series):
        axes[2].plot(
            returns * 100, color=colors[0], label=returns.name, lw=0.5, zorder=1
        )
    elif isinstance(returns, _pd.DataFrame):
        for i, col in enumerate(returns.columns):
            axes[2].plot(
                returns[col] * 100, color=colors[i], label=col, lw=0.5, zorder=1
            )
    axes[2].axhline(0, color="silver", lw=1, zorder=0)
    axes[2].axhline(0, color=colors[-1], linestyle="--", lw=1, zorder=2)

    axes[2].set_yscale("symlog" if log_scale else "linear")
    # axes[2].legend(fontsize=12)

    retmax = utils.round_to_closest(returns.max() * 100, 5)
    retmin = utils.round_to_closest(returns.min() * 100, 5)
    retdiff = retmax - retmin
    steps = 5
    if retdiff > 50:
        steps = retdiff / 5
    elif retdiff > 30:
        steps = retdiff / 4
    steps = utils.round_to_closest(steps, 5)
    axes[2].set_yticks(_np.arange(retmin, retmax, step=steps))

    for ax in axes:
        ax.set_facecolor("white")
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.yaxis.set_major_formatter(_StrMethodFormatter("{x:,.0f}%"))

    _plt.subplots_adjust(hspace=0, bottom=0, top=1)
    fig.autofmt_xdate()

    try:
        _plt.subplots_adjust(hspace=0)
    except Exception:
        pass
    try:
        fig.tight_layout(w_pad=0, h_pad=0)
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)

    if show:
        _plt.show(block=False)

    _plt.close()

    if not show:
        return fig

    return None


def earnings(
    returns,
    start_balance=1e5,
    mode="comp",
    grayscale=False,
    figsize=(10, 6),
    title="Portfolio Earnings",
    fontname="Arial",
    lw=1.5,
    subtitle=True,
    savefig=None,
    show=True,
):

    colors = _GRAYSCALE_COLORS if grayscale else _FLATUI_COLORS
    alpha = 0.5 if grayscale else 0.8

    returns = core.make_portfolio(returns, start_balance, mode)

    if figsize is None:
        size = list(_plt.gcf().get_size_inches())
        figsize = (size[0], size[0] * 0.55)

    fig, ax = _plt.subplots(figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    fig.suptitle(
        title, fontsize=14, y=0.995, fontname=fontname, fontweight="bold", color="black"
    )

    if subtitle:
        ax.set_title(
            "\n%s - %s ;  P&L: %s (%s)                "
            % (
                returns.index.date[1:2][0].strftime("%e %b '%y"),
                returns.index.date[-1:][0].strftime("%e %b '%y"),
                utils.score_str(
                    "${:,}".format(round(returns.values[-1] - returns.values[0], 2))
                ),
                utils.score_str(
                    "{:,}%".format(
                        round((returns.values[-1] / returns.values[0] - 1) * 100, 2)
                    )
                ),
            ),
            fontsize=12,
            color="gray",
        )

    mx = returns.max()
    returns_max = returns[returns == mx]
    ix = returns_max[~_np.isnan(returns_max)].index[0]
    returns_max = _np.where(returns.index == ix, mx, _np.nan)

    ax.plot(
        returns.index,
        returns_max,
        marker="o",
        lw=0,
        alpha=alpha,
        markersize=12,
        color=colors[0],
    )
    ax.plot(returns.index, returns, color=colors[1], lw=1 if grayscale else lw)

    ax.set_ylabel(
        "Value of  ${:,.0f}".format(start_balance),
        fontname=fontname,
        fontweight="bold",
        fontsize=12,
    )

    ax.yaxis.set_major_formatter(_FuncFormatter(_core.format_cur_axis))
    ax.yaxis.set_label_coords(-0.1, 0.5)

    fig.set_facecolor("white")
    ax.set_facecolor("white")
    fig.autofmt_xdate()

    try:
        _plt.subplots_adjust(hspace=0)
    except Exception:
        pass
    try:
        fig.tight_layout(w_pad=0, h_pad=0)
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)

    if show:
        _plt.show(block=False)

    _plt.close()

    if not show:
        return fig

    return None


def returns(
    returns,
    benchmark=None,
    figsize=(10, 6),
    fontname="Arial",
    lw=1.5,
    match_volatility=False,
    compound=True,
    cumulative=True,
    resample=None,
    ylabel="Cumulative Returns",
    subtitle=True,
    savefig=None,
    show=True,
    is_prepare_returns=True,
):

    title = "Cumulative " if compound else "Returns"
    _benchmarks = list(benchmark.keys())
    if _benchmarks:
        if _benchmarks:
            title += "Returns vs " + ", ".join(_benchmarks)
        if match_volatility:
            title += " (Volatility Matched)"
    for _b in _benchmarks:
        benchmark[_b] = core.prepare_benchmark(benchmark.get(_b), returns.index)
    if is_prepare_returns:
        returns = core.prepare_returns(returns)
    fig = _core.plot_timeseries(
        returns,
        benchmark,
        title,
        ylabel=ylabel,
        match_volatility=match_volatility,
        log_scale=False,
        resample=resample,
        compound=compound,
        cumulative=cumulative,
        lw=lw,
        figsize=figsize,
        fontname=fontname,
        subtitle=subtitle,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def log_returns(
    returns,
    benchmark=None,
    figsize=(10, 5),
    fontname="Arial",
    lw=1.5,
    match_volatility=False,
    compound=True,
    cumulative=True,
    resample=None,
    ylabel="Cumulative Returns",
    subtitle=True,
    savefig=None,
    show=True,
    is_prepare_returns=True,
):

    title = "Cumulative " if compound else "Returns"
    _benchmarks = list(benchmark.keys())
    if _benchmarks:
        title += "Returns vs " + ", ".join(_benchmarks) + " (Log Scaled"
        if match_volatility:
            title += ", Volatility Matched)"
        else:
            title += ")"
    else:
        title += " (Log Scaled)"
    for _b in _benchmarks:
        benchmark[_b] = core.prepare_benchmark(benchmark.get(_b), returns.index)

    if is_prepare_returns:
        returns = core.prepare_returns(returns)
    fig = _core.plot_timeseries(
        returns,
        benchmark,
        title,
        ylabel=ylabel,
        match_volatility=match_volatility,
        log_scale=True,
        resample=resample,
        compound=compound,
        cumulative=cumulative,
        lw=lw,
        figsize=figsize,
        fontname=fontname,
        subtitle=subtitle,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def daily_returns(
    returns,
    benchmark,
    figsize=(10, 4),
    fontname="Arial",
    lw=0.5,
    log_scale=False,
    ylabel="Returns",
    subtitle=True,
    savefig=None,
    show=True,
    is_prepare_returns=True,
    active=False,
):

    if is_prepare_returns:
        returns = core.prepare_returns(returns)
        if active and benchmark is not None:
            benchmark = core.prepare_returns(benchmark)
            returns = returns - benchmark

    plot_title = "Daily Active Returns" if active else "Daily Returns"

    fig = _core.plot_timeseries(
        returns,
        None,
        plot_title,
        ylabel=ylabel,
        match_volatility=False,
        log_scale=log_scale,
        resample="D",
        compound=False,
        lw=lw,
        figsize=figsize,
        fontname=fontname,
        subtitle=subtitle,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def yearly_returns(
    returns,
    benchmark=None,
    fontname="Arial",
    hlw=1.5,
    hlcolor="red",
    hllabel="",
    match_volatility=False,
    log_scale=False,
    figsize=(10, 5),
    ylabel=True,
    subtitle=True,
    compounded=True,
    savefig=None,
    show=True,
    is_prepare_returns=True,
):

    title = "EOY Returns "
    _benchmarks = list(benchmark.keys())
    if _benchmarks:
        title += ", ".join(_benchmarks)
        for _b in _benchmarks:
            benchmark[_b] = (
                core.prepare_benchmark(benchmark.get(_b), returns.index)
                .resample(YEAR_END)
                .apply(core.comp)
                .resample(YEAR_END).last()
            )

    if is_prepare_returns:
        returns = core.prepare_returns(returns)

    if compounded:
        returns = returns.resample(YEAR_END).apply(core.comp)
    else:
        returns = returns.resample(YEAR_END).apply(_df.sum)
    returns = returns.resample(YEAR_END).last()

    fig = _core.plot_returns_bars(
        returns,
        benchmark,
        fontname=fontname,
        hline=returns.mean(),
        hlw=hlw,
        hllabel=hllabel,
        hlcolor=hlcolor,
        match_volatility=match_volatility,
        log_scale=log_scale,
        resample=None,
        title=title,
        figsize=figsize,
        ylabel=ylabel,
        subtitle=subtitle,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def distribution(
    returns,
    fontname="Arial",
    ylabel=True,
    figsize=(10, 6),
    subtitle=True,
    compounded=True,
    savefig=None,
    show=True,
    title=None,
    is_prepare_returns=True,
):
    if is_prepare_returns:
        returns = core.prepare_returns(returns)

    fig = _core.plot_distribution(
        returns,
        fontname=fontname,
        figsize=figsize,
        ylabel=ylabel,
        subtitle=subtitle,
        title=title,
        compounded=compounded,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def histogram(
    returns,
    benchmark=None,
    resample=MONTH_END,
    fontname="Arial",
    figsize=(10, 5),
    ylabel=True,
    subtitle=True,
    compounded=True,
    savefig=None,
    show=True,
    is_prepare_returns=True,
):

    if is_prepare_returns:
        returns = core.prepare_returns(returns)
        if benchmark is not None:
            benchmark = core.prepare_returns(benchmark)

    if resample == "W":
        title = "Weekly "
    elif resample == MONTH_END:
        title = "Monthly "
    elif resample == "Q":
        title = "Quarterly "
    elif resample == YEAR_END:
        title = "Annual "
    else:
        title = ""

    return _core.plot_histogram(
        returns,
        benchmark,
        resample=resample,
        fontname=fontname,
        title="Distribution of %sReturns" % title,
        figsize=figsize,
        ylabel=ylabel,
        subtitle=subtitle,
        compounded=compounded,
        savefig=savefig,
        show=show,
    )


def drawdown(
    returns,
    figsize=(10, 5),
    fontname="Arial",
    lw=1,
    log_scale=False,
    match_volatility=False,
    compound=False,
    ylabel="Drawdown",
    resample=None,
    subtitle=True,
    savefig=None,
    show=True,
):

    dd = core.to_drawdown_series(returns)

    fig = _core.plot_timeseries(
        dd,
        title="Underwater Plot",
        hline=dd.mean(),
        hlw=2,
        hllabel="Average",
        returns_label="Drawdown",
        compound=compound,
        match_volatility=match_volatility,
        log_scale=log_scale,
        resample=resample,
        fill=True,
        lw=lw,
        figsize=figsize,
        ylabel=ylabel,
        fontname=fontname,
        subtitle=subtitle,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def drawdowns_periods(
    returns,
    periods=5,
    lw=1.5,
    log_scale=False,
    fontname="Arial",
    title=None,
    figsize=(10, 5),
    ylabel=True,
    subtitle=True,
    compounded=True,
    savefig=None,
    show=True,
    is_prepare_returns=True,
):
    if is_prepare_returns:
        returns = core.prepare_returns(returns)

    fig = _core.plot_longest_drawdowns(
        returns,
        periods=periods,
        lw=lw,
        log_scale=log_scale,
        fontname=fontname,
        title=title,
        figsize=figsize,
        ylabel=ylabel,
        subtitle=subtitle,
        compounded=compounded,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def rolling_beta(
    returns,
    benchmark,
    window1=126,
    window1_label="6-Months",
    window2=252,
    window2_label="12-Months",
    lw=1.5,
    fontname="Arial",
    figsize=(10, 3),
    ylabel=True,
    subtitle=True,
    savefig=None,
    show=True,
    is_prepare_returns=True,
):

    if is_prepare_returns:
        returns = core.prepare_returns(returns)

    benchmark = core.prepare_benchmark(benchmark, returns.index)

    fig = _core.plot_rolling_beta(
        returns,
        benchmark,
        window1=window1,
        window1_label=window1_label,
        window2=window2,
        window2_label=window2_label,
        title="Rolling Beta to Benchmark",
        fontname=fontname,
        lw=lw,
        figsize=figsize,
        ylabel=ylabel,
        subtitle=subtitle,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def rolling_volatility(
    returns,
    benchmark=None,
    period=126,
    period_label="6-Months",
    periods_per_year=252,
    lw=1.5,
    fontname="Arial",
    figsize=(10, 3),
    ylabel="Volatility",
    subtitle=True,
    savefig=None,
    show=True,
):

    returns = core.rolling_volatility(returns, period, periods_per_year)

    if benchmark is not None:
        benchmark = core.prepare_benchmark(benchmark, returns.index)
        benchmark = core.rolling_volatility(
            benchmark, period, periods_per_year, is_prepare_returns=False
        )

    fig = _core.plot_rolling_stats(
        returns,
        benchmark,
        hline=returns.mean(),
        hlw=1.5,
        ylabel=ylabel,
        title="Rolling Volatility (%s)" % period_label,
        fontname=fontname,
        lw=lw,
        figsize=figsize,
        subtitle=subtitle,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def rolling_sharpe(
    returns,
    benchmark=None,
    rf=0.0,
    period=126,
    period_label="6-Months",
    periods_per_year=252,
    lw=1.25,
    fontname="Arial",
    figsize=(10, 3),
    ylabel="Sharpe",
    subtitle=True,
    savefig=None,
    show=True,
):

    returns = core.rolling_sharpe(
        returns,
        rf,
        period,
        True,
        periods_per_year,
    )

    if benchmark is not None:
        benchmark = core.prepare_benchmark(benchmark, returns.index, rf)
        benchmark = core.rolling_sharpe(
            benchmark, rf, period, True, periods_per_year, is_prepare_returns=False
        )

    fig = _core.plot_rolling_stats(
        returns,
        benchmark,
        hline=returns.mean(),
        hlw=1.5,
        ylabel=ylabel,
        title="Rolling Sharpe (%s)" % period_label,
        fontname=fontname,
        lw=lw,
        figsize=figsize,
        subtitle=subtitle,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def rolling_sortino(
    returns,
    benchmark=None,
    rf=0.0,
    period=126,
    period_label="6-Months",
    periods_per_year=252,
    lw=1.25,
    fontname="Arial",
    figsize=(10, 3),
    ylabel="Sortino",
    subtitle=True,
    savefig=None,
    show=True,
):

    returns = core.rolling_sortino(returns, rf, period, True, periods_per_year)

    if benchmark is not None:
        benchmark = core.prepare_benchmark(benchmark, returns.index, rf)
        benchmark = core.rolling_sortino(
            benchmark, rf, period, True, periods_per_year, is_prepare_returns=False
        )

    fig = _core.plot_rolling_stats(
        returns,
        benchmark,
        hline=returns.mean(),
        hlw=1.5,
        ylabel=ylabel,
        title="Rolling Sortino (%s)" % period_label,
        fontname=fontname,
        lw=lw,
        figsize=figsize,
        subtitle=subtitle,
        savefig=savefig,
        show=show,
    )
    if not show:
        return fig


def monthly_heatmap(
    returns,
    benchmark=None,
    annot_size=10,
    figsize=(10, 5),
    cbar=True,
    square=False,
    returns_label="Strategy",
    compounded=True,
    eoy=False,
    fontname="Arial",
    ylabel=True,
    savefig=None,
    show=True,
    active=False,
):
    # colors, ls, alpha = _core._get_colors(grayscale)
    cmap = "RdYlGn"

    returns = core.monthly_returns(returns, eoy=eoy, compounded=compounded) * 100

    fig_height = len(returns) / 2.5

    if figsize is None:
        size = list(_plt.gcf().get_size_inches())
        figsize = (size[0], size[1])

    figsize = (figsize[0], max([fig_height, figsize[1]]))

    if cbar:
        figsize = (figsize[0] * 1.051, max([fig_height, figsize[1]]))

    fig, ax = _plt.subplots(figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # _sns.set(font_scale=.9)
    if active and benchmark is not None:
        ax.set_title(
            f"{returns_label} - Monthly Active Returns (%)\n",
            fontsize=14,
            y=0.995,
            fontname=fontname,
            fontweight="bold",
            color="black",
        )
        benchmark = (
            core.monthly_returns(benchmark, eoy=eoy, compounded=compounded) * 100
        )
        active_returns = returns - benchmark

        ax = _sns.heatmap(
            active_returns,
            ax=ax,
            annot=True,
            center=0,
            annot_kws={"size": annot_size},
            fmt="0.2f",
            linewidths=0.5,
            square=square,
            cbar=cbar,
            cmap=cmap,
            cbar_kws={"format": "%.0f%%"},
        )
    else:
        ax.set_title(
            f"{returns_label} - Monthly Returns (%)\n",
            fontsize=14,
            y=0.995,
            fontname=fontname,
            fontweight="bold",
            color="black",
        )
        ax = _sns.heatmap(
            returns,
            ax=ax,
            annot=True,
            center=0,
            annot_kws={"size": annot_size},
            fmt="0.2f",
            linewidths=0.5,
            square=square,
            cbar=cbar,
            cmap=cmap,
            cbar_kws={"format": "%.0f%%"},
        )
    # _sns.set(font_scale=1)

    # align plot to match other
    if ylabel:
        ax.set_ylabel("Years", fontname=fontname, fontweight="bold", fontsize=12)
        ax.yaxis.set_label_coords(-0.1, 0.5)

    ax.tick_params(colors="#808080")
    _plt.xticks(rotation=0, fontsize=annot_size * 1.2)
    _plt.yticks(rotation=0, fontsize=annot_size * 1.2)

    try:
        _plt.subplots_adjust(hspace=0, bottom=0, top=1)
    except Exception:
        pass
    try:
        fig.tight_layout(w_pad=0, h_pad=0)
    except Exception:
        pass

    if savefig:
        if isinstance(savefig, dict):
            _plt.savefig(**savefig)
        else:
            _plt.savefig(savefig)

    if show:
        _plt.show(block=False)

    _plt.close()

    if not show:
        return fig

    return None


def monthly_returns(
    returns,
    annot_size=10,
    figsize=(10, 5),
    cbar=True,
    square=False,
    compounded=True,
    eoy=False,
    fontname="Arial",
    ylabel=True,
    savefig=None,
    show=True,
):
    return monthly_heatmap(
        returns=returns,
        annot_size=annot_size,
        figsize=figsize,
        cbar=cbar,
        square=square,
        compounded=compounded,
        eoy=eoy,
        fontname=fontname,
        ylabel=ylabel,
        savefig=savefig,
        show=show,
    )
