from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

from typing import List, Dict, TYPE_CHECKING
from matplotlib.ticker import ScalarFormatter
from sklearn.linear_model import LinearRegression
from matplotlib.gridspec import GridSpec

from library.bars import Bars
from library.util import num_alias

if TYPE_CHECKING:
    from library.nash_parser import LPInfo


def centered_subplots(nplots: int, ncols: int, width: float, height: float, dpi=300) -> List[plt.Axes]:
    """
    Create a figure with a specified number of subplots arranged in a grid, centered within the figure.

    Args:
        nplots (int): The total number of subplots to create.
        ncols (int): The number of columns in the subplot grid.
        width (float): The width of each subplot in inches.
        height (float): The height of each subplot in inches.
        dpi (int, optional): The resolution of the figure in dots per inch. Default is 300.

    Returns:
        List[plt.Axes]: A list of matplotlib Axes objects for the created subplots.
    """
    nrows = (nplots - 1) // ncols + 1
    fig = plt.figure(figsize=(width*ncols, height*nrows), dpi=dpi)
    grid = GridSpec(ncols=2*ncols, nrows=nrows, figure=fig)
    d = nrows * ncols - nplots

    axs = []
    for i in range(nplots):
        r, c = divmod(i, ncols)
        if r == nrows-1 and d > 0:
            axs.append(fig.add_subplot(grid[r, d+c*2:d+c*2+2]))
        else:
            axs.append(fig.add_subplot(grid[r, c*2:c*2+2]))
    return fig, axs


def plot_ecdf(arr, ax: plt.Axes, start: float = None, end: float = None, percentage: bool = False, stats: bool = True, **plot_kwargs):
    """
    Plot the Empirical Cumulative Distribution Function (ECDF) of an array on a given matplotlib Axes.

    Args:
        arr (array-like): The data array for which the ECDF is to be plotted.
        ax (plt.Axes): The matplotlib Axes object where the ECDF will be plotted.
        start (float, optional): The starting value for the x-axis. Defaults to the first element of the sorted array.
        end (float, optional): The ending value for the x-axis. Defaults to the last element of the sorted array.
        percentage (bool, optional): If True, format the x-axis as percentages. Defaults to False.
        stats (bool, optional): If True, include mean and median statistics in the plot label. Defaults to True.
        **plot_kwargs: Additional keyword arguments to pass to the plot function.

    Returns:
        list: A list of Line2D objects representing the plotted data.
    """
    x, counts = np.unique(arr, return_counts=True)
    cusum = np.cumsum(counts)
    x = [x[0] if start is None else start] + list(x) + [x[-1] if end is None else end]
    y = [0.] + list(cusum / cusum[-1]) + [1.]

    if percentage:
        avg_str, med_str = f'{np.mean(arr) * 100:.1f} %', f'{np.median(arr) * 100:.1f} %'
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.))
    else:
        avg_str = np.format_float_scientific(np.mean(arr), precision=1, exp_digits=1)
        med_str = np.format_float_scientific(np.median(arr), precision=1, exp_digits=1)
    label = plot_kwargs.pop('label') if 'label' in plot_kwargs else ''
    if stats:
        label += f'  Mean {avg_str} | Med {med_str}'
    return ax.plot(x, y, drawstyle='steps-post', label=label, **plot_kwargs)


# All following functions are used for various plots in plot_dynamics function, which is not very important


def scatter_with_linreg(x, y, ax, xlabel, ylabel):
    X = x.reshape(-1, 1)
    ax.scatter(x, y, alpha=.5, s=3)
    reg = LinearRegression(fit_intercept=False).fit(X, y)
    k, b = reg.coef_[0], reg.intercept_
    ax.plot(x, k*x+b, ls='--', color='k')
    ax.set_title(f'k = {k:.2e}, R^2 = {reg.score(X, y):.2e}')
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.ticklabel_format(axis='both', scilimits=[0, 0])


def hist_liquidtiy(ax: plt.Axes, deltas, players=Bars(), removed=[], left=None, right=None, FP=1, FL=1):
    def tick_map(x): return np.power(1.0001, x) * FP
    def value_map(x): return np.array(x) * FL

    total: Bars = Bars.from_delta(deltas, sort=False)
    total.apply_numpy(tick_map=tick_map, value_map=value_map)

    if len(total) == 0:
        return

    if not players.empty():
        players.apply_numpy(tick_map=tick_map, value_map=value_map)

    ymax = max(total) * 1.1
    if len(removed) > 0:
        all_deltas = deltas + [(r[0], r[2], -1) for r in removed] + [(r[1], -r[2], -1) for r in removed]
        removed: Bars = Bars.from_delta(all_deltas, sort=True)
        removed.apply_numpy(tick_map=tick_map, value_map=value_map)
        removed.barplot(ax, color='y', edgecolor='k', label='Removed')
        ymax = min(max(total) * 2, max(removed) * 1.1)

    total.barplot(ax, color='b', edgecolor='k', label='Total')
    if not players.empty():
        players.barplot(ax, color='r', edgecolor='k', label='Players')

    ax.set_ylim(0, ymax)
    ax.xaxis.set_major_formatter(ScalarFormatter())

    ax.set_xlim(left, right)
    ax.legend()
    ax.set_xlabel('Price')
    ax.set_ylabel('Liquidity')
    ax.grid(True)


def hist_utility(ax: plt.Axes, df):
    if len(df) == 0:
        return
    bins = np.array([5, 10, 50, 100, 500, 1000, 5000, 10000, 50000])
    L = len(bins)
    bins = list(-bins[::-1]) + [0] + list(bins)
    calc, _ = np.histogram(df['utility'], bins=bins)
    ax.bar(np.arange(-L+.5, L), calc, width=1, align='center', edgecolor='black')
    ax.set_xticks(np.arange(-L, L+1))
    ax.set_xticklabels([num_alias(b) for b in bins])
    ax.set_xlabel('Utility')
    ax.grid(True)


def hist_totalgrad(ax: plt.Axes, df):
    if len(df) == 0:
        return
    w = .1
    bins = np.arange(-4, 4.001, w)
    L = len(bins) // 2
    calc, _ = np.histogram(df['totalGrad'], bins=bins)
    ax.bar(bins[:-1] + w/2, calc, width=w, align='center', edgecolor='black')
    ax.set_xlabel('Utility Gradient')
    ax.grid(True)


def hist_fee(ax: plt.Axes, player: Bars, total: Bars, mult=1):
    if player.empty() and total.empty():
        return

    def power_price(p):
        return np.power(1.0001, p) * mult

    player.apply(tick_map=power_price)
    total.apply(tick_map=power_price)

    total.barplot(ax, color='blue', edgecolor='black', label='Total')
    player.barplot(ax, color='red', edgecolor='black', label='Players')

    left, right = total.margins()
    ax.set_xlim(left / 1.2, right * 1.2)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.legend()
    ax.set_xlabel('Price')
    ax.set_ylabel('Fee')
    ax.grid(True)


def hist_jit_duration(ax: plt.Axes, removed):
    bins = [pd.Timedelta(hours=0), pd.Timedelta(hours=1), pd.Timedelta(hours=6), pd.Timedelta(
        days=1), pd.Timedelta(days=3), pd.Timedelta(days=7), pd.Timedelta(days=31), pd.Timedelta(days=20000)]
    values = [0] * (len(bins)-1)
    for _, _, _, value, duration in removed:
        for i, b in enumerate(bins[1:]):
            if duration <= b:
                values[i] += value
                break

    ax.bar(np.arange(len(bins)-1), values, width=1, align='edge', edgecolor='black')
    ax.set_xticks(np.arange(len(bins)))
    ax.set_xticklabels(['0', '1h', '6h', '1d', '3d', '1w', '1m', 'N/A'])
    ax.set_xlabel('Duration')
    ax.set_ylabel('Value')
    ax.grid(True)


def plot_price_change(ax: plt.Axes, price: Dict[pd.Timestamp, float], ymin=None, ymax=None):
    if len(price) == 0:
        return
    ax.plot(price.keys(), price.values())
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=30, horizontalalignment='right')
    ax.grid(True)


def scatter_fee_il(ax: plt.Axes, df):
    if len(df) == 0:
        return
    ax.scatter(df['fee'], -df['IL'], marker='x', alpha=.5)
    up = max(max(df['fee']), max(-df['IL']))
    down = max(1e-6, min(min(df['fee']), min(-df['IL'])))
    arr = np.exp(np.linspace(np.log(down), np.log(up), 100))
    ax.plot(arr, arr, '--', color='k')
    ax.set_aspect('equal')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(down/2, up*2)
    ax.set_ylim(down/2, up*2)
    ax.set_xlabel('Fee Income')
    ax.set_ylabel('Impermanent Loss')
    ax.grid(True)


def scatter_feegrad_ilgrad(ax: plt.Axes, df):
    if len(df) == 0:
        return
    ax.scatter(df['feeGrad'], -df['ILGrad'], marker='x', alpha=.5)
    up = max(np.quantile(df['feeGrad'], .9), np.quantile(-df['ILGrad'], .9))
    down = min(np.quantile(df['feeGrad'], .1), np.quantile(-df['ILGrad'], .1))
    arr = np.linspace(down, up, 100)
    ax.plot(arr, arr, '--', color='k')
    ax.set_aspect('equal')
    ax.set_xlim(down, up)
    ax.set_ylim(down, up)
    ax.set_xlabel('Fee Gradient')
    ax.set_ylabel('Impermanent Loss Gradient')
    ax.grid(True)


def scatter_fund_totalgrad(ax: plt.Axes, df):
    if len(df) == 0:
        return
    ax.scatter(df['initialFund'], df['totalGrad'], marker='x', alpha=.5)
    ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_xlabel('Initial Funds')
    ax.set_ylabel('Utility Gradient')
    ax.grid(True)


def scatter_fund_util_per_invest(ax: plt.Axes, df):
    if len(df) == 0:
        return
    y = df['utility'] / df['initialFund'] * 100
    ax.scatter(df['initialFund'], y, marker='x', alpha=.5)
    down, up = np.quantile(y, [.1, .9])
    ax.set_ylim(down, up)
    ax.set_xscale('log')
    ax.set_xlabel('Initial Funds')
    ax.set_ylabel('Utility Per Investment (%)')
    ax.grid(True)


def scatter_fund_utility(ax: plt.Axes, df):
    if len(df) == 0:
        return
    ax.scatter(df['initialFund'], df['utility'], marker='x', alpha=.5)
    ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_xlabel('Initial Funds')
    ax.set_ylabel('Utility')
    ax.grid(True)


def report_single_LP(data: LPInfo, lpid, axs: List[plt.Axes]):
    actions: Dict[str, Bars] = {key: data.action_in_liq(key) for key in data}

    ax = axs[0]
    ax.set_title(f'LP {lpid+1}: Action from Present')
    for key in ["GT", "NE", "BR"]:
        if key in actions:
            gt_kwargs = {"hatch": "/", "color": "w"} if key == "GT" else {}
            actions[key].barplot(ax, ec="k", alpha=.3, label=f'{key} ${num_alias(int(data.utility(key)))}', **gt_kwargs)

    ax = axs[1]
    ax.set_title(f'LP {lpid+1}: Action from Past')
    keys_r = [key for key in ["R_NE", "R_BR"] if key in actions]
    keys_i = [key for key in ["I_NE", "I_BR"] if key in actions]
    for key in keys_r + keys_i:
        actions[key].barplot(ax, ec="k", alpha=.3, label=f'{key} ${num_alias(int(data.utility(key)))}')
    for key in keys_i:
        actions[key].barplot(ax, ec="k", alpha=.3, label=f'{key} ${num_alias(int(data.utility(key)))}')
    if keys_i or keys_r:
        actions["GT"].barplot(ax, ec="k", alpha=.3, hatch='/', color='w',
                              label=f'{key} ${num_alias(int(data.utility(key)))}')

    ax = axs[2]
    ax.set_title(f'LP {lpid+1}: Budget ${num_alias(data.budget)} / BR ${num_alias(data.utility("BR"))}')
    plans = list(data)
    utils = [data.utility(p) for p in plans]
    ax.bar(np.arange(len(plans)), utils, width=1, align='center', edgecolor='k', color='y')
    ax.axhline(y=data.utility("GT"), ls='--', color='r')
    ax.set_xticks(np.arange(len(plans)))
    ax.set_xticklabels(plans)
    ax.set_ylabel('Utility ($)')

    ax = axs[2].twinx()
    ax.set_ylim(0, 1)
    ax.plot(np.arange(len(plans)), [data.br_overlap(p) for p in plans], marker='x', color='b')
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.))
    ax.set_ylabel('Overlap Score')


def plot_whale_shrimp(data, unit, axs):
    xw, yw, zw, xs, ys, zs = [], [], [], [], [], []
    for whale in data['__whale__']:
        xw.append(whale['added'] / unit)
        yw.append(whale['br']['util'])
        zw.append(whale['br']['used'] / unit)
    for shrimp in data['__shrimp__']:
        xs.append(shrimp['added'] / unit)
        ys.append(shrimp['br']['util'])
        zs.append(shrimp['br']['used'] / unit)

    ax = axs[0]
    ax.plot(xw, yw, '-x', label='Whale')
    ax.plot(xs, ys, '--^', label='Shrimp')
    ax.set_ylabel('Utility ($)')

    ax = axs[1]
    ax.plot(xw, zw, '-x', label='Whale')
    ax.plot(xs, zs, '--^', label='Shrimp')
    ax.set_xlabel("Added Budget (x whale's)")
    ax.set_ylabel("Used Budget (x whale's)")
    ax.set_yscale('log')

    for ax in axs:
        ax.grid(True)
        ax.legend(loc='lower right')
        ax.set_xscale('log')
