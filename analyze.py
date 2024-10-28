import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from collections import defaultdict
from tqdm import tqdm
from typing import List
from sortedcontainers import SortedDict

from library import LXC, ensure_path, sing, report_single_LP, plot_whale_shrimp, plot_ecdf, linestyle_cycle, \
    marker_cycle, LPInfo, Loader, Streak, Essence, clear_excluded, LiqConverter

matplotlib.use('Agg')

NCOLS = 6
PERCENT = 99


def datascout(datadir, gamedir, ignore_first=False):
    dates = {v: set() for v in ['game', 'profit', 'nash']}
    for file in os.listdir(datadir):
        if file.startswith('gamedata-') and file.endswith('.json'):
            dates['game'].add(file[9:-5])
        elif file.startswith('profit-') and file.endswith('.csv'):
            dates['profit'].add(file[7:-4])
    for file in os.listdir(gamedir):
        if file.startswith('nash-') and file.endswith('.json'):
            dates['nash'].add(file[5:-5])

    dates = sorted(dates['game'] & dates['profit'] & dates['nash'])
    return dates[1:] if ignore_first else dates


def dataiter(datadir, gamedir, scout):
    for date in scout:
        with open(f'{datadir}/gamedata-{date}.json', 'r') as f:
            game = json.load(f)
        with open(f'{gamedir}/nash-{date}.json', 'r') as f:
            nash = json.load(f)
        yield date, game, nash


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=int, required=True, help="Pool number")
    parser.add_argument('-m', type=int, default=None, help="Max number of lps to plot in daily figs")
    parser.add_argument('--keyword', type=str, required=True, help="Analysis keyword; e.g., 'raw', 'bent'")
    parser.add_argument('--notify', action='store_true', help="Notify on competition with sound effect")
    args = parser.parse_args()

    PID = args.p
    ESS = Essence()
    STREAK = Streak()

    datadir = f'dynamics/{PID}-archive'
    gamedir = f'game/{PID}-{args.keyword}'

    scout = datascout(datadir, gamedir, ignore_first=True)
    bar = tqdm(total=len(scout))

    roi_list, gt_overlap_list, center_comparison = [defaultdict(list) for _ in range(3)]
    whale_ratios, shrimp_ratios, gt_overlaps = [defaultdict(SortedDict) for _ in range(3)]
    price_change, budget_sums, fee_sums, jit_sums, daily_total_roi = [defaultdict(SortedDict) for _ in range(5)]

    all_lp_hashes, called_lp_hashes = set(), set()

    for date, GINFO, NASH in dataiter(datadir, gamedir, scout):
        bar.set_description(date)

        TOP_DATA = Loader(NASH, LiqConverter(GINFO['q'], GINFO['usd0'][0], GINFO['usd1'][0]))
        LPLIST = TOP_DATA.LP_list()

        ESS.daily_tvl.append(sum(GINFO['budget'].values()))
        ESS.ranges.append(TOP_DATA.attribute('range_count'))
        ESS.ranges_jit_only.append(TOP_DATA.attribute('range_jit'))

        price_change[date] = (GINFO['qnew'] - GINFO['q']) / GINFO['q']
        budget_sums[date] = sum(GINFO['budget'].values())
        fee_sums[date] = sum(GINFO['total_fee'])
        jit_sums[date] = sum(jit for jit in GINFO['jit'] if jit < 1e9)

        ESS.daily_invest.append((jit_sums[date] + budget_sums[date], TOP_DATA.TVL))
        ESS.daily_fee.append((fee_sums[date], sum(GINFO['player_fee'])))

        whale, shrimp = TOP_DATA.hypo_budget(True), TOP_DATA.hypo_budget(False)
        for k, v in whale.items():
            whale_ratios[k][date] = v
        for k, v in shrimp.items():
            shrimp_ratios[k][date] = v

        utils_window, gt_overlap_window = [defaultdict(float) for _ in range(2)]

        all_lp_hashes.update(LPLIST)

        if args.m is not None:
            nr = 1 + (min(args.m, len(TOP_DATA))*3 - 1) // NCOLS
            fig, axs = plt.subplots(nrows=nr, ncols=NCOLS, figsize=(4.5*NCOLS, 3.5*nr), dpi=150)
            axs: List[plt.Axes] = axs.flatten()

        for lpid, lp in enumerate(LPLIST):
            DATA: LPInfo = TOP_DATA[lp]
            STREAK.add(lp, DATA)

            action_bars = {key: DATA.action(key, bar=True) for key in ["NE", "BR"]}
            action_bars["GT"] = DATA.action_gt_full(liquidity=False)
            ESS.compare.append((DATA.action_gt_full(liquidity=True), TOP_DATA.action_in_liq(action_bars["NE"])))
            ESS.utils['NE_all'].append(DATA.ne_utility)
            ESS.rois['NE_all'].append(DATA.ne_utility / DATA.budget)

            if not DATA.is_lazy():
                called_lp_hashes.add(lp)

            for key in ["GT", "NE", "BR"]:
                pmin, pmax = action_bars[key].margins()
                ESS.pos_count[key].append(DATA.pos_count(key))
                ESS.pos_span[key].append(pmax / pmin)

            if "R_NE" in DATA and "I_NE" in DATA:
                for key in DATA:
                    ESS.gt_olaps[key].append(DATA.gt_overlap(key))
                    ESS.utils[key].append(DATA.utility(key))
                    ESS.rois[key].append(DATA.roi(key))
                    ESS.gaps[key].append(DATA.gap(key) / DATA.budget)

            for key in DATA:
                roi_list[key].append(DATA.roi(key))

                gt_overlap_list[key].append(DATA.gt_overlap(key))
                gt_overlap_window[key] += DATA.gt_overlap(key) * DATA.budget
                utils_window[key] += DATA.utility(key)
            if args.m is not None and lpid < args.m:
                report_single_LP(DATA, lpid, axs[lpid*3:lpid*3+3])

        for key, ov in gt_overlap_window.items():
            daily_total_roi[key][date] = utils_window[key] / TOP_DATA.TVL
            if key != "GT":
                gt_overlaps[key][date] = ov / TOP_DATA.TVL

        if args.m is not None:
            min_price, max_price = TOP_DATA.ne_range()
            for ax in axs[:3]:
                ax.set_ylabel('Occurance')
                ax.set_title(f"{len(TOP_DATA)} (top {TOP_DATA.attribute('percentage')}%) Investors")
                ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.))

            for ax in axs[:3+3*len(TOP_DATA)]:
                ax.grid(True)
            for i, ax in enumerate(axs[3:3+3*len(TOP_DATA)]):
                if i % 3 != 2:
                    ax.legend()
                    ax.set_xlabel('Price')
                    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
                    ax.ticklabel_format(style='sci', axis='y')
                    ax.set_ylabel('Liquidity')
                    ax.set_xlim(min_price / 1.1, max_price * 1.1)

            fig.tight_layout()
            fig.savefig(f'{gamedir}/analysis-{date}.jpg')
            plt.close('all')

        bar.update()

    bar.close()

    ensure_path(f'essence-{args.keyword}', empty=False)
    ESS.export(f"essence-{args.keyword}/{PID}.pkl")

    print(f'{len(called_lp_hashes)}/{len(all_lp_hashes)} LPs joined inert. '
          f"{sum(ESS.ranges_jit_only)}/{sum(ESS.ranges)} ranges have 0 player investment. ")

    clear_excluded(roi_list, gt_overlap_list, daily_total_roi, gt_overlaps)

    # ==================================================================================================================

    def date2ts(k):
        return pd.to_datetime(k, format='%y-%m-%d')

    def date2xy(data: dict):
        return [date2ts(k) for k in data], np.array(list(data.values()))

    nrows = 5 if whale_ratios or shrimp_ratios else 4
    fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=(14, 14), dpi=300, squeeze=False)
    axs = axs.flatten()
    axiter = iter(axs)

    plt.xticks(rotation=45)
    ax = next(axiter)
    x, y = date2xy(price_change)
    ax.plot(x, y, 'x-', alpha=.5, lw=.7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d"))
    ax.set_ylabel('Daily Price Change')
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.))

    ax = next(axiter)
    ls, ms = linestyle_cycle(), marker_cycle()
    x, y = date2xy(budget_sums)
    ax.plot(x, y, ls=next(ls), marker=next(ms), label='Budget Sum', lw=.6, alpha=.5)
    x, y = date2xy(jit_sums)
    ax.plot(x, y, ls=next(ls), marker=next(ms), color='k', label='JIT Sum', lw=.6,  alpha=.5)
    ax.set_ylabel('Budget of Player/JIT')
    ax.legend(loc='upper left', ncols=2)
    ax = ax.twinx()
    x, y = date2xy(fee_sums)
    ax.plot(x, y, ls=next(ls), marker=next(ms), color='r', label='Fee Sum')
    ax.set_ylabel('Fees')
    ax.legend(loc='upper right')

    ax = next(axiter)
    ls, ms = linestyle_cycle(), marker_cycle()
    dates = list(sorted(daily_total_roi["R_BR"].keys()))
    br, gt, e_br = [daily_total_roi["BR"][d] for d in dates], [daily_total_roi["GT"][d]
                                                               for d in dates], [daily_total_roi["R_BR"][d] for d in dates]
    dates = [date2ts(k) for k in dates]
    for i, date in enumerate(dates):
        if e_br[i] >= gt[i]:
            ax.plot((date, date), (br[i], e_br[i]), color='cyan', lw=.7, ls='--')
            ax.plot((date, date), (e_br[i], gt[i]), color='cyan', lw=.7)
        else:
            ax.plot((date, date), (br[i], gt[i]), color='red', lw=.7, ls='--')
            ax.plot((date, date), (e_br[i], gt[i]), color='red', lw=.7)

    ax.scatter(dates, br, marker='^', c='k', s=3, label='BR')
    ax.scatter(dates, gt, marker='x', c='g', s=3, label='GT')
    ax.scatter(dates, e_br, marker='*', c='b', s=3, label='E-BR')
    up, down = np.quantile(br+gt+e_br, [.99, .01])
    ax.set_ylim(down * 1.2, up * 1.2)
    ax.set_ylabel('Util / Budget')

    ax = next(axiter)
    ls, ms = linestyle_cycle(), marker_cycle()
    for key, value in gt_overlaps.items():
        x, y = [pd.to_datetime(k, format='%y-%m-%d') for k in value.keys()], list(value.values())
        ax.plot(x, y, ls=next(ls), marker=next(ms), alpha=.5, lw=.7, label=f'{key}')

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d"))
    ax.set_ylabel('Average GT Overlap')
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.))

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d"))
    ax.set_ylabel('Average BR Overlap')
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.))

    if whale_ratios or shrimp_ratios:
        ax = next(axiter)
        ls, ms = linestyle_cycle(), marker_cycle()
        for key, value in whale_ratios.items():
            if key == "BR":
                x, y = date2xy(value)
                ax.plot(x, y, ls=next(ls), marker=next(ms), alpha=.5, lw=.7, label=f'Whale ({key})')

        for key, value in shrimp_ratios.items():
            if key == "BR":
                x, y = date2xy(value)
                ax.plot(x, y, ls=next(ls), marker=next(ms), alpha=.5, lw=.7, label=f'Shrimp ({key})')

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d"))
    ax.set_yscale('log')
    ax.set_ylabel('Max Investment')

    for ax in axs:
        ax.grid(True)
    for ax in axs[1:]:
        ax.legend()

    fig.tight_layout()
    fig.savefig(f'{gamedir}/analysis_summary.jpg')
    plt.close('all')

    # ==================================================================================================================
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), dpi=300, squeeze=False)
    axs = axs.flatten()
    axiter = iter(axs)

    ax = next(axiter)
    for key, ulist in roi_list.items():
        marker = 'x' if key == "GT" else None
        plot_ecdf(ulist, ax, label=f'{key}', marker=marker, markevery=.06)
    ax.set_xlabel('Utility / Budget')
    ax.set_ylabel('Probability <= x')
    ax.set_xlim(-.02, .02)

    ax = next(axiter)
    for key, olist in gt_overlap_list.items():
        marker = 'x' if key == "GT" else None
        plot_ecdf(olist, ax, start=0., label=key, marker=marker, percentage=True, markevery=.06)
    ax.set_xlabel('Overlap with GT')
    ax.set_ylabel('Probability <= x')

    for ax in axs:
        ax.legend()
        ax.grid(True)
    fig.tight_layout()
    fig.savefig(f'{gamedir}/analysis_cdf.jpg')
    plt.close('all')

    # ==================================================================================================================
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 3.5), dpi=300, squeeze=False)
    axs = axs.flatten()
    STREAK.summarize(axs[:])
    fig.tight_layout()
    fig.savefig(f'{gamedir}/analysis_holder_{np.format_float_scientific(Streak.SIMTHR, precision=1, exp_digits=1)}.jpg')
    plt.close('all')

    # ==================================================================================================================
    if args.notify:
        sing(['C4', 'E4', 'G4'], extend=2)

    gt_util, ibr_util, ine_util = np.mean(ESS.utils["GT"]), np.mean(ESS.utils["I_BR"]), np.mean(ESS.utils["I_NE"])
    ibr_olap, ne_olap = np.mean(ESS.gt_olaps["I_BR"]) * 100, np.mean(ESS.gt_olaps["NE"]) * 100
    print(f'  GT util = {gt_util:.1f},\n'
          f'I_BR util = {ibr_util:.1f} ({ibr_util - gt_util:.1f})\n'
          f'I_NE util = {ine_util:.1f} ({ine_util - gt_util:.1f})\n'
          f'I_BR olap = {ibr_olap:.1f}% ({ibr_olap - ne_olap:.1f}%)\n')
    print(LXC.green_lt(f'Finished at {pd.Timestamp.now()}'))
