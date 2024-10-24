import pickle
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Dict, Tuple, Union, Iterator, Any
from matplotlib.patches import Rectangle
from collections import defaultdict
from library import LXC, Bars, linestyle_cycle, marker_cycle, LiqConverter
from library.util import weighted_average


TICKKEY = defaultdict(lambda: "", {
    "YDay": "pall",
    "R_BR": "prev",
    "R_NE": "prev",
    "I_BR": "inert",
    "I_NE": "inert",
})


def exclude(key: str) -> bool:
    return (key.startswith('__') and key.endswith('__')) or key in ["R_NE", "NE", "I_BR"]


def clear_excluded(*summaries: Dict[str, Any]):
    for summary in summaries:
        for key in list(summary.keys()):
            if exclude(key):
                summary.pop(key)


class LPInfo:
    def __init__(self, data: Dict[str, dict], ticks: dict, conv: LiqConverter):
        self.I: Dict[str, dict] = {}
        self.misc: Dict[str, Any] = {}
        self.T = ticks
        self.C = conv
        for key, value in data.items():
            if key.startswith('__') and key.endswith('__'):
                self.misc[key[2:-2]] = value
            else:
                self.I[key] = value

    @property
    def budget(self):
        return self.misc["budget"]

    @property
    def ne_utility(self):
        return self.I["NE"]['ne_util']

    def __iter__(self) -> Iterator[str]:
        return iter(self.I.keys())

    def __list__(self) -> List[str]:
        return list(self.I.keys())

    def __len__(self) -> int:
        return len(self.I)

    def __contains__(self, key):
        return key in self.I

    def action(self, strategy: str, bar=False):
        return Bars(self.T[TICKKEY[strategy]], self.I[strategy]['action']) if bar else self.I[strategy]['action']

    def action_gt_full(self, liquidity=False):
        bars = Bars(self.T['all'], self.I["GT"]["action_data"])
        return self.C.to_liq(bars) if liquidity else bars

    def action_in_liq(self, strategy: str):
        return self.C.to_liq(self.action(strategy, bar=True))

    def pos_count(self, strategy: str):
        return self.I[strategy].get('position_count', sum(np.greater(self.action(strategy, bar=False), 0)))

    def gt_overlap(self, strategy: str):
        return 1. if strategy == "GT" else self.I[strategy]['olap_gt']

    def utility(self, strategy):
        return self.I[strategy]['util']

    def roi(self, strategy):
        return self.I[strategy]['util'] / self.budget

    def gap(self, strategy: str):
        return self.I["BR"]["util"] - self.I[strategy]["util"]

    def is_lazy(self, threshold=.95):
        return "YDay" in self.I and self.I["YDay"]['olap_gt'] > threshold


class Loader:
    def __init__(self, data: dict, conv: LiqConverter) -> None:
        self.tick: Dict[str, Union[int, float]] = {}
        self.lpdata: Dict[str, LPInfo] = {}
        self.misc: Dict[str, Any] = {}
        self.conv: LiqConverter = conv
        self.TVL: float = 0.
        self.max_budget: float = 1e-14

        for key, value in data.items():
            if key.startswith('__') and key.endswith('__'):
                key = key[2:-2]
                if key.startswith('tick'):
                    self.tick[key[4:].strip('_')] = value
                else:
                    self.misc[key] = value
            else:
                self.lpdata[key] = LPInfo(value, self.tick, self.conv)
                self.TVL += self.lpdata[key].budget
                self.max_budget = max(self.max_budget, self.lpdata[key].budget)

    def __len__(self) -> int:
        return len(self.lpdata)

    def __str__(self) -> str:
        desc = f"<Loaded Data Summary>\n"
        desc += "  ticks:      " + ", ".join(self.tick.keys())
        desc += "  artifacts:  " + ", ".join(self.misc.keys())
        counts = defaultdict(int)
        for data in self.lpdata.values():
            for key in data:
                counts[key] += 1

        desc += f"  strategies: Total {len(self)}, " + ", ".join(f"{key} {c}" for key, c in counts.items())
        desc += f"<Summary End>\n"
        return LXC.blue_lt(desc)

    def __getitem__(self, lp) -> LPInfo:
        return self.lpdata[lp]

    def LP_list(self) -> List[str]:
        return list(self.lpdata.keys())

    def action_in_liq(self, action_in_cash) -> Bars:
        return self.conv.to_liq(action_in_cash)

    def ne_range(self) -> Tuple[float, float]:
        return self.tick[''][0], self.tick[''][-1]

    def attribute(self, key) -> Any:
        return self.misc[key]

    def hypo_budget(self, whale=True) -> Dict[str, float]:
        try:
            hypo = self.attribute('whale' if whale else 'shrimp')
        except KeyError:
            return {}
        return {key: value['used'] / self.max_budget for key, value in hypo[-1].items() if isinstance(value, dict)}


class Streak:
    SIMTHR = 1e-2

    def __init__(self) -> None:
        self.streak = defaultdict(int)
        self.roi, self.olap_gt = [defaultdict(list) for _ in range(2)]

    def add(self, lphash: str, lpinfo: LPInfo) -> None:
        s = self.streak[lphash] = self.streak[lphash] + 1 if lpinfo.is_lazy(1-Streak.SIMTHR) else 1
        for key in lpinfo:
            if not exclude(key):
                if key != "GT":
                    while len(self.olap_gt[key]) < s:
                        self.olap_gt[key].append([])
                    self.olap_gt[key][s-1].append((lpinfo.gt_overlap(key), lpinfo.budget))
                while len(self.roi[key]) < s:
                    self.roi[key].append([])
                self.roi[key][s-1].append((lpinfo.roi(key), lpinfo.budget))

    def summarize(self, axs: List[plt.Axes]) -> None:
        bar_kwargs = {'ec': 'k', 'lw': .25}
        width = .8

        ax = axs[0]
        sample = next(iter(self.olap_gt.values()))
        ax.bar(range(1, len(sample)+1), [sum(b for _, b in blist) for blist in sample], width=1, fc='y', **bar_kwargs)
        ax.set_yscale('log')
        ax.set_ylabel('Total Budget')

        ax = axs[1]
        gts, brs, r_brs = [weighted_average(self.roi[k]) for k in ["GT", "BR", "R_BR"]]
        ys = []
        for p, (gt, br, r_br) in enumerate(zip(gts, brs, r_brs)):
            ys.extend([gt, br, r_br])
            if gt >= r_br:
                ax.add_patch(Rectangle((p+1-width/2, gt), width, br-gt, color='y', **bar_kwargs))
                ax.add_patch(Rectangle((p+1-width/2, r_br), width, gt-r_br, color='r', **bar_kwargs))
            else:
                ax.add_patch(Rectangle((p+1-width/2, r_br), width, br-r_br, color='y', **bar_kwargs))
                ax.add_patch(Rectangle((p+1-width/2, gt), width, r_br-gt, color='c', **bar_kwargs))

        down, up = np.quantile(ys, [.01, .99])
        ax.set_xlim(0, p+1)
        ax.set_ylim(down - (up-down) * .1, up + (up-down) * .1)
        ax.set_ylabel('ROI (Util / Budget)')

        def plot_weighted_average(data: Dict[str, List[Tuple[float, float]]]):
            ls, ms = linestyle_cycle(), marker_cycle()
            line_kwargs = {'lw': .7, 'alpha': .7, 'ms': 4}
            for key, sample in data.items():
                wavg = weighted_average(sample)
                ax.plot(range(1, 1+len(wavg)), wavg, ls=next(ls), marker=next(ms), label=key, **line_kwargs)
            ax.legend()

        ax = axs[2]
        plot_weighted_average(self.olap_gt)
        ax.set_ylabel('Avg Overlap with GT')

        for ax in axs:
            ax.set_xlabel(f'Days with Change < {np.format_float_scientific(Streak.SIMTHR, precision=2, exp_digits=1)}')
            ax.grid(True)

        axs[1].grid(False)


class Stats:
    def __init__(self) -> None:
        self.D = defaultdict(list)

    def add_point(self, **kwargs):
        for key in self.D.keys():
            self.D[key].append(kwargs.get(key, None))

    def get_data(self, *keys) -> List[list]:
        if len(keys) == 0:
            return []
        assert all(key in self.D for key in keys)
        res = [[] for _ in keys]
        n = len(self.D[keys[0]])
        for i in range(n):
            row = []
            for key in keys:
                row.append(self.D[key][i])
                if row[-1] is None:
                    break
            if row[-1] is not None:
                for j in range(len(keys)):
                    res[j].append(row[j])
        return res


class Essence:
    def __init__(self):
        self.ranges, self.ranges_jit_only = [[] for _ in range(2)]
        self.daily_tvl, self.daily_invest, self.daily_fee, self.compare = [[] for _ in range(4)]
        self.utils, self.gt_olaps, self.rois, self.gaps = [defaultdict(list) for _ in range(4)]
        self.pos_count, self.pos_span = [{key: [] for key in ["GT", "BR", "NE"]} for _ in range(2)]

    def export(self, filename):
        with open(f'{filename}', 'wb') as f:
            pickle.dump(self.__dict__, f)
