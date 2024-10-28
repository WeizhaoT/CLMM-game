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
""" Name of ticks to load for each strategy. """


def exclude(key: str) -> bool:
    """
    Check if a given key should be excluded from further processing.

    Args:
        key (str): The key to be checked.

    Returns:
        bool: True if the key should be excluded, False otherwise.

    The function checks if the key starts and ends with double underscores ('__') or if it is one of the following:
    "R_NE", "NE", "I_BR". If any of these conditions are met, the function returns True; otherwise, it returns False.
    """
    return (key.startswith('__') and key.endswith('__')) or key in ["R_NE", "NE", "I_BR"]


def clear_excluded(*summaries: Dict[str, Any]):
    """
    Remove keys from the provided summaries that should be excluded based on specific criteria.

    Args:
        summaries (Dict[str, Any]): One or more dictionaries containing summary data.

    The function iterates over each provided summary dictionary and removes any keys that meet the exclusion criteria defined in the `exclude` function.
    """
    for summary in summaries:
        for key in list(summary.keys()):
            if exclude(key):
                summary.pop(key)


class LPInfo:
    """ Strategies of a liquidity provider (LP) in game. """

    def __init__(self, data: Dict[str, dict], ticks: dict, conv: LiqConverter):
        """
        Initialize an LPInfo instance with strategy data, tick information, and a liquidity converter.

        Args:
            data (Dict[str, dict]): A dictionary containing strategy data for the liquidity provider (LP).
                                    Keys are strategy identifiers, and values are dictionaries with strategy details.
            ticks (dict): A dictionary containing tick information, keyed by strategy identifiers.
            conv (LiqConverter): An instance of LiqConverter used for converting actions to liquidity.

        ## Initializes the following attributes:
        - **I (Dict[str, dict])**: This LP's strategies, excluding miscellaneous information.
        - **misc (Dict[str, Any])**: Miscellaneous global information extracted from the data.
        - **T (dict)**: Different ticks keyed by strategies, used for action retrieval.
        - **C (LiqConverter)**: The liquidity/cash converter instance provided as an argument.
        """
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
        """
        Retrieve the Nash Equilibrium utility value for the liquidity provider (LP) when other LPs are at NE.

        Returns:
            float: The utility value associated with the Nash Equilibrium strategy.
        """
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
        """
        Retrieve the action associated with a given strategy for the liquidity provider (LP).

        Args:
            strategy (str): The strategy key for which the action is to be retrieved.
            bar (bool, optional): If True, returns the action as a Bars object using the strategy's tick key.
                                  If False, returns the raw action data. Defaults to False.

        Returns:
            Union[Bars, Any]: The action data for the specified strategy. If `bar` is True, returns a Bars object;
                              otherwise, returns the raw action data.
        """
        return Bars(self.T[TICKKEY[strategy]], self.I[strategy]['action']) if bar else self.I[strategy]['action']

    def action_gt_full(self, liquidity=False) -> Bars:
        """
        Retrieve the full ground truth action, including price ranges with zero fee income. 

        Args:
            liquidity (bool, optional): If True, converts the action data to liquidity using the LiqConverter.
                                        Defaults to False.

        Returns:
            Bars: The action data for the "GT" strategy. If `liquidity` is True, returns the data converted to liquidity;
                  otherwise, returns the raw Bars object.
        """
        bars = Bars(self.T['all'], self.I["GT"]["action_data"])
        return self.C.to_liq(bars) if liquidity else bars

    def action_in_liq(self, strategy: str) -> Bars:
        """
        Convert the action associated with a given strategy to liquidity.

        Args:
            strategy (str): The strategy key for which the action is to be converted.

        Returns:
            Bars: The action data for the specified strategy converted to liquidity.
        """
        return self.C.to_liq(self.action(strategy, bar=True))

    def pos_count(self, strategy: str) -> int:
        """
        Calculate the position count for a given strategy.

        Args:
            strategy (str): The strategy key for which the position count is to be calculated.

        Returns:
            int: The number of positions for the specified strategy. If 'position_count' is available in the strategy's data, it returns that value.
                 Otherwise, it calculates the count by summing the number of positive actions.
        """
        return self.I[strategy].get('position_count', sum(np.greater(self.action(strategy, bar=False), 0)))

    def gt_overlap(self, strategy: str) -> float:
        """
        Calculate the overlap with the ground truth (GT) for a given strategy.

        Args:
            strategy (str): The strategy key for which the overlap is to be calculated.

        Returns:
            float: The overlap value. Returns 1.0 if the strategy is "GT", otherwise returns the overlap value from the strategy's data.
        """
        return 1. if strategy == "GT" else self.I[strategy]['olap_gt']

    def utility(self, strategy) -> float:
        """
        Retrieve the utility value for a given strategy, when other LPs are taking GT.

        Args:
            strategy (str): The strategy key for which the utility value is to be retrieved.

        Returns:
            float: The utility value associated with the specified strategy.
        """
        return self.I[strategy]['util']

    def roi(self, strategy):
        """
        Calculate the return on investment (ROI) for a given strategy.

        Args:
            strategy (str): The strategy key for which the ROI is to be calculated.

        Returns:
            float: The ROI value, calculated as the utility of the strategy divided by the budget.
        """
        return self.I[strategy]['util'] / self.budget

    def gap(self, strategy: str):
        """
        Calculate the utility optimality gap between the best response (BR) strategy and a given strategy.

        Args:
            strategy (str): The strategy key for which the utility gap is to be calculated.

        Returns:
            float: The difference in utility between the best response strategy and the specified strategy.
        """
        return self.I["BR"]["util"] - self.I[strategy]["util"]

    def is_lazy(self, threshold=.95):
        """
        Determine if the liquidity provider (LP) is considered 'lazy' based on the overlap with yesterday's action.

        Args:
            threshold (float, optional): The threshold value for determining laziness. Defaults to 0.95.

        Returns:
            bool: True if the LP's overlap with GT for the "YDay" strategy exceeds the threshold, indicating laziness; False otherwise.
        """
        return "YDay" in self.I and self.I["YDay"]['olap_gt'] > threshold


class Loader:
    """ Information of one single game. Contains LPInfo for each LP. """

    def __init__(self, data: dict, conv: LiqConverter) -> None:
        """
        Initialize a Loader instance with game data and a liquidity converter.

        Args:
            data (dict): A dictionary containing game data, where keys are either LP identifiers or metadata keys.
            conv (LiqConverter): An instance of LiqConverter used for converting actions to liquidity.

        ## Initializes the following attributes:
        - **tick (Dict[str, Union[int, float]])**: 
            A dictionary to store tick information extracted from the data.
        - **lpdata (Dict[str, LPInfo])**: 
            A dictionary mapping LP identifiers to their corresponding LPInfo objects.
        - **misc (Dict[str, Any])**: 
            A dictionary to store miscellaneous information extracted from the data.
        - **conv (LiqConverter)**: 
            The liquidity converter instance provided as an argument.
        - **TVL (float)**: 
            The total value locked, calculated as the sum of budgets of all LPs.
        - **max_budget (float)**: 
            The maximum budget among all LPs, initialized to a very small value.
        """
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
        """
        Retrieve a specific attribute from the miscellaneous information.

        Args:
            key (str): The key corresponding to the attribute to be retrieved.

        Returns:
            Any: The value associated with the specified key in the miscellaneous information.
        """
        return self.misc[key]

    def hypo_budget(self, whale=True) -> Dict[str, float]:
        try:
            hypo = self.attribute('whale' if whale else 'shrimp')
        except KeyError:
            return {}
        return {key: value['used'] / self.max_budget for key, value in hypo[-1].items() if isinstance(value, dict)}


class Streak:
    """ Tracking statistics for long-term LPs with unchanged liquidity of a streak of days """
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


class Essence:
    """ Essential summary of game strategies. """

    def __init__(self):
        self.ranges, self.ranges_jit_only = [[] for _ in range(2)]
        self.daily_tvl, self.daily_invest, self.daily_fee, self.compare = [[] for _ in range(4)]
        self.utils, self.gt_olaps, self.rois, self.gaps = [defaultdict(list) for _ in range(4)]
        self.pos_count, self.pos_span = [{key: [] for key in ["GT", "BR", "NE"]} for _ in range(2)]

    def export(self, filename):
        with open(f'{filename}', 'wb') as f:
            pickle.dump(self.__dict__, f)
