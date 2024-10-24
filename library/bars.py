from __future__ import annotations

import heapq
import bisect
import numpy as np

from typing import List, Tuple
from library.sysutil import LXC


def express(v):
    return f'{v}' if np.isclose(v, 0, atol=1e-5) else LXC.green_lt(f'{v}')


def t2p(*ticks):
    return [1.0001 ** t for t in ticks] if len(ticks) > 1 else 1.0001 ** ticks[0]


class Bars:
    def __init__(self, ticks=[], values=[]) -> None:
        """
        Initializes a Bars object with specified ticks and values.

        Parameters:
        ticks (list): A list of tick values. Must contain more than one element.
        values (list): A list of values corresponding to the intervals between ticks.
                       The length of values should be one less than the length of ticks.

        Raises:
        ValueError: If a single tick is provided or if the lengths of ticks and values do not match the expected relationship.
        """
        if len(ticks) == 1:
            raise ValueError(f'Single tick {ticks} forbidden')

        self.dtype = float
        self.T = ticks
        if len(ticks) > 0:
            if len(values) == len(ticks) - 1:
                self.V = values
            elif len(values) == 0:
                self.V = [0] * (len(ticks) - 1)
            else:
                raise ValueError(f'Ticks {len(ticks)} do not fit values {len(values)}')
        elif len(values) == 0:
            self.V = []
        else:
            raise ValueError(f'Ticks {ticks} do not fit values {values}')

    def __len__(self):
        return len(self.V)

    def __str__(self) -> str:
        return '<empty_bar>' if self.empty() else \
            f'[{self.T[0]}] ' + ' '.join(f'{express(v)} [{b}]' for b, v in zip(self.T[1:], self.V)) + '\n'

    def __max__(self):
        return max(self.V)

    def __min__(self):
        return min(self.V)

    def __getitem__(self, i):
        return self.V[i]

    def __iter__(self):
        return iter(self.V)

    def empty(self):
        return len(self.V) == 0

    def apply(self, tick_map=None, value_map=None):
        if tick_map is not None:
            self.T = [tick_map(t) for t in self.T]
        if value_map is not None:
            self.V = [value_map(v) for v in self.V]

    def apply_numpy(self, tick_map=None, value_map=None):
        if tick_map is not None:
            self.T = list(np.array(tick_map(self.T)))
        if value_map is not None:
            self.V = list(np.array(value_map(self.V)))

    def intervals(self):
        if not self.empty():
            for i in range(len(self.V)):
                yield (self.V[i], self.T[i], self.T[i+1])

    def rebase(self, ticks):
        _, bars = align_bars(Bars(ticks), self, standard=0)
        self.T, self.V = bars.T, bars.V

    def overlap(self, array, budget, i=None, j=None):
        """
        Calculate the overlap between the current Bars object and another array of values.

        Parameters:
        array (Bars or list): The target array or Bars object to compare against.
        budget (float): The budget value used for normalization in the overlap calculation.
        i (int, optional): The starting index for the overlap calculation. Defaults to the beginning of the Bars.
        j (int, optional): The ending index for the overlap calculation. Defaults to the end of the Bars.

        Returns:
        float: The calculated overlap value, normalized by the budget. Returns 0 if the overlap is close to zero.

        Raises:
        ValueError: If the sizes of the specified range in the Bars and the target array do not match.
        """
        if i is None:
            i = 0
        if j is None:
            j = len(self)
        if i < 0 or j > len(self) or len(array) != j-i:
            raise ValueError(f'Sizes do not match: self {len(self)} != target {len(array)}')

        if isinstance(array, Bars):
            array = array.V

        olap = 1 - (abs(sum(self.V) - sum(array)) + sum(abs(a - b) for a, b in zip(array, self.V[i:j])) +
                    sum(self.V[:i]) + sum(self.V[j:])) / (2 * budget)
        return olap if not np.isclose(olap, 0., atol=1e-6) else 0.

    def impermanent_loss(self, q0: float, p1: Tuple[float, float, float], asliq: bool = True) -> float:
        """
        Calculate the impermanent loss for a given initial quantity and price changes.

        Parameters:
        q0 (float): The initial quantity.
        p1 (Tuple[float, float, float]): A tuple containing the new quantity and two price factors (q1, px1, py1).
        asliq (bool, optional): A flag indicating whether the calculation is in liquidity terms. Defaults to True.

        Returns:
        float: The calculated impermanent loss.
        """
        assert asliq
        q1, px1, py1 = p1
        r0 = np.maximum(np.minimum(q0, self.T[1:]), self.T[:-1]) ** .5
        r1 = np.maximum(np.minimum(q1, self.T[1:]), self.T[:-1]) ** .5
        return sum(self.V * (px1 * (1/r0-1/r1) + py1 * (r0-r1)))

    def margins(self, tol=1e-5) -> Tuple[float, float]:
        i, j = self.margin_idxs(tol)
        return self.T[i], self.T[j]

    def margin_idxs(self, tol=1e-5) -> Tuple[int, int]:
        i, j = 0, len(self)-1
        M = max(self.V) * tol
        while i < len(self) and self.V[i] < M:
            i += 1
        while j >= 0 and self.V[j] < M:
            j -= 1
        return i, j+1

    def truncate(self, a: int, b: int) -> Bars:
        return Bars(self.T[a:b+1], self.V[a:b])

    def a(self, i):
        return self.T[i]

    def b(self, i):
        return self.T[i+1]

    def bar(self, ax, **kwargs):
        ax.bar(self.T[:-1], self.V, width=np.array(self.T[1:])-np.array(self.T[:-1]), align='edge', **kwargs)

    def from_delta(deltas: List[tuple], sort=False, assert_positive=True) -> Bars:
        ticks, values = [], []
        i = v = 0
        if sort:
            deltas = sorted(deltas)
        while i < len(deltas):
            j = i
            ticks.append(deltas[i][0])
            while j < len(deltas) and deltas[j][0] == deltas[i][0]:
                v += deltas[j][1]
                j += 1
            if j < len(deltas):
                if isinstance(v, float) and np.isclose(v, 0, atol=1e-6):
                    v = 0
                values.append(v)
                if assert_positive and v < 0:
                    raise ValueError(f'Negative deltas:\n{deltas}, \n{values}')
            i = j
        return Bars(ticks, values)

    def to_delta(self) -> List[tuple]:
        prev_v, deltas = 0, []
        for i in range(len(self)):
            deltas.append((self.T[i], self.V[i] - prev_v))
            prev_v = self.V[i]

        return deltas + [(self.T[-1], -self.V[-1])]

    def average(self, geometric=True, default=0.):
        dividend = divisor = 0.
        for L, a, b in self.intervals():
            if geometric:
                a, b = np.log(a), np.log(b)
            dividend += L/2 * (b**2 - a**2)
            divisor += L * (b-a)

        if divisor == 0:
            return default
        return np.exp(dividend / divisor) if geometric else dividend / divisor

    def resample_fee(self, ticks: list, is_token_0: bool) -> Bars:
        """
        Resample the fee distribution over a new set of ticks.

        Parameters:
        ticks (list): A list of tick values to resample the fee distribution over.
        is_token_0 (bool): A flag indicating whether the calculation is for token 0.

        Returns:
        Bars: A new Bars object representing the resampled fee distribution.
        """
        i = j = curr = 0
        res = []

        while i < len(ticks) - 1:
            incr_j = j < len(self) and self.b(j) <= ticks[i+1]

            if j < len(self) and self.b(j) > ticks[i] and ticks[i+1] > self.a(j):
                a, b = self.a(j), self.b(j)
                l, r = max(a, ticks[i]), min(b, ticks[i+1])
                if is_token_0:
                    curr += self[j] * (r**.5-l**.5)/(b**.5-a**.5)
                else:
                    curr += self[j] * (l**-.5-r**-.5)/(a**-.5-b**-.5)

            if incr_j:
                j += 1
            else:
                res.append(curr)
                curr = 0
                i += 1

        return Bars(ticks, res)

    def calculate_fee(self, q_old: float, q_new: float, fee_rate: float) -> Bars:
        """
        Calculate the fees based on the change in pool price and the fee rate.

        Parameters:
        q_old (float): The initial pool price before the change.
        q_new (float): The new pool price after the change.
        fee_rate (float): The rate at which the fee is calculated.

        Returns:
        Bars: A new Bars object representing the calculated fees over the specified tick intervals.
        """
        fees = []
        t_old, t_new = [np.log(q) / np.log(1.0001) for q in [q_old, q_new]]
        if q_old < q_new:
            j_old, j_new = bisect.bisect_right(self.T, t_old) - 1, bisect.bisect_left(self.T, t_new)
            tick_slice = self.T[j_old:j_new+1]
            for j in range(j_old, j_new):
                qa, qb = max(q_old, t2p(self.a(j))), min(q_new, t2p(self.b(j)))
                fees.append(self[j] * (qb**.5 - qa**.5) * fee_rate / (1-fee_rate))
        else:
            j_new, j_old = bisect.bisect_right(self.T, t_new) - 1, bisect.bisect_left(self.T, t_old)
            tick_slice = self.T[j_new:j_old+1]
            for j in range(j_new, j_old):
                qa, qb = max(q_new, t2p(self.a(j))), min(q_old, t2p(self.b(j)))
                fees.append(self[j] * (qa**-.5 - qb**-.5) * fee_rate / (1-fee_rate))

        return Bars(tick_slice, fees)

    def align(self, ticks, force_bounds=False) -> Bars:
        """
        Aligns the current Bars object with a new set of ticks, optionally forcing bounds.

        Parameters:
        ticks (list): A list of tick values to align with the current Bars object.
        force_bounds (bool, optional): A flag indicating whether to force the alignment to respect the bounds of the new ticks. Defaults to False.

        Returns:
        Bars: A new Bars object with ticks and values aligned to the specified ticks.
        """
        newT, newV = [], []
        i = j = 0
        while i < len(self.T) or j < len(ticks):
            if j == len(ticks) or (i < len(self.T) and self.T[i] <= ticks[j]):
                if not force_bounds or self.T[i] >= ticks[0]:
                    newT.append(self.T[i])
                    if len(newT) > 1:
                        newV.append(self.V[i-1] if i > 0 else 0)
                if j < len(ticks) and np.isclose(self.T[i], ticks[j], atol=1e-8, rtol=1e-8):
                    j += 1
                i += 1
            elif i == len(self.T) or (j < len(ticks) and ticks[j] < self.T[i]):
                newT.append(ticks[j])
                if len(newT) > 1:
                    newV.append(0 if i == 0 or i == len(self.T) else self.V[i-1])
                j += 1
                if j == len(ticks):
                    break

        return Bars(newT, newV)


class LiqConverter:
    def __init__(self, q, usdx, usdy):
        self.q = q
        self.usdx = usdx
        self.usdy = usdy

    def to_liq(self, cash: Bars):
        if cash.empty():
            return Bars()

        res = []
        for v, a, b in cash.intervals():
            qm = max(a, min(b, self.q))
            res.append(v / (self.usdx * (qm**-.5 - b**-.5) + self.usdy * (qm**.5 - a**.5)))
        return Bars(cash.T, res)

    def to_cash(self, liq: Bars):
        if liq.empty():
            return Bars()

        res = []
        for v, a, b in liq.intervals():
            qm = max(a, min(b, self.q))
            res.append(v * (self.usdx * (qm**-.5 - b**-.5) + self.usdy * (qm**.5 - a**.5)))

        return Bars(liq.T, res)

    def single_range(self, ticks, rewards, thr=.7):
        """
        Determine the optimal range of ticks that captures a specified threshold of rewards.

        Parameters:
        ticks (list): A list of tick values representing the range boundaries.
        rewards (list): A list of reward values corresponding to each tick interval.
        thr (float, optional): The threshold proportion of total rewards to capture. Defaults to 0.7.

        Returns:
        tuple: A tuple containing the start tick, end tick, and the minimum distance (dmin) 
               for the optimal range that captures the specified threshold of rewards.
        """
        thr = np.round(thr*sum(rewards), 9)
        imin = jmin = dmin = float('inf')
        r = i = j = 0
        while j < len(ticks):
            if r < thr:
                if j == len(ticks)-1:
                    break
                r += rewards[j]
                j += 1
            else:
                qm = min(ticks[j], max(ticks[i], self.q))
                d = self.usdx * (qm**-.5 - ticks[j]**-.5) + self.usdy * (qm**.5 - ticks[i]**.5)
                if dmin > d:
                    imin, jmin, dmin = i, j, d

                r -= rewards[i]
                i += 1

        return ticks[imin], ticks[jmin], dmin

    def overlap_general(self, budget, action_usd: Bars, *baselines_usd: Bars):
        """
        Calculate the overlap between a given action in USD and multiple baseline USD Bars.

        Parameters:
        budget (float): The budget value used for normalization in the overlap calculation.
        action_usd (Bars): The Bars object representing the action in USD to compare against the baselines.
        baselines_usd (Bars): One or more Bars objects representing the baseline USD values for comparison.

        Returns:
        list or float: A list of overlap values for each baseline if multiple baselines are provided,
                       or a single overlap value if only one baseline is provided.
        """
        liq, olaps = self.to_liq(action_usd), []
        for baseline in baselines_usd:
            liq_temp, baseline_liq = align_bars(liq, self.to_liq(baseline), standard=-1)
            olaps.append(self.to_cash(baseline_liq).overlap(self.to_cash(liq_temp), budget))

        return olaps if len(baselines_usd) > 1 else olaps[0]


def align_bars(*bars: Bars, default=0, standard=-1):
    """
    Aligns multiple Bars objects to a common set of ticks, filling in missing values with a default.

    Parameters:
    bars (Bars): One or more Bars objects to be aligned.
    default (float, optional): The default value to use for missing intervals. Defaults to 0.
    standard (int, optional): The index of the Bars object to use as the standard for determining the range of ticks. 
                              If negative, the full range of all Bars is used. Defaults to -1.

    Returns:
    list: A list of new Bars objects, each aligned to the common set of ticks with values filled in as necessary.
    """
    H = len(bars)
    if H == 1:
        return bars[0]

    lo = float('-inf') if standard < 0 else bars[standard].T[0]
    hi = float('inf') if standard < 0 else bars[standard].T[-1]

    flag_first, tick = True, 0
    t, u, progress = [], [[] for _ in range(H)], [0] * H

    heap = [(bars[i].a(0), 0, i) for i in range(H) if not bars[i].empty()]
    heapq.heapify(heap)

    while heap:
        tick = heap[0][0]
        if tick > hi:
            break

        if tick >= lo:
            if not flag_first:
                for i in range(H):
                    u[i].append(bars[i][progress[i]-1] if 0 < progress[i] <= len(bars[i]) else bars[i].dtype(default))

            t.append(tick)
            flag_first = False

        while heap and heap[0][0] == tick:
            _, j, i = heapq.heappop(heap)
            if j < len(bars[i]):
                heapq.heappush(heap, (bars[i].b(j), j+1, i))
            progress[i] = j+1

    return [type(bar)(t, uu) for bar, uu in zip(bars, u)]


def main_bars_test_1():
    h1 = Bars([1, 4, 9, 16], [2, 3, 4])
    h2 = Bars([22, 26, 30], [5, 8])
    d1 = h1.to_delta()
    h3 = Bars.from_delta(d1)

    b1, b2 = align_bars(h1, h2, default=0, standard=1)

    print(h1, b1, sep='\n', end='\n\n')
    print(h2, b2, sep='\n')
    print(h1.to_delta())
    print(h3)


def main_bars_test_2():
    h1 = Bars([4, 8, 16], [1, 2])
    print(h1.average(geometric=True))
    h1 = h1.align([5, 10], force_bounds=True)
    print(h1)
    # print(h1)


if __name__ == '__main__':
    main_bars_test_2()
    pass
