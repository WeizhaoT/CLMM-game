from __future__ import annotations

import heapq
import bisect
import numpy as np

from typing import List, Tuple, Union
from library.sysutil import LXC


def express(v):
    """
    Format a value as a string, applying a specific style if the value is not close to zero.

    Args:
        v (float): The value to be formatted.

    Returns:
        str: The formatted string representation of the value. If the value is close to zero, it returns the value as a plain string.
            Otherwise, it applies a specific style using LXC.green_lt.
    """
    return f'{v}' if np.isclose(v, 0, atol=1e-5) else LXC.green_lt(f'{v}')


def t2p(*ticks):
    """
    Convert tick values to price values using a specific exponential formula.

    Args:
        ticks (float): One or more tick values to be converted.

    Returns:
        list or float: A list of price values if multiple tick values are provided,
                       or a single price value if only one tick value is provided.
    """
    return [1.0001 ** t for t in ticks] if len(ticks) > 1 else 1.0001 ** ticks[0]


class Bars:
    def __init__(self, ticks: List[float] = [], values: List[float] = []) -> None:
        """
        Initialize a Bars object with specified ticks and values.

        Args:
            ticks (List[float]): A list of tick values representing the boundaries of intervals.
                                Must contain more than one tick if provided.
            values (List[float]): A list of values corresponding to the intervals defined by the ticks.
                                The length of values must be one less than the length of ticks.

        Raises:
            ValueError: If a single tick is provided, or if the number of values does not match the number of intervals.
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

    def overlap(self, array: List[float], budget: float, i: int = None, j: int = None):
        """
        Calculate the overlap (TV distance) between the current Bars object and a given array over a specified budget.

        Args:
            array (list or Bars): The array or Bars object to compare against the current Bars object.
            budget (float): The budget value used for normalization in the overlap calculation.
            i (int, optional): The starting index for the overlap calculation. Defaults to the beginning of the Bars.
            j (int, optional): The ending index for the overlap calculation. Defaults to the end of the Bars.

        Returns:
            float: The calculated overlap value, normalized by the budget. Returns 0 if the overlap is close to zero.
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
        Calculate the impermanent loss for a given initial and final state of a liquidity pool.

        Args:
            q0 (float): The initial quantity or price level of the asset in the pool.
            p1 (Tuple[float, float, float]): A tuple containing the final state parameters:
                - q1 (float): The final quantity or price level of the asset in the pool.
                - px1 (float): The price of the first asset in the final state.
                - py1 (float): The price of the second asset in the final state.
            asliq (bool, optional): A flag indicating whether the calculation is performed in liquidity terms (or cash terms). Defaults to True.

        Returns:
            float: The calculated impermanent loss value, representing the loss incurred due to holding assets in a liquidity pool
                   compared to holding them separately.
        """
        assert asliq
        q1, px1, py1 = p1
        r0 = np.maximum(np.minimum(q0, self.T[1:]), self.T[:-1]) ** .5
        r1 = np.maximum(np.minimum(q1, self.T[1:]), self.T[:-1]) ** .5
        return sum(self.V * (px1 * (1/r0-1/r1) + py1 * (r0-r1)))

    def margins(self, tol=1e-5) -> Tuple[float, float]:
        """
        Determine the margin tick values of the Bars object (excluding zeros on both sides).

        Args:
            tol (float, optional): The tolerance level used to determine significant values. Defaults to 1e-5.

        Returns:
            Tuple[float, float]: A tuple containing the start and end tick values that define the margins of the Bars object.
        """
        i, j = self.margin_idxs(tol)
        return self.T[i], self.T[j]

    def margin_idxs(self, tol=1e-5) -> Tuple[int, int]:
        """
        Determine the indices of the margin ticks in the Bars object, excluding values below a specified tolerance.

        Args:
            tol (float, optional): The tolerance level used to determine significant values. Defaults to 1e-5.

        Returns:
            Tuple[int, int]: A tuple containing the start and end indices that define the margins of the Bars object.
        """
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

    def barplot(self, ax, **kwargs):
        ax.bar(self.T[:-1], self.V, width=np.array(self.T[1:])-np.array(self.T[:-1]), align='edge', **kwargs)

    def from_delta(deltas: List[tuple], sort: bool = False, assert_positive: bool = True) -> Bars:
        """
        Create a Bars object from a list of delta tuples, optionally sorting and asserting positivity.

        Args:
            deltas (List[tuple]): A list of tuples where each tuple contains a tick value and a delta value.
            sort (bool, optional): A flag indicating whether to sort the deltas by tick values. Defaults to False.
            assert_positive (bool, optional): A flag indicating whether to assert that all resulting values are non-negative. Defaults to True.

        Returns:
            Bars: A Bars object constructed from the provided deltas, with ticks and values derived from the deltas.
        """
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
        """
        Convert the Bars object into a list of delta tuples.

        Returns:
            List[tuple]: A list of tuples where each tuple contains a tick value and the change in value
                         from the previous tick. The last tuple includes the last tick and the negative
                         of the last value to ensure the sum of deltas equals zero.
        """
        prev_v, deltas = 0, []
        for i in range(len(self)):
            deltas.append((self.T[i], self.V[i] - prev_v))
            prev_v = self.V[i]

        return deltas + [(self.T[-1], -self.V[-1])]

    def resample_fee(self, ticks: list, is_token_0: bool) -> Bars:
        """
        Resample the fee distribution over a new set of ticks. 
        When interval is broken, the corresponding value is also **broken pro rata**. 
        Upon sampling [c, d] out of interval [a, b], fee percentage is proportional to 
        - If `is_token_0`, (sqrt(d) - sqrt(c)) / (sqrt(b) - sqrt(a))
        - Otherwise, (1/sqrt(c) - 1/sqrt(d)) / (1/sqrt(a) - 1/sqrt(b))

        # Parameters:
        ticks (list): A list of tick values to resample the fee distribution over.
        is_token_0 (bool): A flag indicating whether the calculation is for token 0.

        # Returns:
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
        Calculate the fees incurred over a range of ticks when transitioning from an old quantity to a new quantity.

        Args:
            q_old (float): The initial price.
            q_new (float): The final price.
            fee_rate (float): The fee rate applied to the transaction.

        Returns:
            Bars: A Bars object representing the calculated fees over the specified range of ticks.
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

    def align(self, ticks: List[float], force_bounds: bool = False) -> Bars:
        """
        Align the current Bars object to a new set of tick values, optionally enforcing boundary conditions. 
        When interval is broken, the corresponding value **does not change**. Used only on liquidity distributions. 

        Args:
            ticks (List[float]): A list of new tick values to align the Bars object to.
            force_bounds (bool, optional): A flag indicating whether to enforce that the new ticks are within the bounds
                                           of the current Bars object. Defaults to False.

        Returns:
            Bars: A new Bars object with ticks aligned to the specified list, and values adjusted accordingly.
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
    """ Converts representation of an action between cash distribution and liquidity distribution."""

    def __init__(self, q, usdx, usdy):
        """
        Initialize a LiqConverter object with specified parameters.

        Args:
            q (float): The quantity or price level used for conversion calculations.
            usdx (float): The USD value associated with X.
            usdy (float): The USD value associated with Y.
        """
        self.q = q
        self.usdx = usdx
        self.usdy = usdy

    def to_liq(self, cash: Bars):
        """
        Convert a Bars object representing cash values into liquidity values.

        Args:
            cash (Bars): A Bars object representing liquidity distribution in cash.

        Returns:
            Bars: A new Bars object with liquidity values corresponding to the price intervals.
        """
        if cash.empty():
            return Bars()

        res = []
        for v, a, b in cash.intervals():
            qm = max(a, min(b, self.q))
            res.append(v / (self.usdx * (qm**-.5 - b**-.5) + self.usdy * (qm**.5 - a**.5)))
        return Bars(cash.T, res)

    def to_cash(self, liq: Bars):
        """
        Convert a Bars object representing liquidity values into cash values.

        Args:
            liq (Bars): A Bars object representing liquidity distribution.

        Returns:
            Bars: A new Bars object with cash values corresponding to the price intervals.
        """
        if liq.empty():
            return Bars()

        res = []
        for v, a, b in liq.intervals():
            qm = max(a, min(b, self.q))
            res.append(v * (self.usdx * (qm**-.5 - b**-.5) + self.usdy * (qm**.5 - a**.5)))

        return Bars(liq.T, res)

    def overlap_general(self, budget: float, action_usd: Bars, *baselines_usd: Bars) -> Union[float, List[float]]:
        """
        Calculate the overlap between a given action in USD and one or more baseline USD distributions. 
        Their ticks may not be necessarily aligned. 

        Args:
            budget (float): The budget value used for normalization in the overlap calculation.
            action_usd (Bars): A Bars object representing the action distribution in USD.
            baselines_usd (Bars): One or more Bars objects representing baseline distributions in USD.

        Returns:
            Union[float,List[float]]: A list of overlap values for each baseline if multiple baselines are provided,
                                      or a single overlap value if only one baseline is provided.
        """
        liq, olaps = self.to_liq(action_usd), []
        for baseline in baselines_usd:
            liq_temp, baseline_liq = align_bars(liq, self.to_liq(baseline), standard=-1)
            olaps.append(self.to_cash(baseline_liq).overlap(self.to_cash(liq_temp), budget))

        return olaps if len(baselines_usd) > 1 else olaps[0]


def align_bars(*bars: Bars, default: float = 0, standard: int = -1) -> List[Bars]:
    """
    Align multiple Bars objects to a common set of tick values, optionally using a standard Bars object
    to define the range of ticks. This function ensures that all Bars objects have the same tick intervals,
    filling in missing values with a default value.

    Args:
        bars (Bars): One or more Bars objects to be aligned.
        default (float, optional): The default value to use for missing intervals. Defaults to 0.
        standard (int, optional): The index of the Bars object to use as the standard for tick range.
                                  If negative, the full range of ticks from all Bars is used. Defaults to -1.

    Returns:
        List[Bars]: A list of new Bars objects, each aligned to the common set of tick values.
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
