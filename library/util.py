from __future__ import annotations
from typing import List, Tuple, Union, Dict, Any

import numpy as np
import pandas as pd
import bisect
import matplotlib.pyplot as plt

from pandas.tseries.offsets import MonthEnd
from decimal import Decimal


def num_alias(n: int) -> str:
    """
    This function takes an integer as input and returns a string representation of the number with appropriate aliasing.
    The aliasing is based on the magnitude of the number, where numbers greater than or equal to 1 billion are represented as 'G',
    numbers greater than or equal to 1 million are represented as 'M', and numbers greater than or equal to 1 thousand are represented as 'K'.

    Args:
        n (int): The input number to be aliased.

    Returns:
        str: The aliased string representation of the input number.
    """
    r = abs(n)
    if r >= 1e9:
        x = f'{int(r//1e9)}G'
    elif r >= 1e6:
        x = f'{int(r//1e6)}M'
    elif r >= 1e3:
        x = f'{int(r//1e3)}K'
    else:
        x = f'{int(r)}'
    return x if n >= 0 else f'-{x}'


def frexp(number):
    """
    Decompose a number into its mantissa and exponent.

    This function takes a number and decomposes it into a normalized fraction (mantissa)
    and an exponent, such that the number is represented as mantissa * (10 ** exponent).

    Args:
        number: The number to be decomposed. Can be an integer or a float.

    Returns:
        Tuple[Decimal, int]: A tuple containing the mantissa as a Decimal and the exponent as an integer.
    """
    (_, digits, exponent) = Decimal(number).as_tuple()
    return Decimal(number).scaleb(-(len(digits) + exponent - 1)).normalize(), len(digits) + exponent - 1


def color_cycle(*colors):
    if len(colors) == 0:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    i = 0
    while True:
        yield colors[i]
        i = (i+1) % len(colors)


def linestyle_cycle(styles=['-', '--', '-.', ':']):
    """
    Generate an infinite cycle of line styles for plotting.

    This function creates a generator that yields line styles in a cyclic manner.
    It is useful for iterating over a predefined set of line styles when plotting
    multiple lines on a graph.

    Args:
        styles (list of str): A list of line style strings to cycle through. 
                          Default is ['-', '--', '-.', ':'].

    Yields:
        str: The next line style in the cycle.
    """
    i = 0
    while True:
        yield styles[i]
        i = (i+1) % len(styles)


def marker_cycle(markers=['x', 'o', '^', 's', '*']):
    """
    Generate an infinite cycle of marker styles for plotting.

    This function creates a generator that yields marker styles in a cyclic manner.
    It is useful for iterating over a predefined set of marker styles when plotting
    multiple points or lines on a graph.

    Args:
        markers (list of str): A list of marker style strings to cycle through.
                               Default is ['x', 'o', '^', 's', '*'].

    Yields:
        str: The next marker style in the cycle.
    """
    i = 0
    while True:
        yield markers[i]
        i = (i+1) % len(markers)


def fmt(num, digits=10):
    """
    Format a number to a specified width with appropriate precision.

    This function formats a given number to a specified number of digits.
    If the number is a float, it formats it with one decimal place.
    If the number is an integer, it formats it as a whole number.
    If the formatted number exceeds the specified width, it uses scientific notation.

    Args:
        num: The number to be formatted. Can be an integer or a float.
        digits: The total width of the formatted string. Default is 10.

    Returns:
        str: The formatted string representation of the number.
    """
    f = f'{{:^{digits}.1f}}'.format(num) if isinstance(num, float) else f'{{:^{digits}d}}'.format(int(num))
    if len(f) <= 10:
        return f
    return f'{{:^{digits}.1e}}'.format(num)


def t2p(*ticks: int) -> Union[List[float], float]:
    """
    Convert tick values to price values using a specific exponential formula.

    This function takes one or more tick values and converts all of them to price values
    using the Uniswap v3 formula. 
    If multiple tick values are provided, it returns a list of converted price values. If a
    single tick value is provided, it returns the converted price value directly.

    Args:
        *ticks (int): Variable length argument list of tick values to be converted.

    Returns:
        Union[List[float],float]: A list of converted price values if multiple tick values are provided, 
                                  or a single converted price value if only one tick value is provided.
    """
    return [1.0001 ** t for t in ticks] if len(ticks) > 1 else 1.0001 ** ticks[0]


def bend_price(ux, uy, q) -> Tuple[float, float]:
    """
    Calculate the adjusted prices based on the given parameters.

    This function computes two adjusted prices using the geometric mean of
    ux and uy, and a factor q. It returns a tuple of these two prices.

    Args:
        ux (float): The first price component.
        uy (float): The second price component.
        q (float): The adjustment factor.

    Returns:
        Tuple[float,float]: A tuple containing two adjusted prices. 
                            The first element is the price adjusted by the square root of q, 
                            and the second element is the price adjusted by the inverse square root of q.
    """
    mid = (ux * uy)**.5
    return mid * (q**.5), mid * (q**-.5)


def pseudo_index(r, bins):
    k = bisect.bisect_left(bins, r)
    return k + (r - bins[k]) / (bins[k] - bins[k-1]) if k > 0 else 0


def remap_ticks(tick_new, tick_old, tick_price):
    offset = int(np.log10(tick_price[0] / 1.0001 ** tick_old[0]))
    return np.power(1.0001, tick_new) * (10 ** offset)


def price_est_pdf(px, py, fluc_x, fluc_y, samples=100):
    ix, iy = [np.linspace(-np.log(1+fluc), np.log(1+fluc), samples) for fluc in [fluc_x, fluc_y]]
    px, py = np.meshgrid(px * np.exp(ix), py * np.exp(iy))
    px, py = px.reshape((-1, )), py.reshape((-1, ))
    return np.array([(x/y, x, y) for x, y in zip(px, py)]), np.ones((samples * samples, )) / (samples**2)


class DateIter:
    """A periodic timer. 
    """

    def __init__(self, start, delta: str):
        self.time = pd.Timestamp(f'{pd.Timestamp(start).strftime("%Y-%m-%d")} 23:59:59')
        self.delta = delta
        if delta.endswith('w'):
            self.delta = f'{7*int(delta[:-1])}d'
        self.proceed(self.time)

    def ddl(self):
        return self.time

    def ddl_str(self):
        return self.time.strftime("%y-%m-%d")

    def proceed(self, datetime=None):
        """
        Advance the current time by the specified delta or to a given datetime.

        This method updates the internal time of the DateIter instance. If a datetime
        is provided, it adjusts the time to the end of the month if the delta is in months.
        Otherwise, it advances the time by the specified delta.

        Args:
            datetime (optional): A specific datetime to advance to. If None, the time is
                            advanced by the delta.

        Returns:
            None
        """
        if datetime is None:
            datetime = self.time + pd.Timedelta('1min')
        else:
            if self.delta.endswith('m'):
                self.time += MonthEnd(1)
            return

        if self.delta.endswith('m'):
            months = int(self.delta[:-1])
            self.time += MonthEnd(months)
        else:
            self.time += pd.Timedelta(self.delta)


def weighted_average(lists: List[Tuple[float, float]]) -> np.ndarray:
    """
    Calculate the weighted average of multiple lists of values and weights.

    This function takes a list of tuples, where each tuple contains a list of values and a list of corresponding weights.
    It calculates the weighted average for each pair of values and weights, and returns an array of these weighted averages.

    Args:
        lists (List[Tuple[float, float]]): A list of tuples, where each tuple contains a list of values and a list of weights.
            The first element of the tuple is a list of values, and the second element is a list of weights.
            The lengths of the value and weight lists must be the same.

    Returns:
        np.ndarray: An array of weighted averages, where each element corresponds to the weighted average of the values and weights
            in the corresponding tuple in the input list.
    """
    arr = []
    for l_ in lists:
        value, weight = np.transpose(l_)
        arr.append(sum(value * weight) / sum(weight))

    return np.array(arr)


def filter_budget(budget_dict: Dict[str, float], q: float, max_players: int) -> Tuple[List[str], List[str], List[float], float]:
    """
    Filter and partition a budget dictionary based on a threshold and maximum number of players.

    This function sorts the budget dictionary by values in descending order and partitions
    the keys into two lists: one that meets the threshold condition and another that does not.
    It also calculates the cumulative budget percentage of the selected keys.

    Args:
        budget_dict (Dict[str, float]): A map of lp_hash -> lp's budget.
        q (float): A threshold ratio used to determine the cumulative budget limit.
        max_players (int): The maximum number of players to include in the selected list.

    Returns:
        Tuple[List[str],List[str],List[float],float]: A tuple containing:
            - A list of selected player identifiers.
            - A list of remaining player identifiers.
            - A list of budgets corresponding to the selected players.
            - The cumulative budget percentage of the selected players.
    """
    if len(budget_dict) == 0:
        return [], [], [], 0.

    keys, B = zip(*sorted(budget_dict.items(), key=lambda x: -x[1]))
    curr, thr = 0, sum(B) * q
    for i, b in enumerate(B):
        curr += b
        if curr > thr or i >= max_players-1:
            return keys[:i+1], keys[i+1:], B[:i+1], curr / sum(B)
    return list(keys), [], list(B), curr / sum(B)


def check_int(s: Any) -> int:
    """
    Convert a given input to an integer if possible.

    This function attempts to convert the input to an integer. If the conversion
    fails, it returns 0.

    Args:
        s (Any): The input value to be converted to an integer.

    Returns:
        int: The integer representation of the input if conversion is successful,
            otherwise 0.
    """
    try:
        return int(s)
    except:
        return 0
