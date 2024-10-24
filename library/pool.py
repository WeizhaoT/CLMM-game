from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import gc
import json
from sortedcontainers import SortedDict, SortedList
from typing import List, Dict, Tuple

from tqdm import tqdm
from library import align_bars, Bars, t2p, fmt, bend_price
from library.plotter import *
from matplotlib import gridspec

matplotlib.use('Agg')


def convert_price(sqrtPX96, digit=0):
    """
    Converts a square root price to a regular price with the specified number of decimal places.

    Parameters:
    sqrtPX96 (int, float): The square root of the price, where P is the actual price.
    digit (int, optional): The number of decimal places in the resulting price. Default is 0.

    Returns:
    float: The converted price with the specified number of decimal places.
    """
    return ((int(sqrtPX96) / 2.0**96) ** 2) * (10 ** digit)


def tokens_from_liquidity(liq: float, price: float, ltick: int, utick: int, FX: float = 1., FY: float = 1.):
    """
    Calculate the amount of Token 0 and Token 1 that corresponds to a given liquidity in a given price range.

    Parameters:
    liq (float): The liquidity amount.
    price (float): The current price of Token 0 in terms of Token 1.
    ltick (int): The lower tick of the price range.
    utick (int): The upper tick of the price range.
    FX (float, optional): Offset the amount by the number of decimals of Token 0. Default is 1.
    FY (float, optional): Offset the amount by the number of decimals of Token 1. Default is 1.

    Returns:
    tuple: A tuple containing the amount of Token 0 and Token 1, respectively.

    The function calculates the amount of Token 0 and Token 1 that corresponds to a given liquidity in a given price range.
    It takes into account the exchange rates from Token 0 and Token 1 to USD.
    """
    lprice, uprice = t2p(ltick, utick)
    mprice = max(lprice, min(uprice, price))
    return liq * (mprice**-.5 - uprice**-.5) * FX, liq * (mprice**.5 - lprice**.5) * FY


class Fee2D:
    def __init__(self, x: int = 0, y: int = 0) -> None:
        self.x, self.y = x, y

    def __str__(self):
        return f'X={self.x}, Y={self.y}'

    def __add__(self, fee, token_type_eq_0=True):
        if isinstance(fee, Fee2D):
            return Fee2D(self.x + fee.x, self.y + fee.y)
        else:
            return Fee2D(self.x + fee, self.y) if token_type_eq_0 else Fee2D(self.x, self.y + fee)

    def zero(self) -> bool:
        return self.x == 0 and self.y == 0

    def split(self, *ticks: int):
        """
        Splits the current Fee2D object into multiple Fee2D objects based on the provided tick boundaries.

        Parameters:
        *ticks (int): A variable number of tick boundaries that define the intervals for splitting.
                    There must be at least two tick values, and they must be in strictly increasing order.

        Returns:
        List[Fee2D]: A list of Fee2D objects, each representing the portion of the original Fee2D
                    object within the corresponding interval defined by the tick boundaries.
        """
        assert len(ticks) >= 2
        for i in range(len(ticks) - 1):
            if ticks[i+1] <= ticks[i]:
                print(ticks)
                raise

        P = t2p(*ticks)
        return [Fee2D(x=self.x * (P[i]**-.5 - P[i+1]**-.5) / (P[0]**-.5 - P[-1]**-.5),
                      y=self.y * (P[i+1]**0.5 - P[i]**0.5) / (P[-1]**0.5 - P[0]**0.5))
                for i in range(len(ticks) - 1)]


class Histogram2D:
    def __init__(self, ticks: List[int] = [], tokens: List[Fee2D] = []) -> None:
        self.dtype = Fee2D
        self.T = []
        self.tokens: Dict[int, Fee2D] = {}
        if len(tokens) > 0 and len(ticks) == len(tokens) + 1:
            self.T = ticks
            self.tokens = {t: fee for t, fee in zip(ticks[1:], tokens)}

    def __str__(self) -> str:
        s = Fee2D()
        for t in self.T[1:]:
            s += self.tokens[t]

        tl, tr = 0, len(self.T) - 1
        while tl < tr and self.tokens[self.T[tl+1]].zero():
            tl += 1
        while tl < tr and self.tokens[self.T[tr]].zero():
            tr -= 1

        if tl == tr:
            return '<all zero histogram>'

        ticks = self.T[tl:tr+1]
        row = 'Ticks |' + '  '.join(fmt(t, 8) for t in ticks) + f'|{"sum":^9s}\n'
        row += '------|' + '-' * (8 + 10*(len(ticks)-1)) + '|' + '-'*9 + '\n'
        row += 'X     |     ' + '  '.join([fmt(self.tokens[t].x, 8) for t in ticks[1:]]) + f'     |{fmt(s.x, 9)}\n'
        row += 'Y     |     ' + '  '.join([fmt(self.tokens[t].y, 8) for t in ticks[1:]]) + f'     |{fmt(s.y, 9)}\n'
        return row

    def __getitem__(self, i):
        return self.tokens[self.T[i+1]]

    def __len__(self):
        return max(len(self.T) - 1, 0)

    def empty(self):
        return len(self.T) == 0

    def a(self, i):
        return self.T[i]

    def b(self, i):
        return self.T[i+1]

    def merge(self, fees: Bars, token_type_eq_0: bool):
        """
        Merges the current histogram with another set of fees, updating the internal state.

        Parameters:
        fees (Bars): A Bars object representing the fees to be merged with the current histogram.
        token_type_eq_0 (bool): A boolean indicating whether the token type is equal to 0. 
                                If True, the x component of the Fee2D is updated; otherwise, the y component is updated.

        Returns:
        None: This function updates the internal state of the histogram and does not return a value.
        """
        i = j = 0
        self.T, old_ticks = [], self.T
        self_pizza, fee_pizza = Fee2D(), Fee2D()
        self_range, fee_range = None, None

        while i < len(old_ticks) or j < len(fees.T):
            i0, j0 = i, j
            if j == len(fees.T) or (i < len(old_ticks) and old_ticks[i] < fees.T[j]):
                if len(self.T) > 0:
                    if j < len(fees.T) and fee_range is not None:
                        fee_cut, fee_pizza = fee_pizza.split(fee_range[0], old_ticks[i], fee_range[1])
                        fee_range = (old_ticks[i], fee_range[1])
                    else:
                        fee_cut = Fee2D()
                    self.tokens[old_ticks[i]] = self_pizza + fee_cut

                self.T.append(old_ticks[i])
                i += 1
            elif i == len(old_ticks) or (j < len(fees.T) and old_ticks[i] > fees.T[j]):
                if len(self.T) > 0:
                    if i < len(old_ticks) and self_range is not None:
                        self_cut, self_pizza = self_pizza.split(self_range[0], fees.T[j], self_range[1])
                        self_range = (fees.T[j], self_range[1])
                    else:
                        self_cut = Fee2D()
                    self.tokens[fees.T[j]] = self_cut + fee_pizza

                self.T.append(fees.T[j])
                j += 1
            else:
                self.T.append(old_ticks[i])
                self.tokens[old_ticks[i]] = self_pizza + fee_pizza
                i += 1
                j += 1
            if i0 < i and i < len(old_ticks):
                self_range, self_pizza = (old_ticks[i-1], old_ticks[i]), self.tokens[old_ticks[i]]
            if j0 < j and j < len(fees.T):
                fee_range = (fees.T[j-1], fees.T[j])
                fee_pizza = Fee2D(x=fees[j-1]) if token_type_eq_0 else Fee2D(y=fees[j-1])

    def to_list(self):
        return list(self.tokens.items())

    def clear(self):
        self.tokens.clear()
        self.T.clear()

    def fix(self, hist: Histogram2D):
        """
        Adjusts the current histogram to match the structure of another histogram.

        This function modifies the current histogram's tokens to align with the intervals
        defined by the provided histogram. It ensures that the tokens in the current histogram
        are split or assigned based on the intervals and values in the provided histogram.

        Parameters:
        hist (Histogram2D): The histogram to align with. The current histogram will be adjusted
                            to match the intervals and values of this histogram.

        Returns:
        None: This function updates the internal state of the current histogram and does not return a value.
        """
        i = j = 0
        while i < len(self.T) and j < len(hist.T):
            i0 = i
            while i < len(self.T) and self.T[i] < hist.T[j]:
                i += 1
            if j == 0:
                if i0 < i:
                    for k in range(i0+1, i+1):
                        self.tokens[self.T[k]] = Fee2D()
            else:
                if i0 == i-1:
                    self.tokens[self.T[i]] = hist.tokens[hist.T[j]]
                else:
                    parts = hist.tokens[hist.T[j]].split(*self.T[i0:i+1])
                    for k in range(i0+1, i+1):
                        self.tokens[self.T[k]] = parts[k-i0-1]
            j += 1
        if i < len(self.T) - 1:
            for k in range(i+1, len(self.T)):
                self.tokens[self.T[k]] = Fee2D()

    def FeeX(self, mult=1.):
        return [self.tokens[t].x * mult for t in self.T[1:]]

    def FeeY(self, mult=1.):
        return [self.tokens[t].y * mult for t in self.T[1:]]

    def evaluate(self, price0, price1) -> Bars:
        if len(self.tokens) == 0:
            return Bars()

        return Bars(self.T, [self.tokens[t].x*price0 + self.tokens[t].y*price1 for t in self.T[1:]])


class Position:
    def __init__(self, token_id, owner, ltick, utick, liq, price, birth=None) -> None:
        self.id = token_id
        self.owner = owner
        self.tl, self.tu = ltick, utick

        self.liq = liq
        self.init_price = price
        self.birth = birth
        self.fee0 = self.fee1 = self.marginal0 = self.marginal1 = 0

    def clear_fee(self):
        self.fee0 = self.fee1 = self.marginal0 = self.marginal1 = 0

    def amounts(self, price):
        return tokens_from_liquidity(self.liq, price, self.tl, self.tu)


class LiqTable:
    def __init__(self):
        self.ticks: SortedList = SortedList()
        self.TAB = np.array([])
        self.vacants = set()
        self.rid_to_pid = []
        self.pid_to_rid: Dict[int, int] = {}

    def __str__(self):
        if len(self.ticks) == 0:
            return '<empty table>\n'

        rows = [f'{"":8}' + f'{"":4}'.join([f'{t:^8d}' for t in self.ticks])]
        for row, p in enumerate(self.rid_to_pid):
            if p >= 0:
                rows.append(f'{p:<8d}{"":5}' + f'{"":2}'.join([fmt(num) for num in self.TAB[row]]))

        return '\n'.join(rows) + '\n'

    def clear(self):
        self.ticks.clear()
        self.TAB = np.array([])
        self.vacants.clear()
        self.rid_to_pid.clear()
        self.pid_to_rid.clear()

    def update_liquidity(self, pid, tl, tu, L):
        """
        Updates the liquidity for a given position identified by pid.

        Parameters:
        pid (int): The position identifier.
        tl (int): The lower tick boundary for the liquidity range.
        tu (int): The upper tick boundary for the liquidity range.
        L (float): The amount of liquidity to be set. If L is zero, the position is removed.

        Returns:
        None: This function updates the internal state of the liquidity table and does not return a value.
        """
        if L == 0:
            if pid in self.pid_to_rid:
                self._pop_pos(pid)
            return

        if len(self.ticks) == 0:
            self.ticks.update([tl, tu])
            self.TAB = np.array([[L]])
            self._add_if_absent(pid)
            return

        rid = self._add_if_absent(pid)
        jl, ju = self.align([tl, tu], add_row=rid >= len(self.TAB))
        self.TAB[rid, jl:ju] = float(L)

    def align(self, ticks, add_row=False):
        """
        Aligns the internal liquidity table with the specified ticks, optionally adding a new row.

        This function adjusts the internal state of the liquidity table to ensure that it includes
        the specified ticks. If any of the specified ticks are not present in the current table,
        they are added, and the table is expanded accordingly. Optionally, a new row can be added
        to the table.

        Parameters:
        ticks (list): A list of tick values to align with the internal liquidity table.
        add_row (bool): A boolean flag indicating whether to add a new row to the table. 
                        Defaults to False.

        Returns:
        list: A list of indices corresponding to the positions of the specified ticks in the 
              aligned liquidity table.
        """
        n, m = np.shape(self.TAB)
        idx = [self.ticks.bisect_left(t) for t in ticks]
        flags = [j > m or self.ticks[j] != t for j, t in zip(idx, ticks)]

        dm = 0
        for t, flag in zip(ticks, flags):
            dm += flag
            if flag:
                self.ticks.add(t)

        if dm == 0:
            if add_row:
                self.TAB, temp = np.zeros((n+1, m)), self.TAB
                self.TAB[:n] = temp
            return idx

        self.TAB, past_table = np.zeros((n+add_row, m+dm)), self.TAB

        rs, p, behind = [], 0, 0
        for flag, j in zip(flags, idx):
            if j > m:
                rs.append((p, j-1+behind, False))
                p = m+dm
                break
            elif flag:
                rs.append((p, j+behind, True))
                behind += 1
                p = j+behind

        behind = 0
        for i, flag in enumerate(flags):
            idx[i] += behind
            behind += flag

        if p < m+dm:
            rs.append((p, m+dm, False))

        p = 0
        for a, b, f in rs:
            if b > a:
                self.TAB[:n, a:b] = past_table[:, p:p+b-a]
                p += b-a
            if f:
                self.TAB[:n, b] = self.TAB[:n, b-1]

        return idx

    def get_fees(self, total_liq: Bars, q_old: float, q_new: float, fee_rate: float) -> Tuple[dict, Bars]:
        """
        Calculate the fees associated with a change in liquidity between two quantities.

        This function computes the fees incurred when liquidity changes from `q_old` to `q_new`
        over a range of ticks. It returns a dictionary mapping position identifiers to their
        respective fees and a Bars object representing the distribution of fees across the ticks.

        Parameters:
        total_liq (Bars): A Bars object representing the total liquidity across ticks.
        q_old (float): The initial quantity of liquidity.
        q_new (float): The new quantity of liquidity after the change.
        fee_rate (float): The rate at which fees are charged.

        Returns:
        Tuple[dict, Bars]: A tuple containing:
            - A dictionary mapping position identifiers (pid) to their respective fees.
            - A Bars object representing the distribution of fees across the ticks.
        """
        t_old, t_new = [np.log(q) / np.log(1.0001) for q in [q_old, q_new]]
        if q_old < q_new:
            if t_new <= self.ticks[0] or t_old >= self.ticks[-1]:
                return {pid: 0. for pid in self.pid_to_rid}, Bars()
            j_old, j_new = self.ticks.bisect_right(t_old)-1, self.ticks.bisect_left(t_new)
            trunc_ticks = list(self.ticks[j_old:j_new+1])
            share = self.TAB[:, j_old:j_new]

            fees, share_norm = np.zeros((len(trunc_ticks)-1, )), share.sum(0)
            nft_liq, _ = align_bars(Bars(trunc_ticks, share_norm), total_liq, standard=0)
            j = 0
            for vn, a, b in (nft_liq.intervals()):
                assert j+1 < len(trunc_ticks)
                if b > trunc_ticks[j+1]:
                    j += 1

                qa, qb = t2p(a, b)
                if q_old < qb and q_new > qa:
                    qa, qb = max(q_old, qa), min(q_new, qb)
                    fees[j] += vn * (qb**.5 - qa**.5)

            fees *= fee_rate / (1-fee_rate)
            fees_pos = np.sum(share * (fees / share_norm), 1)
            return {pid: fees_pos[rid] for pid, rid in self.pid_to_rid.items()}, Bars(trunc_ticks, fees)
        else:
            if t_old <= self.ticks[0] or t_new >= self.ticks[-1]:
                return {pid: 0. for pid in self.pid_to_rid}, Bars()
            j_new, j_old = self.ticks.bisect_right(t_new) - 1, self.ticks.bisect_left(t_old)
            trunc_ticks = list(self.ticks[j_new:j_old+1])
            share = self.TAB[:, j_new:j_old]
            fees, share_norm = np.zeros((len(trunc_ticks)-1, )), share.sum(0)
            nft_liq, _ = align_bars(Bars(trunc_ticks, share_norm), total_liq, standard=0)
            j = 0
            for vn, a, b in (nft_liq.intervals()):
                assert j+1 < len(trunc_ticks)
                if b > trunc_ticks[j+1]:
                    j += 1

                qa, qb = t2p(a, b)
                if q_new < qb and q_old > qa:
                    qa, qb = max(q_new, qa), min(q_old, qb)
                    fees[j] += vn * (qa**-.5 - qb**-.5)

            fees *= fee_rate / (1-fee_rate)
            fees_pos = np.sum(share * (fees / share_norm), 1)
            return {pid: fees_pos[rid] for pid, rid in self.pid_to_rid.items()}, Bars(trunc_ticks, fees)

    def _add_if_absent(self, pid) -> int:
        """
        Adds a position identifier to the internal mapping if it is not already present.

        This function checks if a given position identifier (pid) is already present in the
        internal mapping. If it is not present, it assigns a new row identifier (rid) to it.
        If there are any vacated row identifiers available, it reuses one of them; otherwise,
        it creates a new row identifier.

        Parameters:
        pid (int): The position identifier to be added or checked.

        Returns:
        int: The row identifier (rid) associated with the given position identifier.
        """
        if pid in self.pid_to_rid:
            return self.pid_to_rid[pid]
        elif len(self.vacants) > 0:
            self.pid_to_rid[pid] = rid = self.vacants.pop()
            self.rid_to_pid[rid] = pid
        else:
            self.pid_to_rid[pid] = rid = len(self.rid_to_pid)
            self.rid_to_pid.append(pid)

        return rid

    def _pop_pos(self, pid):
        rid = self.pid_to_rid.pop(pid)
        self.rid_to_pid[rid] = -1
        self.TAB[rid] = 0.
        self.vacants.add(rid)


class Pool:
    LOGBASE = np.log(1.0001)

    def __init__(self, price, usd_price0, usd_price1, time, fee=0, decimal0=0, decimal1=0):
        self.fee_rate = fee
        self.FX, self.FY = 10**(-decimal0), 10**(-decimal1),
        self.FP, self.FL = 10**(decimal0-decimal1), 10**(-(decimal0+decimal1)/2)
        self.price = price      # -dy/dx <==> How much Token 1 is 1 unit of Token 0 worth?
        self.usd0, self.usd1 = usd_price0, usd_price1

        self.flag_settled = True
        self.flag_caught_up = False
        self.deltas = []
        self.liqbars = Bars()
        self.oor0, self.oor1 = [], []  # decr/incr by tick
        self.NFT: Dict[int, Position] = {}
        self.player_pos: Dict[int, int] = {}
        self.FT_deltas = SortedDict()

        self.LTAB = LiqTable()

        self.fee_hist_total = Histogram2D()
        self.fee_hist_nft = Histogram2D()

        self.first_register = True
        self.last_price = price
        self.last_usd0, self.last_usd1 = usd_price0, usd_price1
        self.removed_liq = []
        self.base_time = time
        self.price_change = {}

    def _get_amounts(self, price=None):
        if price is None:
            price = self.price

        x, y = list(map(sum, zip(*[p.amounts(price) for p in self.NFT.values()])))
        prev = liq = 0
        for i, (tick, val) in enumerate(self.FT_deltas.items()):
            if val == 0:
                continue
            if i > 0:
                dx, dy = tokens_from_liquidity(liq, price, prev, tick)
                x, y = x+dx, y+dy

            liq, prev = liq+val, tick

        return x, y

    def validate_ft(self):
        liq = prev = 0
        for i, (tick, val) in enumerate(self.FT_deltas.items()):
            if i > 0 and prev == tick:
                return False
            liq, prev = liq+val, tick
            if liq < 0:
                return False
        return True

    def settle(self):
        if not self.flag_settled:
            deltas = sorted([(p.tl, p.liq) for p in self.NFT.values()] +
                            [(p.tu, -p.liq) for p in self.NFT.values()] +
                            [(k, v) for k, v in self.FT_deltas.items() if v != 0])
            self.liqbars = Bars.from_delta(deltas, sort=False, assert_positive=True)
            self.LTAB.align(self.liqbars.T)
            self.liqbars = self.liqbars.align(self.LTAB.ticks)

    def update_position(self, token_id, liquidity, tl, tu, lp=None, timestamp=None):
        """
        Update the position of a token in the pool.

        This function updates the liquidity position of a given token within specified tick
        boundaries. It adjusts the liquidity and records any changes in the position.

        Parameters:
        token_id (int): The identifier of the token whose position is being updated.
        liquidity (float): The amount of liquidity to be added or removed.
        tl (int): The lower tick boundary for the liquidity position.
        tu (int): The upper tick boundary for the liquidity position.
        lp (int, optional): The liquidity provider's identifier. Defaults to the token_id if not provided.
        timestamp (optional): The timestamp of the update. Used for tracking the timing of liquidity changes.

        Returns:
        Tuple[float, float]: The amounts of tokens x and y resulting from the liquidity update.
        """
        if lp is None:
            lp = token_id

        x, y = tokens_from_liquidity(liquidity, self.price, tl, tu)
        if token_id > 0:
            if token_id not in self.NFT:
                self.NFT[token_id] = Position(token_id, lp, tl, tu, liquidity, self.price, timestamp)
            else:
                self.NFT[token_id].liq += liquidity
                base = max(self.base_time, self.NFT[token_id].birth)
                if liquidity < 0:
                    self.removed_liq.append((tl, tu, -liquidity, -(x + y/self.price) * self.FX, timestamp - base))

            if self.NFT[token_id].liq < 0:
                raise ValueError(f'Position {token_id} has negative liquidity {self.NFT[token_id].liq}')
            if self.NFT[token_id].liq == 0:
                self.NFT.pop(token_id)
        else:
            self.FT_deltas[tl] = self.FT_deltas.get(tl, 0) + liquidity
            self.FT_deltas[tu] = self.FT_deltas.get(tu, 0) + liquidity
            if liquidity < 0:
                self.removed_liq.append((tl, tu, -liquidity, -(x + y/self.price), pd.Timedelta(days=10000)))

            # assert self.validate_ft()

        self.flag_settled = False
        return x, y

    def swap(self, price_new, usd0, usd1, compute_fee=True, timestamp=None):
        """
        Perform a swap operation in the pool, updating the price and calculating fees if required.

        This function updates the pool's price to a new value, calculates the change in token amounts,
        and optionally computes and distributes fees based on the price change.

        Parameters:
        price_new (float): The new price after the swap.
        usd0 (float): The updated USD price of token 0.
        usd1 (float): The updated USD price of token 1.
        compute_fee (bool, optional): Flag indicating whether to compute and apply fees. Defaults to True.
        timestamp (optional): The timestamp of the swap operation, used for tracking price changes.

        Returns:
        Tuple[float, float]: The change in amounts of tokens x and y resulting from the swap.
        """
        self.price, price_prev = price_new, self.price
        x0, y0 = self._get_amounts(price_prev)
        x1, y1 = self._get_amounts(price_new)

        self.usd0, self.usd1 = usd0, usd1

        self.price_change[timestamp] = 1/(price_new * self.FP)

        if compute_fee:
            self.settle()
            fees, nft_fee_bars = self.LTAB.get_fees(self.liqbars, price_prev, price_new, self.fee_rate)
            for pid, fee in fees.items():
                if price_prev < price_new:
                    self.NFT[pid].fee1 += fee
                else:
                    self.NFT[pid].fee0 += fee

            total_fee_bars: Bars = self.liqbars.calculate_fee(price_prev, price_new, self.fee_rate)
            self.fee_hist_total.merge(total_fee_bars, token_type_eq_0=price_new < price_prev)
            self.fee_hist_nft.merge(nft_fee_bars, token_type_eq_0=price_new < price_prev)

        return x1-x0, y1-y0

    def reset_profit(self, time):
        self.last_price = self.price
        self.last_usd0, self.last_usd1 = self.usd0, self.usd1
        self.price_change = {}
        self.removed_liq = []
        self.base_time = time
        self.fee_hist_total.clear()
        self.fee_hist_nft.clear()
        for pos in self.NFT.values():
            pos.clear_fee()

    def register_player_pos(self, updates: dict, last_ending: dict, bar: tqdm = None):
        """
        Register and update player positions in the pool.

        This function updates the liquidity positions of players based on the provided updates
        and last ending positions. It also optionally updates a progress bar to reflect the
        progress of the registration process.

        Parameters:
        updates (dict): A dictionary containing the latest updates to player positions, where
                        keys are position identifiers and values are the updated liquidity amounts.
        last_ending (dict): A dictionary containing the last known ending positions of players,
                            where keys are position identifiers and values are the liquidity amounts.
        bar (tqdm, optional): An optional tqdm progress bar object to visually track the progress
                              of the registration process. Defaults to None.

        Returns:
        None
        """
        all_updates = {}
        if self.first_register:
            self.first_register = False
            self.player_pos = {posid: obj.liq for posid, obj in self.NFT.items()}
            all_updates = {posid: obj.liq for posid, obj in self.NFT.items()}

        all_updates |= (last_ending | updates)
        if bar is not None:
            bar.reset(total=len(all_updates))
        for posid, liq in all_updates.items():
            if liq > 0:
                self.player_pos[posid] = liq
            elif posid in self.player_pos:
                self.player_pos.pop(posid)
            obj = self.NFT[posid]
            self.LTAB.update_liquidity(posid, obj.tl, obj.tu, liq)
            if bar is not None:
                bar.update()

    def report_profit(self, time, path=None, datetime=None):
        df, pos_deltas, bmap = [], [], {}

        ux_prev, uy_prev = bend_price(self.last_usd0, self.last_usd1, self.last_price * self.FP)
        ux, uy = bend_price(self.usd0, self.usd1, self.price * self.FP)

        for pid, liq in self.player_pos.items():
            P = self.NFT[pid]
            x_before, y_before = tokens_from_liquidity(liq, self.last_price, P.tl, P.tu, self.FX, self.FY)
            x_after, y_after = tokens_from_liquidity(liq, self.price, P.tl, P.tu, self.FX, self.FY)

            v_init = x_before * ux_prev + y_before * uy_prev
            v_cost = (x_before - x_after) * ux + (y_before - y_after) * uy

            total_fee = P.fee0 * self.FX * ux + P.fee1 * self.FY * uy
            utility = total_fee - v_cost
            fee_grad = P.marginal0 + P.marginal1
            il_grad = v_cost / v_init
            total_grad = fee_grad - il_grad

            df.append([P.owner, P.tl, P.tu, liq * self.FL, t2p(P.tl) * self.FP, t2p(P.tu) * self.FP, v_init,
                       v_cost, total_fee, utility, total_grad, fee_grad, il_grad])

            pos_deltas.extend([(P.tl, liq), (P.tu, -liq)])
            bmap[P.owner] = bmap.get(P.owner, 0.) + v_init

        if datetime is not None:
            df = pd.DataFrame(df, columns=['LP', 'tickLower', 'tickUpper', 'liquidity', 'priceLow', 'priceHigh',
                                           'initialFund', 'IL', 'fee', 'utility', 'totalGrad', 'feeGrad', 'ILGrad'])
            df.sort_values('initialFund', ascending=False).to_csv(f'{path}/profit-{datetime}.csv', index=False)

            snap: Bars = Bars.from_delta(pos_deltas, sort=True)
            fee_ha, fee_hp, snap = align_bars(self.fee_hist_total, self.fee_hist_nft, snap, default=0., standard=-1)
            assert isinstance(fee_ha, Histogram2D) and isinstance(fee_hp, Histogram2D)
            fee_ha.fix(self.fee_hist_total)
            fee_hp.fix(self.fee_hist_nft)

            fee_a = fee_ha.evaluate(ux * self.FX, uy * self.FY)
            fee_p = fee_hp.evaluate(ux * self.FX, uy * self.FY)

            fee_a_raw = fee_ha.evaluate(self.usd0 * self.FX, self.usd1 * self.FY)
            fee_p_raw = fee_hp.evaluate(self.usd0 * self.FX, self.usd1 * self.FY)

            jit, jit_raw = [], []
            for i, (L, a, b) in enumerate(snap.intervals()):
                if fee_p[i] == 0:
                    jit.append(1e10)
                    jit_raw.append(1e10)
                elif fee_a[i] < fee_p[i]:
                    print(datetime)
                    print(fee_ha)
                    print(fee_hp)
                    raise ValueError
                elif np.isclose(1, fee_p[i]/fee_a[i], rtol=1e-6):
                    jit.append(0.)
                    jit_raw.append(0.)
                else:
                    x, y = tokens_from_liquidity(L, self.last_price, a, b)
                    jit.append((x * ux_prev * self.FX + y * uy_prev * self.FY) * (fee_a[i]/fee_p[i] - 1))
                    jit_raw.append((x * self.last_usd0 * self.FX + y * self.last_usd1 * self.FY)
                                   * (fee_a_raw[i]/fee_p_raw[i] - 1))

            data = {
                'q': self.last_price * self.FP,
                'qnew': self.price * self.FP,
                'usd0': [ux_prev, ux],
                'usd1': [uy_prev, uy],
                'usd0_raw': [self.last_usd0, self.usd0],
                'usd1_raw': [self.last_usd1, self.usd1],
                'tick': [t2p(t) * self.FP for t in fee_a.T],
                'tick_orig': list(fee_ha.T),
                'total_fee': list(fee_a),
                'player_fee': list(fee_p),
                'total_fee_raw': list(fee_a_raw),
                'player_fee_raw': list(fee_p_raw),
                'total_fee_x': fee_ha.FeeX(self.FX),
                'total_fee_y': fee_ha.FeeY(self.FY),
                'player_fee_x': fee_hp.FeeX(self.FX),
                'player_fee_y': fee_hp.FeeY(self.FY),
                'jit': jit,
                'jit_raw': jit_raw,
                'liquidity': [p * self.FL for p in snap.V],
                'budget': {k: v for (k, v) in sorted(bmap.items(), key=lambda x: -x[1])}
            }

            with open(f'{path}/gamedata-{datetime}.json', 'w') as f:
                json.dump(data, f, indent=4)

        gc.collect()
        self.reset_profit(time)
        return (self, df, fee_p, fee_a, snap, datetime, path)


def plot_dynamics(args: Tuple[Pool, pd.DataFrame, Bars, Bars, list, str, str]):
    pool, df, player, total, snap, date, path = args
    df = df.drop(columns=['priceLow', 'priceHigh']).groupby('LP').agg('sum')

    fig = plt.figure(figsize=(20, 12), dpi=150, num=1, clear=True)
    gs = gridspec.GridSpec(nrows=3, ncols=5)

    ax: plt.Axes = plt.subplot(gs[1, :2])
    hist_utility(ax, df)

    # ax: plt.Axes = plt.subplot(gs[1, 3:])
    # hist_totalgrad(ax, df)

    ax: plt.Axes = plt.subplot(gs[2, :2])
    hist_fee(ax, player, total, pool.FP)

    ax: plt.Axes = plt.subplot(gs[2, 3:])
    hist_liquidtiy(ax, pool.deltas, snap, pool.removed_liq, 1/5000, 1/500, pool.FP, pool.FL)

    ax: plt.Axes = plt.subplot(gs[2, 2])
    hist_jit_duration(ax, pool.removed_liq)

    ax: plt.Axes = plt.subplot(gs[1, 2])
    plot_price_change(ax, pool.price_change, 1000, 5000)

    ax: plt.Axes = plt.subplot(gs[0, 0])
    scatter_fee_il(ax, df)

    # ax: plt.Axes = plt.subplot(gs[0, 1])
    # scatter_feegrad_ilgrad(ax, df)

    ax: plt.Axes = plt.subplot(gs[0, 2])
    scatter_fund_utility(ax, df)

    ax: plt.Axes = plt.subplot(gs[0, 3])
    scatter_fund_util_per_invest(ax, df)

    # ax: plt.Axes = plt.subplot(gs[0, 4])
    # scatter_fund_totalgrad(ax, df)

    fig.tight_layout()
    fig.savefig(f'{path}/plots-{date}.jpg')
    fig.clear()
    plt.cla()
    plt.clf()
    plt.close(fig)


def main_liq_table_test():
    tab = LiqTable()
    tab.update_liquidity(1056, 3, 13, 12.)
    tab.update_liquidity(1057, 5, 16, 4.)
    print(tab)
    tab.update_liquidity(1058, 13, 20, 33.)
    tab.update_liquidity(1056, 3, 13, -12.)
    tab.update_liquidity(1059, 3, 13, 22.)
    print(tab)

    tab.align([1, 4, 5, 7, 22])

    # total_liquidity = Bars([3, 5, 13, 16, 20], [22, 26, 37, 33])
    # print(tab.get_fees(total_liquidity, 18, 4, 0.5), '\n')
    # print(tab.get_fees(total_liquidity, 18, 16, 0.5))
    # print(tab.get_fees(total_liquidity, 16, 13, 0.5))
    # print(tab.get_fees(total_liquidity, 13, 5, 0.5))
    # print(tab.get_fees(total_liquidity, 5, 4, 0.5))
    # tab.update_liquidity(1058, 4, 10, 4.)
    print(tab)


def main_histogram2d_test():
    vx1 = Bars([4, 10, 16], [100, 200])
    vx2 = Bars([2, 5], [1000])
    h = Histogram2D()
    h.merge(vx1, False)
    h.merge(vx2, True)
    print(h)

    b = Bars([1, 3, 9, 17, 26], [1, 2, 3, 4])
    h2, b2 = align_bars(h, b)
    print(h2)
    print(b2)

    h2.fix(h)
    print(h2)


if __name__ == '__main__':
    main_histogram2d_test()
    # main_liq_table_test()
