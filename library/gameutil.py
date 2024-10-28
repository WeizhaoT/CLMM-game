import numpy as np
import matplotlib.pyplot as plt

import gc
import warnings
import matplotlib
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from typing import Callable, List, Tuple, TYPE_CHECKING

from library import Bars
from library.util import color_cycle


if TYPE_CHECKING:
    from library.game import Game


matplotlib.use('Agg')


class OverdueError(Exception):
    """ A custom exception raised when the optimization process exceeds the specified maximum number of epochs. """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def root(func: Callable[[float], float], target: float = 0, thr: float = 1e-9, maxiter: int = 10000) -> float:
    """
    Finds the root of a given function using a binary search method.

    Parameters:
        func (Callable[[float], float]): The function for which the root is to be found.
        target (float, optional): The target value for the function. Defaults to 0.
        thr (float, optional): The threshold for convergence. Defaults to 1e-9.
        maxiter (int, optional): The maximum number of iterations allowed. Defaults to 10000.

    Returns:
        float: The estimated root of the function where func(x) is approximately equal to the target.
    """
    f0 = func(0.)
    if abs(f0 - target) < thr:
        return 0.

    x = 1.
    while bool(func(x) > target) == bool(f0 > target):
        x *= 2.

    l, r = (x/2. if x > 1. else 0.), x
    for _ in range(maxiter):
        fm = func((l+r)/2.)
        if np.isclose(fm, target, rtol=thr, atol=thr):
            return (l+r)/2.
        if bool(fm > target) == bool(f0 > target):
            l += (r-l) / 2.
        else:
            r -= (r-l) / 2.

    warnings.warn(f'Root not fully converged {abs(fm-target)/target:.1e}', UserWarning)
    return (l+r) / 2.


def single_range_br(val_x: float, val_y: float, il: float, bound: float, weights: Bars) -> float:
    """
    When LP can only deposit liquidity in a single range, compute its best response liquidity in that range

    Args:
        val_x (float): USD value of X tokens.
        val_y (float): USD value of Y tokens.
        il (float): The impermanent loss of the price range.
        bound (float): The upper bound of liquidity.
        weights (Bars): An instance of Bars containing the weights fo other LPs.

    Returns:
        float: The optimal liquidity.
    """

    DX, DY = weights.T[0]**-.5 - weights.T[-1]**-.5, weights.T[-1]**.5 - weights.T[0]**.5

    fees = []
    for _, a, b in weights.intervals():
        fees.append(val_x * (a**-.5 - b**-.5) / DX + val_y * (b**.5 - a**.5) / DY)

    def grad(x):
        return sum(fee * float(w) / (max(x+float(w), 1e-7) ** 2) for fee, w in zip(fees, weights))

    if grad(0) <= il:
        return 0
    elif grad(bound) >= il:
        return bound

    return root(grad, il, thr=1e-8)


def IL(pstart: Tuple[float, float, float], pend: Tuple[float, float, float], a: float, b: float):
    """
    Calculates the impermanent loss between two price states within a specified range.

    Parameters:
        pstart (Tuple[float, float, float]): The starting state represented as a tuple (q0, x0, y0),
            where q0 is the initial price, x0 is the initial amount of asset X, and y0 is the initial amount of asset Y.
        pend (Tuple[float, float, float]): The ending state represented as a tuple (q1, x1, y1),
            where q1 is the final price, x1 is the final amount of asset X, and y1 is the final amount of asset Y.
        a (float): The lower bound of the price range.
        b (float): The upper bound of the price range.

    Returns:
        float: The calculated impermanent loss between the two states within the specified range.
    """
    q0, x0, y0 = pstart
    q1, x1, y1 = pend
    p0 = max(a, min(b, q0))
    p1 = max(a, min(b, q1))
    return (x1 * (p0**-.5-p1**-.5) + y1 * (p0**.5-p1**.5)) / (x0 * (p0**-.5-b**-.5) + y0 * (p0**.5-a**.5))


def convert_impermanent_loss(price_ticks: List[float],
                             prices: List[Tuple[float, float, float]],
                             probs: List[float],
                             p0: Tuple[float, float, float]) -> np.ndarray:
    """
    Converts a series of price states and their probabilities into an array of impermanent losses for specified price ranges.

    Parameters:
        price_ticks (List[float]): A list of price tick values defining the boundaries of price ranges.
        prices (List[Tuple[float, float, float]]): A list of tuples representing different price states, 
            where each tuple contains (price, amount of asset X, amount of asset Y).
        probs (List[float]): A list of probabilities associated with each price state.
        p0 (Tuple[float, float, float]): The initial price state represented as a tuple (initial price, initial amount of asset X, initial amount of asset Y).

    Returns:
        np.ndarray: An array of impermanent losses calculated for each price range defined by the price ticks.
    """
    return np.array([
        sum(IL(p0, p, a, b) * pi for p, pi in zip(prices, probs)) for a, b in zip(price_ticks[:-1], price_ticks[1:])
    ])


def plot_dynamics_util(dynamics: List[Tuple[float, float, float]], figpath: str, converged: bool):
    """
    Plots the dynamics of a given optimization process, including step differences, value function, and utility over epochs.

    Parameters:
        dynamics (List[Tuple[float, float, float]]): A list of tuples where each tuple contains step difference, 
            value function, and utility values for each epoch.
        figpath (str): The file path where the plot will be saved.
        converged (bool): A boolean indicating whether the optimization process has converged.

    Returns:
        None: The function saves the plot to the specified file path and does not return any value.
    """
    stepdiff, vfunc, utility = zip(*dynamics)
    utility = np.array(utility)
    E = min(len(stepdiff), utility.shape[0])
    N = utility.shape[1]
    x = np.arange(1, E+1)

    ncols = 2
    nrows = (N+1) // ncols + 1

    _PLOT_KWARGS = {'lw': .6, 'marker': None, 'markevery': .05}

    plt.ioff()
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*3.5))
    axs = axs.flatten()

    ax = axs[0]
    ax.set_title(r'$\|x^{{(t)}} - x^{{(t-1)}} \|_{{\infty}}$ ' + ('(DIVERGED)' if not converged else ''))
    ax.plot(x, stepdiff, **_PLOT_KWARGS)

    ax = axs[1]
    ax.set_title(r'$-V(x)$')
    ax.plot(x, -np.array(vfunc), **_PLOT_KWARGS)

    for i, ax in enumerate(axs[2:N+2]):
        ax.set_title(r'$U_{{{:d}}}(x^{{(t)}})$'.format(i+1))
        ax.plot(x, utility[:, i], **_PLOT_KWARGS)

    for ax in axs:
        ax.set_yscale('log')
        ax.set_xlabel('Epochs')
        ax.grid(True)

    fig.tight_layout()
    fig.savefig(figpath)
    plt.close('all')
    gc.collect()


def plot_dynamics_util(dynamics: List[Tuple[float, float, float]], figpath: str, converged: bool):
    stepdiff, vfunc, utility = zip(*dynamics)
    utility = np.array(utility)
    E = min(len(stepdiff), utility.shape[0])
    N = utility.shape[1]
    x = np.arange(1, E+1)

    ncols = 2
    nrows = (N+1) // ncols + 1

    _PLOT_KWARGS = {'lw': .6, 'marker': None, 'markevery': .05}

    plt.ioff()
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*3.5))
    axs = axs.flatten()

    ax = axs[0]
    ax.set_title(r'$\|x^{{(t)}} - x^{{(t-1)}} \|_{{\infty}}$ ' + ('(DIVERGED)' if not converged else ''))
    ax.plot(x, stepdiff, **_PLOT_KWARGS)

    ax = axs[1]
    ax.set_title(r'$-V(x)$')
    ax.plot(x, -np.array(vfunc), **_PLOT_KWARGS)

    for i, ax in enumerate(axs[2:N+2]):
        ax.set_title(r'$U_{{{:d}}}(x^{{(t)}})$'.format(i+1))
        ax.plot(x, utility[:, i], **_PLOT_KWARGS)

    for ax in axs:
        ax.set_yscale('log')
        ax.set_xlabel('Epochs')
        ax.grid(True)

    fig.tight_layout()
    fig.savefig(figpath)
    plt.close('all')
    gc.collect()


def plot_strategy_util(budgets: List[float],
                       strategy: np.ndarray,
                       utility: np.ndarray,
                       il: np.ndarray,
                       baseline=None,
                       baseutil=None,
                       baseil=None,
                       fig=None,
                       axs: List[plt.Axes] = None):
    """
    Plots the strategy utilization, utility, and impermanent loss for a given set of strategies and budgets.

    Parameters:
        budgets (List[float]): A list of n budget values for liquidity providers (LP). n is the number of LPs
        strategy (np.ndarray): A 2D array of shape (n, m) representing the strategy allocations for each LP across different ranges. \
            m is the number of price ranges.
        utility (np.ndarray): A 2D array of shape (n, m) representing the utility values for each LP across different ranges.
        il (np.ndarray): A 2D array of shape (n, m) representing the impermanent loss values for each LP across different ranges.
        baseline (optional): NOT IMPLEMENTED
        baseutil (optional): NOT IMPLEMENTED
        baseil (optional): NOT IMPLEMENTED
        fig (optional): A matplotlib figure object to use for plotting.
        axs (List[plt.Axes], optional): A list of matplotlib axes objects to use for plotting.
    """
    n, m = np.shape(strategy)

    _TEXT_KWARGS = {'fontsize': 'large', 'fontweight': 'demi', 'ha': 'center',  'va': 'bottom'}

    usum, il = np.sum(utility, 1), np.sum(il, 1) / np.sum(strategy, 1)

    has_baseline, update_fig = (baseline is not None), False
    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(n*2.4 + 5., 6))
    else:
        update_fig = True

    for ax in axs:
        ax.clear()
        ax.set_xlabel('LP')
        ax.set_xlim(0, n+1)
        ax.set_xticks(range(1, n+1))
        ax.grid(axis='y')

    def add_baseline_shadow(ax, i, h, baseh):
        if has_baseline:
            ax.add_patch(Rectangle((i+.5, 0), 1, min(h[i], baseh[i]),
                         edgecolor='white', fill=False, lw=0, hatch='///', alpha=.5))
            if baseh[i] > h[i]:
                ax.add_patch(Rectangle((i+.5, h[i]), 1, baseh[i]-h[i],
                             edgecolor='black', fill=False, lw=0, hatch='///', alpha=.5))

    cs = color_cycle()
    patches = [mpatches.Patch(color=next(cs), label=f'[{l}, {l+1}]') for l in range(m)]
    fig.legend(handles=patches, loc='upper center', bbox_to_anchor=[.5, 1.03], ncol=12)

    # ================================ Strategy ================================
    ax = axs[0]
    ax.set_title(f"Funds Used, Total = {strategy.sum():.4f}")
    ax.set_ylim(0, budgets[-1] * 1.05)

    for i in range(n):
        h, cs = 0, color_cycle()
        for s in strategy[i]:
            ax.add_patch(Rectangle((i+.5, h), 1, s, facecolor=next(cs), lw=0.))
            h += s

        ax.text(i+1, h-budgets[-1]*.05, f'{h:.4f}', color='white', **_TEXT_KWARGS)

    for i in range(n):
        ax.add_patch(Rectangle((i+.5, 0), 1, budgets[i], edgecolor='black', fill=False, lw=1.5))
        add_baseline_shadow(ax, i, strategy.sum(1), baseline)

    # ================================ Utility =================================
    ax = axs[1]
    ax.set_title(f"Utility, Total = {utility.sum():.4f}")

    ymax = max(max(usum), max(baseutil) if has_baseline else float('-inf'))
    ax.set_ylim(0, ymax * 1.05)

    for i, urow in enumerate(utility):
        h, cs = 0, color_cycle()
        for u in urow:
            ax.add_patch(Rectangle((i+.5, h), 1, u, facecolor=next(cs), lw=0.))
            h += u

        ax.text(i+1, h-ymax*.05, f'{h:.4f}', color='white', **_TEXT_KWARGS)

    for i in range(n):
        ax.add_patch(Rectangle((i+.5, 0), 1, usum[i], edgecolor='black', fill=False, lw=1.5))
        add_baseline_shadow(ax, i, usum, baseutil)

    # ============================ Impermanent Loss ============================
    ax = axs[2]
    ax.set_title("Unit Impermanent Loss")
    ymax = max(max(il), max(baseil/baseline) if has_baseline else float('-inf'))
    ax.set_ylim(0, ymax * 1.05)

    for i in range(n):
        ax.add_patch(Rectangle((i+.5, 0), 1, il[i], edgecolor='black', fill=False, lw=1.5))
        if has_baseline:
            ax.add_patch(Rectangle((i+.5, 0), 1, baseil[i]/baseline[i],
                         edgecolor='black', fill=False, lw=1.5, ls=':', hatch='///', alpha=.5))

        txt = ax.text(i+1, il[i]-ymax*.06, f'{il[i]:.4f}', color='blue', **_TEXT_KWARGS)
        txt.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='white'))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if not update_fig:
        plt.show()


def proj_simpleton(x, B):
    """
    Projects a vector onto the simplex defined by the constraint that the sum of its elements is less than or equal to B.

    Parameters:
        x (array-like): The input vector to be projected.
        B (float): The upper bound for the sum of the elements of the projected vector.

    Returns:
        np.ndarray: The projected vector where all elements are non-negative and the sum of the elements is less than or equal to B.
    """
    x = np.array(x)
    p = np.maximum(0, -np.sort(-x))

    if p[0] <= 0:
        return np.zeros(x.shape, dtype=float)

    c = np.cumsum(p)
    if c[-1] <= B:
        return np.maximum(0, x, dtype=float)

    k = np.searchsorted(c[:-1] - np.arange(1, len(x)) * p[1:], B, side='right')  # seq[k-1] <= B < seq[k]
    return np.maximum(0, x-(c[k]-B) / (k+1), dtype=float)


class Optimizer:
    """Base class for Nash equilibrium solving algorithms."""

    def __init__(self, game: Game, **kwargs):
        self.game = game

    def update(self, x, **kwargs):
        pass


class Relaxation(Optimizer):
    """Relaxation method."""

    def __init__(self, game: Game, gamma: float, **kwargs):
        """
        Initializes the Relaxation optimizer with a game instance and a relaxation parameter.

        Parameters:
            game (Game): An instance of the Game class representing the game to be optimized.
            gamma (float): The relaxation parameter that controls the step size in the update process.
            **kwargs: Additional keyword arguments for further customization.

        Returns:
            None
        """
        super().__init__(game)
        self.epochs: int = 0
        self.gamma = gamma

    def update(self, x, **kwargs):
        """
        Updates the strategy using the relaxation method.

        Args:
            x (array-like): The current strategy vector.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: The updated strategy vector after applying the relaxation method.
        """
        y = self.game.best_response(x)
        d = y - x
        return x + d * self.gamma / (self.epochs + 1)
