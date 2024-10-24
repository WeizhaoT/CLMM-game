import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

import gc
import warnings
import matplotlib
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from itertools import cycle

from library import Bars

matplotlib.use('Agg')


class OverdueError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def standard_deviation(PI):
    return np.sqrt(sum(p * (i-(len(PI)-1)/2) ** 2 for i, p in enumerate(PI)))


def raw_revenue(num_prices, ratio=1):
    k = (num_prices-1) // 2
    base = 1 / (ratio * k + k + 1)
    return base * (ratio - (ratio - 1) * abs(np.arange(num_prices) - k) / k)


def root(func, target=0, thr=1e-9, maxiter=10000):
    """
    Finds the root of a given function using a binary search method.

    Parameters:
    func (callable): The function for which the root is to be found.
    target (float, optional): The target value for the function. Defaults to 0.
    thr (float, optional): The threshold for convergence. Defaults to 1e-9.
    maxiter (int, optional): The maximum number of iterations allowed. Defaults to 10000.

    Returns:
    float: The estimated root of the function.
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
    Computes the optimal value of x that maximizes a utility function subject to a given impermanent loss constraint.

    The utility function is defined as:
    U = sum(f_m * x / (x + w_m)) - IL * x
    where f_m are the fees and w_m are the weights.

    Parameters:
    val_x (float): The coefficient for the x component in the fee calculation.
    val_y (float): The coefficient for the y component in the fee calculation.
    il (float): The impermanent loss constraint.
    bound (float): The upper bound for the value of x.
    weights (Bars): An instance of Bars containing the weight intervals.

    Returns:
    float: The optimal value of x that satisfies the impermanent loss constraint.
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


def IL(pstart, pend, a, b):
    """
    Calculate the impermanent loss between two price points within specified bounds.

    Parameters:
    pstart (tuple): A tuple (q0, x0, y0) representing the initial state, where q0 is the initial price,
                    x0 is the initial amount of asset x, and y0 is the initial amount of asset y.
    pend (tuple): A tuple (q1, x1, y1) representing the final state, where q1 is the final price,
                  x1 is the final amount of asset x, and y1 is the final amount of asset y.
    a (float): The lower bound for the price.
    b (float): The upper bound for the price.

    Returns:
    float: The calculated impermanent loss between the two price points, adjusted for the specified bounds.
    """
    q0, x0, y0 = pstart
    q1, x1, y1 = pend
    p0 = max(a, min(b, q0))
    p1 = max(a, min(b, q1))
    return (x1 * (p0**-.5-p1**-.5) + y1 * (p0**.5-p1**.5)) / (x0 * (p0**-.5-b**-.5) + y0 * (p0**.5-a**.5))


def impermanent_loss_rate_agg(a, b, prices, probs, il_func, p0=None, **kwargs):
    return sum(IL(p0, p, a, b) * pi for p, pi in zip(prices, probs))


def convert_impermanent_loss(price_ticks, prices, probs, il_func, p0=None, **kwargs):
    return np.array([impermanent_loss_rate_agg(price_ticks[l], price_ticks[l+1], prices, probs, il_func, p0, **kwargs) for l in range(len(price_ticks)-1)])


TEXT_KWARGS = {
    'fontsize': 'large',
    'fontweight': 'demi',
    'ha': 'center',
    'va': 'bottom'
}


def plot_dynamics_util(dynamics, figpath, converged):
    stepdiff, vfunc, utility = zip(*dynamics)
    utility = np.array(utility)
    E = min(len(stepdiff), utility.shape[0])
    N = utility.shape[1]
    x = np.arange(1, E+1)

    ncols = 2
    nrows = (N+1) // ncols + 1

    PLOT_KWARGS = {'lw': .6, 'marker': None, 'markevery': .05}

    plt.ioff()
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*3.5))
    axs = axs.flatten()

    ax = axs[0]
    ax.set_title(r'$\|x^{{(t)}} - x^{{(t-1)}} \|_{{\infty}}$ ' + ('(DIVERGED)' if not converged else ''))
    ax.plot(x, stepdiff, **PLOT_KWARGS)

    ax = axs[1]
    ax.set_title(r'$-V(x)$')
    ax.plot(x, -np.array(vfunc), **PLOT_KWARGS)

    for i, ax in enumerate(axs[2:N+2]):
        ax.set_title(r'$U_{{{:d}}}(x^{{(t)}})$'.format(i+1))
        ax.plot(x, utility[:, i], **PLOT_KWARGS)

    for ax in axs:
        ax.set_yscale('log')
        ax.set_xlabel('Epochs')
        ax.grid(True)

    fig.tight_layout()
    fig.savefig(figpath)
    plt.close('all')
    gc.collect()


def plot_strategy_util(budgets, strategy, utility, il, baseline=None, baseutil=None, baseil=None, fig=None, axs=None):
    n, m = np.shape(strategy)

    usum, il = np.sum(utility, 1), np.array(il) / np.sum(strategy, 1)

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

    def colorgen():
        return cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    def add_baseline_shadow(ax, i, h, baseh):
        if has_baseline:
            ax.add_patch(Rectangle((i+.5, 0), 1, min(h[i], baseh[i]),
                         edgecolor='white', fill=False, lw=0, hatch='///', alpha=.5))
            if baseh[i] > h[i]:
                ax.add_patch(Rectangle((i+.5, h[i]), 1, baseh[i]-h[i],
                             edgecolor='black', fill=False, lw=0, hatch='///', alpha=.5))

    COLOR_ITER = colorgen()
    patches = [mpatches.Patch(color=next(COLOR_ITER), label=f'[{l}, {l+1}]') for l in range(m)]
    fig.legend(handles=patches, loc='upper center', bbox_to_anchor=[.5, 1.03], ncol=12)

    # ================================ Strategy ================================
    ax = axs[0]
    ax.set_title(f"Funds Used, Total = {strategy.sum():.4f}")
    ax.set_ylim(0, budgets[-1] * 1.05)

    for i in range(n):
        h, COLOR_ITER = 0, colorgen()
        for s in strategy[i]:
            ax.add_patch(Rectangle((i+.5, h), 1, s, facecolor=next(COLOR_ITER), lw=0.))
            h += s

        ax.text(i+1, h-budgets[-1]*.05, f'{h:.4f}', color='white', **TEXT_KWARGS)

    for i in range(n):
        ax.add_patch(Rectangle((i+.5, 0), 1, budgets[i], edgecolor='black', fill=False, lw=1.5))
        add_baseline_shadow(ax, i, strategy.sum(1), baseline)

    # ================================ Utility =================================
    ax = axs[1]
    ax.set_title(f"Utility, Total = {utility.sum():.4f}")

    ymax = max(max(usum), max(baseutil) if has_baseline else float('-inf'))
    ax.set_ylim(0, ymax * 1.05)

    for i, urow in enumerate(utility):
        h, COLOR_ITER = 0, colorgen()
        for u in urow:
            ax.add_patch(Rectangle((i+.5, h), 1, u, facecolor=next(COLOR_ITER), lw=0.))
            h += u

        ax.text(i+1, h-ymax*.05, f'{h:.4f}', color='white', **TEXT_KWARGS)

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

        txt = ax.text(i+1, il[i]-ymax*.06, f'{il[i]:.4f}', color='blue', **TEXT_KWARGS)
        txt.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='white'))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if not update_fig:
        plt.show()


def proj_simpleton(x, B):
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
    def __init__(self, game, **kwargs):
        self.game = game

    def update(self, lp, par, x0, **kwargs):
        pass


class BestResponseScipy(Optimizer):
    def update(self, lp, par, x0=None):
        if x0 is None:
            x0 = self.game.B[lp] / len(self.game.fee) / 2.

        def _neg_utility(xlp, xsum):
            lp_weights = xlp ** self.game.beta
            return np.sum(self.game.ilrate * xlp) - np.sum(self.game.fee * lp_weights / (lp_weights + xsum + self.game.chi))

        res: opt.OptimizeResult = opt.minimize(_neg_utility, x0=x0, args=(par,), method='trust-constr', tol=1e-10,
                                               bounds=[[0, None]] * len(self.game.fee),
                                               constraints=opt.LinearConstraint([1] * len(self.game.fee), 0, self.game.B[lp]))
        if not res.success:
            raise ValueError(f'Optimization failed at LP {lp}!\n    reason={res.message}')

        return res.x


class BestResponse(Optimizer):
    def update(self, lp, par, *args, **kwargs):
        return self.game._best_response_single(lp, par)


class ExactLineSearch(Optimizer):
    def __init__(self, game, gamma, **kwargs):
        super().__init__(game)
        self.epochs = 0
        self.gamma = gamma

    def update(self, x, **kwargs):
        y = self.game.best_response(x)
        d = y - x
        return x + d * self.gamma / (self.epochs + 1)


class InexactLineSearch(Optimizer):
    def __init__(self, game, beta, sigma):
        super().__init__(game)
        self.beta = beta
        self.sigma = sigma

    def update(self, x, **kwargs):
        y = self.game.best_response(x)
        d = y - x
        v0 = self.game.vfunc(x, y)
        delta = np.sqrt(np.sum(d ** 2)) * self.sigma

        left, right = 0, 1
        while True:
            t = self.beta ** right
            if self.game.vfunc(x + d*t) > v0 - delta * (t**2):
                break
            right *= 2

        while right - left > 1:
            mid = (right + left) // 2
            t = self.beta ** mid
            if self.game.vfunc(x + d*t) > v0 - delta * (t**2):
                right = mid
            else:
                left = mid

        return x + d*(self.beta ** left)


class MomentumGradientAscent(Optimizer):
    def __init__(self, game, lr=5e0, beta=.8, **kwargs):
        super().__init__(game)
        self.beta = beta
        self.momentum = [0.] * game.n

        if isinstance(lr, (float, int)):
            self.lr = LearningRateScheduler(game.n, lr)
        else:
            self.lr = lr

        self.lr_step = 0

    def update(self, lp, par, x0):
        self.lr_step = self.lr.sample()
        grad = self.game.fee * (par + self.game.chi) * x0**(self.game.beta-1) / \
            np.maximum(1e-7, par + x0**self.game.beta + self.game.chi) ** 2 - self.game.ilrate
        self.momentum[lp] = self.beta * self.momentum[lp] + (1-self.beta) * self.lr_step * grad
        return proj_simpleton(x0 + self.momentum[lp], self.game.B[lp])


class LearningRateScheduler:
    def __init__(self, N, lr, *args, **kwargs) -> None:
        self.lr = lr
        self.N = N

    def sample(self):
        return self.lr


class CosineAnnealingScheduler(LearningRateScheduler):
    def __init__(self, N, lr, lr_min, epochs, *args, **kwargs) -> None:
        super().__init__(N, lr)
        self.lr_min = lr_min
        self.epochs = epochs
        self.cnt = -1

    def sample(self):
        self.cnt = (self.cnt + 1) % (self.N * self.epochs)
        return self.lr_min + (self.lr - self.lr_min) * (.5 + .5 * np.cos((self.cnt // self.N) * np.pi / self.epochs))
