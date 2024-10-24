import numpy as np
import psutil
from tqdm import tqdm
from collections import deque

from library.gameutil import *

np.set_printoptions(precision=4, suppress=True, threshold=5)


class Game:
    MAX_EPOCHS = {'GD': 30000, 'BR': 150, 'ILS': 3000, 'ELS': 10000}
    MODES = {'s': 's', 'seq': 's', 'sequential': 's',
             'p': 'p', 'para': 'p', 'parallel': 'p',
             'r': 'r', 'rand': 'r', 'random': 'r'}
    EPS = 1e-7

    def __init__(self, beta, B, ticks, rewards, jit, q, qnew, qnew_probs=None, name='', **kwargs):
        """
        Initializes a Game instance with the given parameters.

        # Parameters:
        * beta (float): The exponent used in calculations involving power functions.
        * B (list or array-like): A list or array representing some capacity or budget constraints.
        * ticks (int): The number of ticks or time steps in the game.
        * rewards (list or array-like): A list or array of rewards associated with the game.
        * jit (float or array-like): A jitter value or array used for calculations, must be non-negative.
        * q (float): A parameter used in the calculation of impermanent loss.
        * qnew (float): A new value of q used in the calculation of impermanent loss.
        * qnew_probs (list or array-like, optional): Probabilities associated with qnew, default is None.
        * name (str, optional): The name of the game instance, default is an empty string.
        * **kwargs: Additional keyword arguments.

        # Raises:
        * ValueError: If any value in jit is negative.
        """

        self.beta = beta
        self.n = len(B)
        self.B = np.array(B)
        self.name = name
        self.ticks = ticks

        try:
            iter(jit)
            jit = np.array(jit)
        except TypeError:
            jit = jit * np.ones(np.shape(rewards))

        if not all(jit >= 0):
            raise ValueError(f'{list(jit)} not completely non-negative')

        self.chi = np.power(jit, beta)
        self.fee = np.array(rewards)

        if qnew_probs is not None:
            self.ilrate = convert_impermanent_loss(ticks, qnew, qnew_probs, None, q)
        else:
            self.ilrate = convert_impermanent_loss(ticks, [qnew], [1.], None, q)

        self.ilrate = np.maximum(0, self.ilrate)

    def find_ne(self, mode='p', method='BR', threshold=(1e-9, 5), return_uty=False, path_dynamics=None, bar=None, **opt_kwargs):
        if isinstance(method, str):
            return self._find_ne(mode, method, threshold, return_uty, path_dynamics=path_dynamics, bar=bar, **opt_kwargs)
        if isinstance(method, dict):
            raise NotImplementedError

    def _find_ne(self, mode, method, threshold, return_uty=False, suppress_overdue=False, path_dynamics=None, bar=None, **opt_kwargs):
        """
        Finds the Nash Equilibrium (NE) for the game using the specified optimization method.

        # Parameters:
        * mode (str): The mode of operation, which can be 's' for sequential, 'p' for parallel, or 'r' for random.
        * method (str): The optimization method to use, such as 'BR' (Best Response), 'ILS' (Inexact Line Search),
                        'ELS' (Exact Line Search), or 'GD' (Gradient Descent).
        * threshold (float or tuple): The convergence threshold. If a float, it represents the epsilon value.
                                    If a tuple, it contains (epsilon, under_epochs).
        * return_uty (bool, optional): If True, returns the utility along with the strategy. Default is False.
        * suppress_overdue (bool, optional): If True, suppresses the OverdueError if optimization fails. Default is False.
        * path_dynamics (str, optional): Path to save the dynamics of the optimization process. Default is None.
        * bar (tqdm, optional): A tqdm progress bar instance. Default is None.
        * **opt_kwargs: Additional keyword arguments for the optimizer.

        # Returns:
        np.ndarray or tuple: The strategy at Nash Equilibrium. If return_uty is True, returns a tuple of the strategy
                                and its utility.
        """

        if isinstance(threshold, float):
            epsilon, under_epochs = threshold, 1
        else:
            epsilon, under_epochs = threshold

        if method == 'BR':
            optimizer = BestResponse(self, **opt_kwargs)
        elif method == 'ILS':
            optimizer = InexactLineSearch(self, **opt_kwargs)
        elif method == 'ELS':
            optimizer = ExactLineSearch(self, **opt_kwargs)
        elif method == 'GD':
            optimizer = MomentumGradientAscent(self, **opt_kwargs)
        else:
            raise ValueError(f'Invalid method {method}')

        mode = Game.MODES[mode]

        x = np.ones((self.n, len(self.fee))) / (2. * len(self.fee)) * self.B[:, None]
        wsum = np.sum(x ** self.beta, 0)
        lastx = x.copy()

        tqdm_native = bar is None
        if tqdm_native:
            bar = tqdm(total=opt_kwargs.get('epochs', Game.MAX_EPOCHS[method]))
        else:
            bar.reset(total=opt_kwargs.get('epochs', Game.MAX_EPOCHS[method]))

        underthr, dynamics, success = 0, [], False

        window = deque()
        window_size, pop_num = 500, 10
        process = psutil.Process()

        for _ in range(bar.total):
            if method in ['ILS', 'ELS']:
                x = optimizer.update(x)
            elif mode == 's':
                for lp in range(self.n):
                    temp = x[lp].copy()
                    x[lp] = optimizer.update(lp, wsum-x[lp]**self.beta, temp)
                    wsum += x[lp] ** self.beta - temp ** self.beta
            elif mode in 'pr':
                xn = x.copy()
                lps = range(self.n) if mode == 'p' else np.random.choice(range(self.n), (self.n, ))
                for lp in lps:
                    xn[lp] = optimizer.update(lp, wsum-x[lp]**self.beta, x[lp])

                x = xn
                wsum = np.sum(x ** self.beta, 0)

            stepdiff = np.max(abs(x-lastx) / self.B[:, None])
            if method == 'GD':
                stepdiff /= optimizer.lr_step

            vfunc = self.V(x)
            if path_dynamics is not None:
                dynamics.append((stepdiff, vfunc, self.U(x)))

            bar.set_description(
                f'{self.name} | '
                f'N={self.n:<3} M={len(self.fee):<4} '
                # f'{process.memory_info().rss/2**20:>6.1f} MiB | '
                f'V={np.format_float_scientific(-vfunc, precision=1, exp_digits=1):>7} '
                f'D={np.format_float_scientific(stepdiff, precision=1, exp_digits=1):>7}')

            underthr = 0 if -vfunc > epsilon else underthr + 1

            if underthr >= under_epochs:
                bar.total = bar.n
                bar.refresh()
                success = True
                break

            window.append(-vfunc)
            if len(window) > window_size:
                base = 0
                for _ in range(pop_num):
                    base += window.popleft()
                if (-vfunc) / base * pop_num > .75:
                    bar.total = bar.n
                    bar.refresh()
                    break

            bar.update()
            lastx = x.copy()

        if tqdm_native:
            bar.close()

        if path_dynamics is not None:
            plot_dynamics_util(dynamics, path_dynamics, converged=success)

        if not success:
            print(f'Warning: Optimization failed to meet early-stop conditions!')
            if not suppress_overdue:
                raise OverdueError()

        return (lastx, self.U(lastx)) if return_uty else lastx

    def Um(self, x):
        """
        Calculates the marginal utility for each player given their strategy profile.

        Parameters:
        x (np.ndarray): A 2D array representing the strategy profile of all players.
                        The shape of x should be (n, m) where n is the number of players and m is the number of strategies.

        Returns:
        np.ndarray: A 2D array representing the marginal utility for each player.
                    The shape of the return array is (n, m), where n is the number of players and m is the number of strategies.
        """
        return self.fee * (x**self.beta) / np.maximum(np.sum(x ** self.beta, 0) + self.chi, 1e-10) - self.ilrate * x

    def Fm(self, x):
        return self.fee * (x**self.beta) / np.maximum(np.sum(x ** self.beta, 0) + self.chi, 1e-10)

    def Cm(self, x):
        return self.ilrate * x

    def U(self, x):
        """
        Computes the utility for each player given the strategy profile.

        Parameters:
        x (np.ndarray): A 2D array representing the strategy profile of all players.
                        The shape of x should be (n, m) where n is the number of players and m is the number of strategies.

        Returns:
        np.ndarray: A 1D array representing the utility for each player.
        """
        return np.sum(self.Um(x), 1)

    def F(self, x):
        """
        Computes the total fee for each player given the strategy profile.

        Parameters:
        x (np.ndarray): A 2D array representing the strategy profile of all players.
                        The shape of x should be (n, m) where n is the number of players and m is the number of strategies.

        Returns:
        np.ndarray: A 1D array representing the fee for each player.
        """
        return np.sum(self.fee * (x**self.beta) / np.maximum(np.sum(x ** self.beta, 0) + self.chi, 1e-10), 1)

    def C(self, x):
        """
        Computes the cost for each player given the strategy profile.

        Parameters:
        x (np.ndarray): A 2D array representing the strategy profile of all players.
                        The shape of x should be (n, m) where n is the number of players and m is the number of strategies.

        Returns:
        np.ndarray: A 1D array representing the cost for each player.
        """
        return np.sum(self.ilrate * x, 1)

    def V(self, x, y=None):
        """
        Computes the average utility difference between the current strategy and the best response strategy.

        Parameters:
        x (np.ndarray): A 2D array representing the current strategy profile of all players.
                        The shape of x should be (n, m) where n is the number of players and m is the number of strategies.
        y (np.ndarray, optional): A 2D array representing the best response strategy profile of all players.
                                  If None, the best response is computed based on the current strategy x.

        Returns:
        float: The mean utility difference between the current strategy and the best response strategy.
        """
        if y is None:
            y = self.best_response(x)

        udiff = [self.U_under(x, x[lp], lp) - self.U_under(x, y[lp], lp) for lp in range(self.n)]
        return np.mean(udiff)

    def U_under(self, x, y, lp) -> float:
        """
        Computes the utility for a specific player given a hypothetical strategy.

        Parameters:
        x (np.ndarray): A 2D array representing the current strategy profile of all players.
                        The shape of x should be (n, m) where n is the number of players and m is the number of strategies.
        y (np.ndarray): A 1D array representing the hypothetical strategy for the specified player.
        lp (int): The index of the player for whom the utility is to be computed.

        Returns:
        float: The utility value for the specified player given the hypothetical strategy.
        """
        y = np.array(y)
        total_weight = np.maximum(1e-10, np.sum(x**self.beta, 0) + self.chi + y**self.beta - x[lp]**self.beta)
        util = np.sum(self.fee * (y**self.beta) / total_weight - self.ilrate * y)
        return util

    def _best_response_single(self, lp, denom):
        """
        Computes the best response strategy for a single player in the game.

        Parameters:
        lp (int): The index of the player for whom the best response is to be computed.
        denom (np.ndarray): An array representing the denominator values used in the calculation,
                            adjusted for the current player's strategy.

        Returns:
        np.ndarray: An array representing the best response strategy for the specified player.
        """
        denom[np.isclose(denom, 0, atol=1e-6)] = 1e-8
        if not all(denom >= 0):
            raise ValueError(f'{list(denom)} is partially negative')

        def _xi(lam):
            r = np.maximum(np.sqrt(self.fee * denom / np.maximum(self.ilrate + lam, 1e-14)) - denom, 0)
            r[np.isclose(denom, 0)] = self.B[lp] / len(self.fee) * 1e-9
            return r

        x0 = _xi(0)
        if x0.sum() > self.B[lp]:
            lam = root(lambda t: _xi(t).sum(), target=self.B[lp], thr=1e-9)
            x0 = _xi(lam)

        return x0

    def best_response(self, x, lp=None) -> np.ndarray:
        """
        Computes the best response strategy for the game.

        Parameters:
        x (np.ndarray): A 2D array representing the current strategy profile of all players.
                        The shape of x should be (n, m) where n is the number of players and m is the number of strategies.
        lp (int, optional): The index of the player for whom the best response is to be computed.
                            If None, computes the best response for all players. Default is None.

        Returns:
        np.ndarray: A 2D array representing the best response strategy. If lp is specified, returns the best response for that player.
                    Otherwise, returns the best response for all players.
        """
        assert np.shape(x) == (self.n, len(self.fee))
        denom = np.sum(x**self.beta, 0) + self.chi
        if lp is None:
            y = np.zeros(np.shape(x))
            for lp in range(self.n):
                y[lp, :] = self._best_response_single(lp, denom-x[lp]**self.beta)
        else:
            y = self._best_response_single(lp, denom-x[lp]**self.beta)

        return y

    def hypothetical_response(self, x, lps=None, added=1):
        """
        Computes the hypothetical response for specified players by temporarily increasing their budget.

        Parameters:
        x (np.ndarray): A 2D array representing the current strategy profile of all players.
                        The shape of x should be (n, m) where n is the number of players and m is the number of strategies.
        lps (array-like, optional): An array of player indices for whom the hypothetical response is to be computed.
                                    If None, computes the response for all players. Default is None.
        added (int, optional): The amount to temporarily add to each player's budget. Default is 1.

        Returns:
        tuple: A tuple containing:
            - np.ndarray: A 2D array representing the hypothetical response strategy for the specified players.
            - np.ndarray: An array of utilities for the specified players after the hypothetical response.
        """
        x = np.array(x)
        assert np.shape(x) == (self.n, len(self.fee))

        if lps is None:
            lps = np.arange(self.n, dtype=int)

        denom = np.sum(x**self.beta, 0) + self.chi
        us = []
        for lp in lps:
            self.B[lp], temp = self.B[lp] + added, self.B[lp]
            y = self._best_response_single(lp, denom-x[lp, :]**self.beta)
            self.B[lp] = temp
            us.append(self.U_under(x, y, lp))

        return y[lps, :], np.array(us)

    def plot_strategy(self, strategy, fig=None, axs=None):
        plot_strategy_util(self.B, strategy, self.Fm(strategy), self.Cm(strategy), None, None, None, fig=fig, axs=axs)
