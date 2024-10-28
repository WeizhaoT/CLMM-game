import numpy as np
from tqdm import tqdm
from collections import deque

from typing import List, Tuple, Union, Optional
from library.gameutil import *

np.set_printoptions(precision=4, suppress=True, threshold=5)


class Game:
    MAX_EPOCHS = {'RELAX': 10000}
    """ Max number of epochs for each method supported. """
    EPS = 1e-7
    """ Epsilon value for convergence checks. """

    def __init__(self,
                 alpha: float,
                 B: List[float],
                 ticks: List[float],
                 rewards: List[float],
                 jit: List[float],
                 q: Tuple[float, float, float],
                 qnew: Union[Tuple[float, float, float], List[Tuple[float, float, float]]],
                 qnew_probs: Optional[List[float]] = None,
                 name: str = '',
                 **kwargs):
        """
        Initializes a Game instance with the given parameters.

        Args:
            alpha (float): The exponent used in the utility calculations.
            B (List[float]): Budget of each player.
            ticks (List[float]): Ticks of prices.
            rewards (List[float]): Fee reward in each price range.
            jit (List[float]): Just-in-time (JIT) LPs' share in each price range.
            q (Tuple[float, float, float]): Initial (pool price, USD price of X, USD price of Y) of the game.
            qnew (Union[Tuple[float, float, float], List[Tuple[float, float, float]]]): \
                Ending (pool price, USD price of X, USD price of Y) of the game. \
                If a list, it represents a distribution and `qnew_probs` must be provided.
            qnew_probs (Optional[List[float]]): A list of probabilities associated with the new state. 
                If not provided, `qnew` must be a Tuple[float, float, float].
            name (str): The name of the game instance, default is an empty string.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If any JIT share is negative.
        """

        self.alpha = alpha
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

        self.chi = np.power(jit, alpha)
        self.fee = np.array(rewards)

        if qnew_probs is not None:
            self.ilrate = convert_impermanent_loss(ticks, qnew, qnew_probs, q)
        else:
            self.ilrate = convert_impermanent_loss(ticks, [qnew], [1.], q)

        self.ilrate = np.maximum(0, self.ilrate)

    def find_ne(self,
                method: str = 'RELAX',
                threshold: Union[float, Tuple[float, float]] = (1e-9, 5),
                path_dynamics: str = None,
                bar: tqdm = None,
                suppress_overdue: bool = False,
                **opt_kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Finds the Nash Equilibrium (NE) for the game using the specified optimization method.

        Args:
            method (str): Only supports 'RELAX' (relaxation method).
            threshold (Union[float, Tuple[float, float]]): The convergence threshold. \
                If a float, it represents the epsilon value. \
                If a tuple, it contains (epsilon, under_epochs).
            return_uty (bool, optional): If True, returns the utility along with the strategy. Default is False.
            path_dynamics (str, optional): Path to save the dynamics plot of the optimization process. Default is None.
            bar (tqdm, optional): A tqdm progress bar instance. If None, a new progress bar is created. Default is None.
            suppress_overdue (bool, optional): If True, suppresses the OverdueError if optimization fails. Default is False.
            **opt_kwargs: Additional keyword arguments for the optimizer.

        Returns:
            Tuple[np.ndarray,np.ndarray]: The strategy profile and the utility at Nash Equilibrium.
        """

        if isinstance(threshold, float):
            epsilon, under_epochs = threshold, 1
        else:
            epsilon, under_epochs = threshold

        if method == 'RELAX':
            optimizer = Relaxation(self, **opt_kwargs)
        else:
            raise ValueError(f'Invalid method {method}')

        x = np.ones((self.n, len(self.fee))) / (2. * len(self.fee)) * self.B[:, None]
        lastx = x.copy()

        tqdm_native = bar is None
        if tqdm_native:
            bar = tqdm(total=opt_kwargs.get('epochs', Game.MAX_EPOCHS[method]))
        else:
            bar.reset(total=opt_kwargs.get('epochs', Game.MAX_EPOCHS[method]))

        underthr, dynamics, success = 0, [], False

        window = deque(maxlen=500)
        window_size, pop_num = 500, 10

        for _ in range(bar.total):
            x = optimizer.update(x)
            stepdiff = np.max(abs(x-lastx) / self.B[:, None])
            vfunc = self.nikaido_isoda(x)
            if path_dynamics is not None:
                dynamics.append((stepdiff, vfunc, self.U(x)))

            bar.set_description(
                f'{self.name} | '
                f'N={self.n:<3} M={len(self.fee):<4} '
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

        return lastx, self.U(lastx)

    def Um(self, x):
        """
        Fine-Grained utility for each player at each price range given their strategy profile.

        Args:
            x (np.ndarray): A 2D array representing the strategy profile of all players.
                            The shape of x should be (n, m) where n is the number of players and m is the number of ranges.

        Returns:
            np.ndarray: A 2D array representing the fine-grained utility for each player.
                        The shape of the return array is also (n, m)
        """
        return self.fee * (x**self.alpha) / np.maximum(np.sum(x ** self.alpha, 0) + self.chi, 1e-10) - self.ilrate * x

    def Fm(self, x):
        """
        Computes the fee component of the utility for each player at each price range.

        Args:
            x (np.ndarray): A 2D array representing the strategy profile of all players.
                            The shape of x should be (n, m) where n is the number of players and m is the number of ranges.

        Returns:
            np.ndarray: A 2D array representing the fee component for each player.
                        The shape of the return array is also (n, m)
        """
        return self.fee * (x**self.alpha) / np.maximum(np.sum(x ** self.alpha, 0) + self.chi, 1e-10)

    def Cm(self, x):
        """
        Computes the cost component (impermanent loss) for each player at each price range given their strategy profile.

        Args:
            x (np.ndarray): A 2D array representing the strategy profile of all players.
                            The shape of x should be (n, m) where n is the number of players and m is the number of ranges.

        Returns:
            np.ndarray: A 2D array representing the impermanent loss for each player.
                        The shape of the return array is also (n, m)
        """
        return self.ilrate * x

    def U(self, x):
        """
        Computes the total utility for each player given the strategy profile.

        Args:
            x (np.ndarray): A 2D array representing the strategy profile of all players.
                            The shape of x should be (n, m) where n is the number of players and m is the number of strategies.

        Returns:
            np.ndarray: A 1D array of length n representing the total utility for each player.
        """
        return np.sum(self.Um(x), 1)

    def F(self, x):
        """
        Total fee of the utility for each player given the strategy profile.

        Args:
            x (np.ndarray): A 2D array representing the strategy profile of all players.
                            The shape of x should be (n, m) where n is the number of players and m is the number of strategies.

        Returns:
            np.ndarray: A 1D array of length n representing the total fee component for each player.
        """
        return np.sum(self.fee * (x**self.alpha) / np.maximum(np.sum(x ** self.alpha, 0) + self.chi, 1e-10), 1)

    def C(self, x):
        """
        Total cost (impermanent loss) for each player given the strategy profile.

        Args:
            x (np.ndarray): A 2D array representing the strategy profile of all players.
                            The shape of x should be (n, m) where n is the number of players and m is the number of strategies.

        Returns:
            np.ndarray: A 1D array of length n representing the total impermanent loss for each player.
        """
        return np.sum(self.ilrate * x, 1)

    def U_under(self, x, y, lp) -> float:
        """
        Computes the utility for a specific player given a replaced strategy.

        Args:
            x (np.ndarray): A 2D array representing the current strategy profile of all players.
                            The shape of x should be (n, m) where n is the number of players and m is the number of strategies.
            y (np.ndarray): A 1D array representing the replaced strategy for the specified player. \
                Replacing the player's current strategy x[lp] with y.
            lp (int): The index of the player for whom the utility is to be computed.

        Returns:
            float: The utility value for the specified player given the replaced strategy.
        """
        y = np.array(y)
        total_weight = np.maximum(1e-10, np.sum(x**self.alpha, 0) + self.chi + y**self.alpha - x[lp]**self.alpha)
        util = np.sum(self.fee * (y**self.alpha) / total_weight - self.ilrate * y)
        return util

    def _best_response_single(self, lp: int, denom: np.ndarray):
        """
        Computes the best response strategy for a single player in the game.

        Args:
            lp (int): The index of the player for whom the best response is to be computed.
            denom (np.ndarray): An array representing other LPs' shares, the denominator values used in the calculation,

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

    def best_response(self, x: np.ndarray, lp: int = None) -> np.ndarray:
        """
        Computes the best response strategy for the game.

        Args:
            x (np.ndarray): A 2D array representing the current strategy profile of all players.
                            The shape of x should be (n, m) where n is the number of players and m is the number of strategies.
            lp (int, optional): The index of the player for whom the best response is to be computed.
                                If None, computes the best response for all players. Default is None.

        Returns:
            np.ndarray: A 2D array representing the best response strategy. If lp is specified, returns the best response for that player.
                        Otherwise, returns the best response for all players.
        """
        assert np.shape(x) == (self.n, len(self.fee))
        denom = np.sum(x**self.alpha, 0) + self.chi
        if lp is None:
            y = np.zeros(np.shape(x))
            for lp in range(self.n):
                y[lp, :] = self._best_response_single(lp, denom-x[lp]**self.alpha)
        else:
            y = self._best_response_single(lp, denom-x[lp]**self.alpha)

        return y

    def nikaido_isoda(self, x, y=None):
        """
        Computes the Nikaido-Isoda function value, which measures the deviation from Nash Equilibrium.

        Args:
            x (np.ndarray): A 2D array representing the current strategy profile of all players.
                            The shape of x should be (n, m) where n is the number of players and m is the number of strategies.
            y (np.ndarray, optional): A 2D array representing the best response strategy profile. 
                                      If None, it is computed using the current strategy profile x.

        Returns:
            float: The mean utility difference between the current strategy profile and the best response strategy profile.
                   A value closer to zero indicates a strategy profile closer to Nash Equilibrium.
        """
        if y is None:
            y = self.best_response(x)

        udiff = [self.U_under(x, x[lp], lp) - self.U_under(x, y[lp], lp) for lp in range(self.n)]
        return np.mean(udiff)

    def hypothetical_response(self, x: np.ndarray, lps: int = None, added: float = 1.) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the hypothetical best response strategy and utility for specified players 
        when their budget is temporarily increased.

        Args:
            x (np.ndarray): A 2D array representing the current strategy profile of all players.
                            The shape of x should be (n, m) where n is the number of players and m is the number of strategies.
            lps (int, optional): The indices of the players for whom the hypothetical response is to be computed.
                                 If None, computes for all players. Default is None.
            added (float, optional): The amount by which the budget of the specified players is temporarily increased. Default is 1.

        Returns:
            Tuple[np.ndarray,np.ndarray]: A tuple containing:
                - A 2D array representing the hypothetical best response strategy for the specified players.
                - A 1D array representing the hypothetical utility for the specified players.
        """
        x = np.array(x)
        assert np.shape(x) == (self.n, len(self.fee))

        if lps is None:
            lps = np.arange(self.n, dtype=int)

        denom = np.sum(x**self.alpha, 0) + self.chi
        us = []
        for lp in lps:
            self.B[lp], temp = self.B[lp] + added, self.B[lp]
            y = self._best_response_single(lp, denom-x[lp, :]**self.alpha)
            self.B[lp] = temp
            us.append(self.U_under(x, y, lp))

        return y[lps, :], np.array(us)

    def plot_strategy(self, strategy: np.ndarray, fig=None, axs: List[plt.Axes] = None):
        """
        Plots the strategy profile of the game, including the fee and cost components.

        Args:
            strategy (np.ndarray): A 2D array representing the strategy profile of all players.
                                   The shape of the array should be (n, m) where n is the number of players
                                   and m is the number of strategies.
            fig (matplotlib.figure.Figure, optional): A matplotlib figure object to use for plotting.
                                                      If None, a new figure is created. Default is None.
            axs (List[plt.Axes], optional): An array of matplotlib axes objects to use for plotting.
                                        If None, new axes are created. Default is None.

        Returns:
            None: This function does not return any value. It generates a plot of the strategy profile.
        """
        plot_strategy_util(self.B, strategy, self.Fm(strategy), self.Cm(strategy), None, None, None, fig=fig, axs=axs)
