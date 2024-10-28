import json
import numpy as np
import pandas as pd

from typing import List, Dict, Tuple
from tqdm import tqdm
from library import price_est_pdf, Bars,  OverdueError, single_range_br, Game, LiqConverter, filter_budget
from library.bars import align_bars


class Intelligence(LiqConverter):
    NATIVE = ['BR', 'NE']
    """ Strategies from native game """
    OTHERS = {
        'YDay': 'tick_pall',
        'R_BR': 'tick_prev',
        'R_NE': 'tick_prev',
        'I_BR': 'tick_inert',
        'I_NE': 'tick_inert',
    }
    """ Strategy -> tickname map """

    def __init__(self, infopath: str, reloadpath: str, csvpath: str, raw: bool, max_players: int, date: str, prog: tqdm = None):
        """
        Initializes an instance of the Intelligence class, setting up various parameters and data structures
        for analyzing liquidity provider actions and game strategies.

        Args:
            infopath (str): Path to the JSON file containing initial information and parameters.
            reloadpath (str): Path to a JSON file for reloading previous state data. If provided, max_players must be None.
            csvpath (str): Path to a CSV file containing ground truth data. Required if reloadpath is not provided.
            raw (bool): Flag indicating whether to use raw data for calculations.
            max_players (int): Maximum number of players to consider. Required if reloadpath is not provided.
            date (str): The date for which the analysis is being conducted.
            prog (tqdm, optional): A tqdm progress bar instance for tracking progress. Defaults to None.

        Raises:
            AssertionError: If csvpath is not provided when reloadpath is not specified.
        """
        with open(infopath, 'r') as f:
            self.info = json.load(f)

        self.date: str = date
        self.offset: int = int(np.round(np.log10(self.info['tick'][0] / np.power(1.0001, self.info['tick_orig'][0]))))

        fee_now = Bars(self.info['tick'], self.info['total_fee_raw'] if raw else self.info['total_fee'])
        self.JL, self.JU = fee_now.margin_idxs()
        self.fee_usd: Bars = fee_now.truncate(self.JL, self.JU)
        self.T: List[float] = self.fee_usd.T

        self.fee_x: Bars = Bars(self.info['tick'], self.info['total_fee_x']).truncate(self.JL, self. JU)
        self.fee_y: Bars = Bars(self.info['tick'], self.info['total_fee_y']).truncate(self.JL, self. JU)

        self.q0, self.q1 = self.info['q'], self.info['qnew']
        self.x0, self.x1 = self.info['usd0_raw'] if raw else self.info['usd0']
        self.y0, self.y1 = self.info['usd1_raw'] if raw else self.info['usd1']

        jit = self.info['jit_raw'] if raw else self.info['jit']

        self.p0: Tuple[float, float, float] = (self.q0, self.x0, self.y0)
        self.p1: Tuple[float, float, float] = (self.q1, self.x1, self.y1)
        super().__init__(*self.p0)

        if reloadpath and max_players is None:
            with open(reloadpath, 'r') as f:
                self.cargo = json.load(f)
            self.P = [key for key in self.cargo.keys() if not key.startswith('__')]
            NP = list(self.info['budget'].keys())[len(self.P):]
            self.B = [self.cargo[key]['__budget__'] for key in self.P]
            self.perc = sum(self.B) / sum(self.info['budget'].values())
        else:
            assert csvpath, "Must provide a valid CSV path when not reloading."
            self.P, NP, self.B, self.perc = filter_budget(self.info['budget'], .99, max_players=max_players)

            self.cargo = {lp: {'__budget__': b} for lp, b in self.pb_iter()} | \
                {'__percentage__': self.perc,
                 '__range_count__': len(self.fee_usd),
                 '__range_jit__': int(sum(np.greater(jit[self.JL:self.JU], 1e9))),
                 '__tick_all__': list(self.info['tick']),
                 '__tick__': list(self.T), }

        self.BMap: Dict[str, float] = {lp: b for lp, b in self.pb_iter()}
        # Read ground truth and update JIT
        self.gt_arr, liq_weight = np.zeros((2, len(self.B), len(self.fee_usd)))
        self.gt_bar: Dict[str, Bars] = {}
        self.prog = prog

        native_tqdm = prog is None and csvpath
        if csvpath:
            if native_tqdm:
                prog = tqdm(total=len(self.info['budget']), desc='Parsing GT')
            else:
                prog.reset(total=len(self.info['budget']))
                prog.set_description('Parsing GT')

            csv = pd.read_csv(csvpath, index_col='LP')
            for i, lp in enumerate(self.P):
                bar = crawl_liq(csv.loc[[lp]], self.info['tick_orig'])
                bar.apply(tick_map=lambda x: (10**self.offset) * (1.0001**x))
                liq_weight[i] = bar[self.JL:self.JU]
                self.gt_bar[lp] = bar = self.to_cash(bar)
                self.gt_arr[i] = bar[self.JL:self.JU]
                self.cargo[lp] |= {'GT': {'action': list(self.gt_arr[i]),  'action_data': list(bar.V),
                                          'util_data': sum(csv.loc[[lp]]['utility']),
                                          'fee_data': sum(csv.loc[[lp]]['fee']),
                                          'cost_data': sum(csv.loc[[lp]]['IL']),
                                          'position_count': len(csv.loc[[lp]]), 'olap_gt': 1., }}
                prog.update()

            self.jit = np.array(jit[self.JL:self.JU])
            for lp in NP:
                bar = crawl_liq(csv.loc[[lp]], self.info['tick_orig'])
                bar.apply(tick_map=lambda x: (10**self.offset) * (1.0001**x))
                bar = self.to_cash(bar)
                self.jit += np.array(bar.V)[self.JL:self.JU]
                prog.update()

            self.cargo['__jit__'] = list(self.jit)
        else:
            self.jit = np.array(self.cargo['__jit__'])
            for i, lp in enumerate(self.P):
                self.gt_bar[lp] = bar = Bars(self.info['tick'], self.cargo[lp]['GT']['action_data'])
                self.gt_arr[i] = bar[self.JL:self.JU]
                liq_weight[i] = self.to_liq(bar)[self.JL:self.JU]

        self.G = Game(1.0, self.B, self.T, self.fee_usd.V, self.jit, self.p0, self.p1, name=f'{self.perc*100:>5.1f}%')
        self.cargo['__tau__'] = list(self.G.ilrate)

        self.WP_arr = np.sum(liq_weight, 0) - liq_weight
        jit_liq = self.to_liq(Bars(self.T, self.jit)).V
        self.WA = {lp: Bars(self.T, self.WP_arr[i] + jit_liq) for i, lp in enumerate(self.P)}

        fee_derived, cost_derived = self.G.F(self.gt_arr), self.G.C(self.gt_arr)
        self.add_all('GT', 'util', fee_derived - cost_derived)
        self.add_all('GT', 'fee', fee_derived)
        self.add_all('GT', 'cost', cost_derived)
        if native_tqdm:
            prog.close()

    def export(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.cargo, f, indent=2)

    def game_param(self):
        qnew, probs = price_est_pdf(self.x1, self.y1, .05, .05)
        return {
            'beta': 1.,
            'B': self.BMap,
            'actions': {lp: act for lp, act in zip(self.P, self.gt_arr)},
            'ticks': self.T,
            'ticks_all': list(self.info['tick']),
            'rewards': self.fee_usd.V,
            'jit': self.jit,
            'q': self.p1,
            'qnew': qnew,
            'qnew_probs': probs,
        }

    def inert_params(self):
        """
        Retrieves inert game parameters

        Returns:
            dict: A dictionary containing the following inertial parameters:
                - 'B': A mapping of liquidity providers to their budgets.
                - 'p0': A tuple representing the initial state parameters (q0, x0, y0).
                - 'p1': A tuple representing the new state parameters (q1, x1, y1).
                - 'fee': A Bars object representing the total fee in USD.
                - 'fee_x': A Bars object representing the fee structure in the x direction.
                - 'fee_y': A Bars object representing the fee structure in the y direction.
                - 'jit': An array representing the jitter values.
                - 'actions': A dictionary mapping each liquidity provider to their respective actions.
        """
        return {
            'B': self.BMap,
            'p0': self.p0,
            'p1': self.p1,
            'fee': self.fee_usd,
            'fee_x': self.fee_x,
            'fee_y': self.fee_y,
            'jit': self.jit,
            'actions': {lp: act for lp, act in zip(self.P, self.gt_arr)},
        }

    def add(self, lp, category, key, value):
        """
        Adds a value to the cargo dictionary for a specified liquidity provider (LP) under a given category and key.

        Args:
            lp (str): The identifier for the liquidity provider.
            category (str): The category under which the value should be stored.
            key (str): The key within the category under which the value should be stored.
            value: The value to be stored. It can be any data type, and if iterable, it will be converted to a list.

        Returns:
            None: This function updates the internal state of the cargo dictionary and does not return any value.
        """
        try:
            iter(value)
            value = list(value)
        except:
            pass
        if lp not in self.cargo:
            self.cargo[lp] = {category: {key: value}}
        elif category not in self.cargo[lp]:
            self.cargo[lp][category] = {key: value}
        else:
            self.cargo[lp][category][key] = value

    def add_all(self, category, key, values):
        """
        Adds a set of values to the cargo dictionary for all liquidity providers (LPs) under a specified category and key.

        Args:
            category (str): The category under which the values should be stored for each LP.
            key (str): The key within the category under which the values should be stored.
            values (iterable): An iterable of values to be stored, with each value corresponding to an LP in self.P.

        Returns:
            None: This function updates the internal state of the cargo dictionary and does not return any value.
        """
        for lp, value in zip(self.P, values):
            self.add(lp, category, key, value)

    def add_same(self, category, key, value):
        """
        Adds the same value to the cargo dictionary for all liquidity providers (LPs) under a specified category and key.

        Args:
            category (str): The category under which the value should be stored for each LP.
            key (str): The key within the category under which the value should be stored.
            value: The value to be stored. It can be any data type, and if iterable, it will be converted to a list.

        Returns:
            None: This function updates the internal state of the cargo dictionary and does not return any value.
        """
        for lp in self.P:
            self.add(lp, category, key, value)

    def add_global(self, key, value):
        """
        Adds a global value to the cargo dictionary under a specified key.

        Args:
            key (str): The key under which the value should be stored, prefixed with '__' in the cargo dictionary.
            value: The value to be stored. It can be any data type.

        Returns:
            None: This function updates the internal state of the cargo dictionary and does not return any value.
        """
        self.cargo[f'__{key}__'] = value

    def lacks(self, key):
        """
        Checks if any liquidity provider (LP) in the current set lacks a specified key in their cargo data.

        Args:
            key (str): The key to check for in each LP's cargo data.

        Returns:
            bool: True if any LP lacks the specified key in their cargo data, False otherwise.
        """
        return any(lp not in self.cargo or key not in self.cargo[lp] for lp in self.P)

    def pb_iter(self):
        return iter(zip(self.P, self.B))

    def utility(self, liq_weights: Bars, liq_action: Bars):
        """
        Calculates the utility of a given liquidity action based on the provided liquidity weights.

        Args:
            liq_weights (Bars): A Bars object representing the weights of liquidity across different ticks.
            liq_action (Bars): A Bars object representing the liquidity action to be evaluated.

        Returns:
            float: The calculated utility value, which is the total fee derived from the action minus the impermanent loss.
        """
        liq_weights, liq_action = align_bars(liq_weights, liq_action, standard=0)
        fee_x = self.fee_x.resample_fee(liq_action.T, True)
        fee_y = self.fee_y.resample_fee(liq_action.T, False)
        fee_total = sum((fx * self.x1 + fy * self.y1) * a / np.maximum(1e-14, a+w)
                        for fx, fy, a, w in zip(fee_x.V, fee_y.V, liq_action, liq_weights))
        return fee_total - liq_action.impermanent_loss(self.q0, self.p1)

    def overlap(self, lp, action, category='GT'):
        """
        Calculates the overlap between a given action and a baseline for a specified liquidity provider (LP).

        Args:
            lp (str): The identifier for the liquidity provider.
            action (Bars or list): The action to compare against the baseline. Can be a Bars object or a list.
            category (str, optional): The category of the baseline to compare against. Defaults to 'GT'. 
                                      Must be one of the categories defined in Intelligence.NATIVE if not 'GT'.

        Returns:
            float: The calculated overlap value between the action and the baseline.
        """
        if isinstance(action, Bars):
            baseline = self.gt_bar[lp] if category == 'GT' else Bars(self.T, self.cargo[lp][category]['action'])
            return self.overlap_general(self.BMap[lp], action, baseline)

        if category == 'GT':
            return self.gt_bar[lp].overlap(action, self.BMap[lp], self.JL, self.JU)
        else:
            assert category in Intelligence.NATIVE
            baseline = Bars(self.T, self.cargo[lp][category]['action'])
            return baseline.overlap(action, self.BMap[lp])

    def derive(self, strategy=None):
        """
        Derives utility and overlap metrics for each liquidity provider (LP) based on specified strategies.

        Args:
            strategy (Optional[Union[str, List[str]]]): A strategy or list of strategies to consider. If None, all 
            strategies are considered. Strategies can be from the native set ('BR', 'NE') or others defined in the 
            Intelligence class.

        Returns:
            None: The function updates the internal state of the object by adding utility and overlap metrics for each 
            LP and strategy.
        """
        if isinstance(strategy, str):
            strategy = [strategy]
        if strategy is not None:
            strategy = set(strategy)
        for i, lp in enumerate(self.P):
            for category in Intelligence.NATIVE:
                if strategy is None or category in strategy:
                    if category in self.cargo[lp]:
                        action = self.cargo[lp][category]['action']
                        util = self.G.U_under(self.gt_arr, action, i)
                        self.add(lp, category, 'util', util)
                        self.add(lp, category, 'olap_gt', self.overlap(lp, action, category='GT'))
                        if category != 'BR':
                            self.add(lp, category, 'olap_br', self.overlap(lp, action, category='BR'))
                    else:
                        print(f'Warning: {lp} does not have action under {category}')

            for category, tickname in Intelligence.OTHERS.items():
                if strategy is None or category in strategy:
                    if category in self.cargo[lp]:
                        action = Bars(self.cargo[f'__{tickname}__'], self.cargo[lp][category]['action'])
                        util = self.utility(self.WA[lp], self.to_liq(action))
                        self.add(lp, category, 'util', util)
                        self.add(lp, category, 'olap_gt', self.overlap(lp, action, category='GT'))
                        self.add(lp, category, 'olap_br', self.overlap(lp, action, category='BR'))

    def best_response(self):
        """
        Computes the best response strategy for the current game state and updates the cargo with the strategy.

        This function calculates the best response strategy for each liquidity provider (LP) based on the current
        game state and the ground truth actions. It then updates the internal cargo dictionary with the computed
        best response strategy for each LP under the 'BR' category.

        Returns:
            None: This function updates the internal state of the object and does not return any value.
        """
        br_strategy = self.G.best_response(self.gt_arr)
        self.add_all('BR', 'action', br_strategy)

    def nash_equilibrium(self, gamma):
        """
        Computes the Nash Equilibrium (NE) for the current game state and updates the cargo with the NE strategy.

        Args:
            gamma (float): The gamma value used in the NE search algorithm, influencing the convergence behavior.

        Returns:
            None: This function updates the internal state of the object by adding the NE strategy and its utility
            for each liquidity provider (LP) under the 'NE' category.
        """
        ne, ne_util = ne_solve(self.G, retry=4, gamma=gamma, tuning=5.0, bar=self.prog)
        if ne is None:
            print(f'Warning: {self.G.name} did not complete')
        else:
            ne[np.where(np.isclose(ne, 0, atol=1e-6))] = 0
            self.add_all('NE', 'action', ne)
            self.add_all('NE', 'ne_util', ne_util)

    def responsive_yday(self, yday_actions):
        """
        Updates the cargo with actions from the previous day for each liquidity provider (LP).

        Args:
            yday_actions (dict): A dictionary where keys are LP identifiers and values are lists of actions 
                                 taken by the LP on the previous day.

        Returns:
            None: This function updates the internal state of the cargo dictionary with the previous day's 
            actions for each LP under the 'YDay' category.
        """
        for lp in self.P:
            if lp in yday_actions:
                yday_budget = sum(yday_actions[lp])
                self.add(lp, 'YDay', 'action', [a * self.BMap[lp] / yday_budget for a in yday_actions[lp]])

    def responsive_br(self, par_prev):
        """
        Updates the cargo with the best response strategy for each liquidity provider (LP) based on previous parameters.

        Args:
            par_prev (dict): A dictionary containing previous game parameters, including 'actions' and 'B' (budgets).

        Returns:
            None: This function updates the internal state of the cargo dictionary with the best response strategy
            for each LP under the 'R_BR' category.
        """
        orig_prev, others = par_prev.pop('actions'), par_prev['B'].copy()
        for lp in self.P:
            budget_dict_temp = others.copy() | self.BMap
            lp_prev, par_prev['B'] = zip(*sorted(budget_dict_temp.items(), key=lambda x: -x[1]))
            game_prev = Game(**par_prev)
            strategy = np.array([orig_prev.get(h, np.zeros((len(par_prev['rewards']),))) for h in lp_prev])
            self.add(lp, 'R_BR', 'action', game_prev.best_response(strategy, lp_prev.index(lp)))

    def responsive_ne(self, par_prev, gamma):
        """
        Computes the responsive Nash Equilibrium (NE) for the current game state using previous parameters
        and updates the cargo with the NE strategy.

        Args:
            par_prev (dict): A dictionary containing previous game parameters, which are used to initialize
                             the game for NE computation. It should include keys relevant to the game setup.
            gamma (float): The gamma value used in the NE search algorithm, influencing the convergence behavior.

        Returns:
            None: This function updates the internal state of the object by adding the NE strategy and its utility
            for each liquidity provider (LP) under the 'R_NE' category.
        """
        game_run = Game(**(par_prev | {'B': self.B} | {'name': f'{self.perc*100:>5.1f}%'}))
        ne_run, ne_util = ne_solve(game_run, retry=4, gamma=gamma, tuning=5.0, bar=self.prog)
        if ne_run is None:
            print(f'Warning: {game_run.name} did not complete')
        else:
            ne_run[np.where(np.isclose(ne_run, 0, atol=1e-6))] = 0
            self.add_all('R_NE', 'action', ne_run)
            self.add_all('R_NE', 'ne_util', ne_util)

    def inert_summary(self, priors: List[dict], Rhi=1.5, Rlo=None):
        """
        Summarizes inertial parameters based on prior data and expansion ratios.

        Args:
            priors (List[dict]): A list of dictionaries containing prior data, each with keys such as 'fee', 'fee_x', 
                                 'fee_y', 'jit', 'p0', and 'p1'.
            Rhi (float, optional): The high expansion ratio for tick range adjustment. Defaults to 1.5.
            Rlo (float, optional): The low expansion ratio for tick range adjustment. If None, defaults to Rhi.

        Returns:
            dict: A dictionary containing the summarized inertial parameters, including average fees, jit, tick range, 
                  initial and new price estimates, probabilities, and impermanent loss.
        """
        if Rlo is None:
            Rlo = Rhi

        self.add_global('inert_expansion', [Rlo, Rhi])
        all_fee_usd, all_fee_x, all_fee_y, all_jits = [], [], [], []
        x_changes, y_changes = [], []
        qlo, qhi = float('inf'), float('-inf')
        p_start = priors[0]['p1']

        for p in priors:
            all_fee_usd.append(sum(p['fee'].V))
            all_fee_x.append(sum(p['fee_x'].V) * p['p1'][1])
            all_fee_y.append(sum(p['fee_y'].V) * p['p1'][2])
            all_jits.append(sum(j for j in p['jit'] if j < 1e9))
            lo, hi = p['fee'].margins()
            qlo, qhi = min(qlo, lo), max(qhi, hi)
            kx, ky = [r / s for r, s in zip(p['p0'][1:], p['p1'][1:])]
            x_changes.append(max(kx, 1/kx))
            y_changes.append(max(ky, 1/ky))

        pnew, probs = price_est_pdf(p_start[1], p_start[2], np.mean(x_changes)-1, np.mean(y_changes)-1)

        il = 0
        for p, pi in zip(pnew, probs):
            r_start = min(qhi, max(qlo, p_start[0]))
            r = min(qhi, max(qlo, p[0]))
            il += pi * (p[1] * (r_start**-.5 - r**-.5) + p[2] * (r_start**.5-r**.5))

        return {
            'fee_usd': np.mean(all_fee_usd),
            'fee_x': np.mean(all_fee_x),
            'fee_y': np.mean(all_fee_y),
            'jit': np.mean(all_jits),
            'ticks': [qlo / Rlo, qhi * Rhi],
            'p': p_start,
            'pnew': pnew,
            'probs': probs,
            'il': il,
        }

    def inert_br(self, p):
        """
        Computes and updates the best response strategy for each liquidity provider (LP) based on inertial parameters.

        Args:
            p (dict): A dictionary containing inertial parameters, including:
                      - 'ticks': The tick range for alignment.
                      - 'jit': The jitter value to be added to the liquidity.
                      - 'fee_x': The fee structure in the x direction.
                      - 'fee_y': The fee structure in the y direction.
                      - 'il': The impermanent loss value.

        Returns:
            None: This function updates the internal state of the cargo dictionary with the best response strategy
            for each LP under the 'I_BR' category.
        """
        for i, lp in enumerate(self.P):
            wbar = Bars(self.T, self.WP_arr[i])
            wbar = wbar.align(p['ticks'], force_bounds=True)
            wbar.apply(value_map=lambda x: x+p['jit'])
            max_liq = self.to_liq(Bars(p['ticks'], [self.B[i]])).V[0]
            act = self.to_cash(Bars(p['ticks'], [single_range_br(p['fee_x'], p['fee_y'], p['il'], max_liq, wbar)]))
            self.add(lp, 'I_BR', 'action', list(act.V))

    def inert_ne(self, p, gamma):
        """
        Computes the Nash Equilibrium (NE) for the current game state using inertial parameters
        and updates the cargo with the NE strategy.

        Args:
            p (dict): A dictionary containing inertial parameters, including:
                      - 'ticks': The tick range for alignment.
                      - 'fee_usd': The fee structure in USD.
                      - 'jit': The jitter value to be added to the liquidity.
                      - 'p': The initial price estimates.
                      - 'pnew': The new price estimates.
                      - 'probs': The probabilities associated with the new price estimates.
            gamma (float): The gamma value used in the NE search algorithm, influencing the convergence behavior.

        Returns:
            None: This function updates the internal state of the object by adding the NE strategy and its utility
            for each liquidity provider (LP) under the 'I_NE' category.
        """
        game = Game(1., self.B, p['ticks'], [p['fee_usd']], [p['jit']], p['p'],
                    p['pnew'], p['probs'], name=f'{self.perc*100:>5.1f}%')
        ne, ne_util = ne_solve(game, retry=4, gamma=gamma, tuning=5.0, bar=self.prog)
        if ne is None:
            print(f'Warning: {game.name} did not complete')
        else:
            self.add_all('I_NE', 'action', ne)
            self.add_all('I_NE', 'ne_util', ne_util)

    def report_ibr_stats(self, invariants=False):
        """
        Reports statistics related to the Inertial Best Response (I_BR) strategy for each liquidity provider (LP).

        Args:
            invariants (bool, optional): If True, additional invariant statistics are included in the report. 
                                         Defaults to False.

        Returns:
            tuple: A tuple containing lists of statistics. If `invariants` is False, the tuple contains:
                   - utils (list): Utilities of the I_BR strategy for each LP.
                   - olaps (list): Overlaps of the I_BR strategy with the ground truth for each LP.
                   If `invariants` is True, the tuple additionally includes:
                   - gt_utils (list): Utilities of the ground truth strategy for each LP.
                   - ne_olaps (list): Overlaps of the Nash Equilibrium strategy with the ground truth for each LP.
                   - lazy (list): Boolean indicators of whether the previous day's overlap is greater than 0.95 for each LP.
        """
        gt_utils, utils, olaps, ne_olaps, lazy = [[] for _ in range(5)]
        for lp in self.P:
            if "I_BR" in self.cargo[lp]:
                utils.append(self.cargo[lp]['I_BR']['util'])
                olaps.append(self.cargo[lp]['I_BR']['olap_gt'])
                if invariants:
                    gt_utils.append(self.cargo[lp]['GT']['util'])
                    ne_olaps.append(self.cargo[lp]['NE']['olap_gt'])
                    lazy.append("YDay" in self.cargo[lp] and self.cargo[lp]["YDay"]['olap_gt'] > .95)
        return (utils, olaps, gt_utils, ne_olaps, lazy) if invariants else (utils, olaps)


def ne_solve(game: Game, threshold=(1e-6, 5), retry=4, gamma=1e-1, tuning=5.0, bar: tqdm = None):
    """
    Attempts to find the Nash Equilibrium (NE) for a given game using specified parameters.

    Args:
        game (Game): The game object for which the NE is to be found.
        threshold (tuple): A tuple specifying the convergence threshold for the NE search.
        retry (int): The number of retry attempts if the NE search fails due to an OverdueError.
        gamma (float): The initial gamma value used in the NE search algorithm.
        tuning (float): The factor by which gamma is adjusted upon encountering an OverdueError.
        bar (tqdm, optional): A tqdm progress bar object for visualizing progress, if applicable.

    Returns:
        tuple: A tuple containing the NE strategy and its utility if successful, otherwise (None, None).
    """
    for _ in range(retry):
        try:
            return game.find_ne(mode='p', method='RELAX', threshold=threshold, bar=bar, verbose=True, gamma=gamma)
        except OverdueError:
            gamma /= tuning

    return None, None


def crawl_liq(rows, ticks_int: List[int]) -> Bars:
    """
    Constructs a Bars object representing liquidity changes over specified tick intervals.

    Args:
        rows: A DataFrame-like object where each row contains 'tickLower', 'tickUpper', and 'liquidity' values.
        ticks_int: A list of integers representing the tick intervals to align the Bars object with.

    Returns:
        Bars: A Bars object aligned with the specified tick intervals, representing the liquidity changes.
    """
    delta = [(row['tickLower'], row['liquidity']) for _, row in rows.iterrows()] + \
        [(row['tickUpper'], -row['liquidity']) for _, row in rows.iterrows()]
    bar: Bars = Bars.from_delta(delta, sort=True)
    bar = bar.align(ticks_int)
    assert len(bar.T) == len(ticks_int)
    return bar
