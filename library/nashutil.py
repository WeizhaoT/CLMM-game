import json
import numpy as np
import pandas as pd

from typing import List, Dict, Tuple
from tqdm import tqdm
from library import price_est_pdf, Bars,  OverdueError, single_range_br, Game, LiqConverter, filter_budget
from library.bars import align_bars


class Intelligence(LiqConverter):
    NATIVE = ['BR', 'NE']
    OTHERS = {
        'YDay': 'tick_pall',
        'R_BR': 'tick_prev',
        'R_NE': 'tick_prev',
        'I_BR': 'tick_inert',
        'I_NE': 'tick_inert',
    }

    def __init__(self, infopath: str, reloadpath: str, csvpath: str, raw: bool, max_players: int, date: str, prog: tqdm = None):
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
        for lp, value in zip(self.P, values):
            self.add(lp, category, key, value)

    def add_same(self, category, key, value):
        for lp in self.P:
            self.add(lp, category, key, value)

    def add_global(self, key, value):
        self.cargo[f'__{key}__'] = value

    def lacks(self, key):
        return any(lp not in self.cargo or key not in self.cargo[lp] for lp in self.P)

    def pb_iter(self):
        return iter(zip(self.P, self.B))

    def utility(self, liq_weights: Bars, liq_action: Bars):
        liq_weights, liq_action = align_bars(liq_weights, liq_action, standard=0)
        fee_x = self.fee_x.resample_fee(liq_action.T, True)
        fee_y = self.fee_y.resample_fee(liq_action.T, False)
        fee_total = sum((fx * self.x1 + fy * self.y1) * a / np.maximum(1e-14, a+w)
                        for fx, fy, a, w in zip(fee_x.V, fee_y.V, liq_action, liq_weights))
        return fee_total - liq_action.impermanent_loss(self.q0, self.p1)

    def overlap(self, lp, action, category='GT'):
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
        br_strategy = self.G.best_response(self.gt_arr)
        self.add_all('BR', 'action', br_strategy)

    def nash_equilibrium(self, gamma):
        ne, ne_util = ne_solve(self.G, retry=4, gamma=gamma, tuning=5.0, bar=self.prog)
        if ne is None:
            print(f'Warning: {self.G.name} did not complete')
        else:
            ne[np.where(np.isclose(ne, 0, atol=1e-6))] = 0
            self.add_all('NE', 'action', ne)
            self.add_all('NE', 'ne_util', ne_util)

    def responsive_yday(self, yday_actions):
        for lp in self.P:
            if lp in yday_actions:
                yday_budget = sum(yday_actions[lp])
                self.add(lp, 'YDay', 'action', [a * self.BMap[lp] / yday_budget for a in yday_actions[lp]])

    def responsive_br(self, par_prev):
        orig_prev, others = par_prev.pop('actions'), par_prev['B'].copy()
        for lp in self.P:
            budget_dict_temp = others.copy() | self.BMap
            lp_prev, par_prev['B'] = zip(*sorted(budget_dict_temp.items(), key=lambda x: -x[1]))
            game_prev = Game(**par_prev)
            strategy = np.array([orig_prev.get(h, np.zeros((len(par_prev['rewards']),))) for h in lp_prev])
            self.add(lp, 'R_BR', 'action', game_prev.best_response(strategy, lp_prev.index(lp)))

    def responsive_ne(self, par_prev, gamma):
        game_run = Game(**(par_prev | {'B': self.B} | {'name': f'{self.perc*100:>5.1f}%'}))
        ne_run, ne_util = ne_solve(game_run, retry=4, gamma=gamma, tuning=5.0, bar=self.prog)
        if ne_run is None:
            print(f'Warning: {game_run.name} did not complete')
        else:
            ne_run[np.where(np.isclose(ne_run, 0, atol=1e-6))] = 0
            self.add_all('R_NE', 'action', ne_run)
            self.add_all('R_NE', 'ne_util', ne_util)

    def inert_summary(self, priors: List[dict], Rhi=1.5, Rlo=None):
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
        for i, lp in enumerate(self.P):
            wbar = Bars(self.T, self.WP_arr[i])
            wbar = wbar.align(p['ticks'], force_bounds=True)
            wbar.apply(value_map=lambda x: x+p['jit'])
            max_liq = self.to_liq(Bars(p['ticks'], [self.B[i]])).V[0]
            act = self.to_cash(Bars(p['ticks'], [single_range_br(p['fee_x'], p['fee_y'], p['il'], max_liq, wbar)]))
            self.add(lp, 'I_BR', 'action', list(act.V))

    def inert_ne(self, p, gamma):
        game = Game(1., self.B, p['ticks'], [p['fee_usd']], [p['jit']], p['p'],
                    p['pnew'], p['probs'], name=f'{self.perc*100:>5.1f}%')
        ne, ne_util = ne_solve(game, retry=4, gamma=gamma, tuning=5.0, bar=self.prog)
        if ne is None:
            print(f'Warning: {game.name} did not complete')
        else:
            self.add_all('I_NE', 'action', ne)
            self.add_all('I_NE', 'ne_util', ne_util)

    def report_ibr_stats(self, invariants=False):
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
    for _ in range(retry):
        try:
            return game.find_ne(mode='p', method='ELS', threshold=threshold, return_uty=True, bar=bar, verbose=True, gamma=gamma)
        except OverdueError:
            gamma /= tuning

    return None, None


def crawl_liq(rows, ticks_int: List[int]) -> Bars:
    delta = [(row['tickLower'], row['liquidity']) for _, row in rows.iterrows()] + \
        [(row['tickUpper'], -row['liquidity']) for _, row in rows.iterrows()]
    bar: Bars = Bars.from_delta(delta, sort=True)
    bar = bar.align(ticks_int)
    assert len(bar.T) == len(ticks_int)
    return bar
