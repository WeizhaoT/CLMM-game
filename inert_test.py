import os
import argparse
import numpy as np
import json
import pickle

from tqdm import tqdm
from collections import deque
from library import ensure_path, sing, Intelligence


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=int, required=True)
    parser.add_argument('-s', '--skip-prob', type=float, default=0)
    parser.add_argument('-u', type=float, nargs='+', required=True)
    parser.add_argument('-l', type=float, nargs='+', required=False)
    parser.add_argument('-a', action='store_true')
    args = parser.parse_args()

    PID = args.p
    upper = sorted(args.u)
    utils, olaps = [[[] for _ in range(len(upper))] for _ in range(2)]
    gt_utils, ne_olaps, lazies = [[] for _ in range(3)]

    DATA_PATH, AXV_PATH = f'dynamics/{PID}-archive', f'game/{PID}-archive'
    priors = deque([], maxlen=7)

    ensure_path(AXV_PATH, empty=False)

    files = sorted([file for file in os.listdir(DATA_PATH) if file.startswith('gamedata-') and file.endswith('.json')])
    par_prev, gt_prev = {}, {}

    bar_hi = tqdm(total=len(files), position=0)

    for file in files:
        date = file[9:-5]
        bar_hi.set_description(f'{date} {"loading":<9}')
        if date < '24-01-01' or not os.path.exists(f'{DATA_PATH}/profit-{date}.csv') or np.random.random() < args.skip_prob:
            bar_hi.update()
            continue

        reload_path = f'{AXV_PATH}/nash-{date}.json'
        LIB = Intelligence(f'{DATA_PATH}/{file}', reload_path, '', None, date)

        if len(LIB.B) == 0:
            par_prev, gt_prev = {}, {}
            priors.clear()
            bar_hi.update()
            continue

        if len(priors) > 0:
            bar_hi.set_description(f'{date} {"I_BR":<9}')
            for i, expansion in enumerate(upper):
                inert_params = LIB.inert_summary(priors, expansion)
                LIB.add_global('tick_inert', list(inert_params['ticks']))
                LIB.inert_br(inert_params)
                LIB.derive()

                if i == 0:
                    u, ov, u_gt, ov_ne, lazy = LIB.report_ibr_stats(invariants=True)
                    gt_utils.extend(u_gt)
                    ne_olaps.extend(ov_ne)
                    lazies.extend(lazy)
                else:
                    u, ov = LIB.report_ibr_stats(invariants=False)

                utils[i].extend(u)
                olaps[i].extend(ov)

        par_prev = LIB.game_param()
        gt_prev = LIB.gt_bar
        priors.appendleft(LIB.inert_params())
        bar_hi.set_description(f'{date} finished')
        bar_hi.update()

    bar_hi.close()

    ensure_path('essence', empty=False)
    jsonpath, filepath = f'essence/I_BR-{PID}.json', f'essence/I_BR-{PID}.pkl'
    if args.a and os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)

        utils.extend(obj['utils'])
        olaps.extend(obj['olaps'])

        i, idx = 0, []
        upper = [(e, -j) for j, e in enumerate(upper)]
        upper.extend([(e, j) for j, e in enumerate(obj['r'], start=len(upper))])
        r_sorted = sorted(upper)

        while i < len(r_sorted):
            e = r_sorted[i][0]
            while i+1 < len(r_sorted) and np.isclose(r_sorted[i+1][0], e, rtol=1e-6, atol=1e-6):
                i += 1

            idx.append(abs(r_sorted[i][1]))
            i += 1

        upper = [upper[i][0] for i in idx]
        utils = [utils[i] for i in idx]
        olaps = [olaps[i] for i in idx]

    with open(filepath, 'wb') as f:
        pickle.dump({
            'r': upper,
            'utils': utils,
            'olaps': olaps,
            'gt_utils': gt_utils,
            'ne_olaps': ne_olaps,
            'lazies': lazies,
        }, f)

    with open(jsonpath, 'w') as f:
        json.dump({
            'r': upper,
            'utils': list(map(np.mean, utils)),
            'olaps': list(map(np.mean, olaps)),
            'gt_utils': np.mean(gt_utils),
            'ne_olaps': np.mean(ne_olaps)
        }, f, indent=4)

    print(f"{'R':>8}{'util':>20}{'olap':>20}")
    gt_util, ne_olap = np.mean(gt_utils), np.mean(ne_olaps) * 100
    for r, util, olap, in zip(upper, utils, olaps, gt_utils, ne_olaps):
        util, olap,  = np.mean(util), np.mean(olap) * 100,
        util_str = f"{util:>.1f} ({util-gt_util:>+7.1f})"
        olap_str = f"{olap:>.1f}% ({olap-ne_olap:>+6.1f}%)"
        print(f"{r:>8g}{util_str:>20s}{olap_str:>20s}")

    sing([['E5', 'G5'], ['D5', 'F5'], ['C5', 'E5']], extend=2)
