import os
import argparse

from tqdm import tqdm
from collections import deque
from library import ensure_path, sing, Intelligence


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=int, required=True, help="pool ID")
    parser.add_argument('-g', '--gamma', type=float, default=1e-1, help="gamma in relaxation method of NE solving")
    parser.add_argument('--raw', action='store_true', help="use raw USD price instead of bent USD price")
    parser.add_argument('--inert-expansion', type=float, default=None, help="inert game expansion factor E")
    parser.add_argument('--notify', action='store_true', help="notify by sound")
    parser.add_argument('--skip-gt', action='store_true', help="skip reloading ground truth")

    parser.add_argument('-n', type=int, default=30, help="max number of players; do not set if reloading")
    parser.add_argument('-r', '--reload', type=str, default=None, help="reload previous results")

    args = parser.parse_args()

    PID = args.p
    expand = 1.5 if args.inert_expansion is None else args.inert_expansion

    DATA_PATH, AXV_PATH = f'dynamics/{PID}-archive', f'game/{PID}-archive'
    SAVE_PATH = f'game/{PID}'
    priors = deque([], maxlen=7)

    ensure_path(SAVE_PATH, empty=True)

    files = sorted([file for file in os.listdir(DATA_PATH) if file.startswith('gamedata-') and file.endswith('.json')])
    par_prev, gt_prev = {}, {}

    bar_hi = tqdm(total=len(files), position=0)
    bar_lo = tqdm(total=1, position=1, desc='Standby')

    for file in files:
        date = file[9:-5]
        bar_hi.set_description(f'{date} {"loading":<9}')
        if date < '24-01-01' or not os.path.exists(f'{DATA_PATH}/profit-{date}.csv'):
            bar_hi.update()
            continue

        reload_path, gt_path = f'game/{PID}-{args.reload}/nash-{date}.json', f'{DATA_PATH}/profit-{date}.csv'
        if args.reload is None or not os.path.exists(reload_path):
            reload_path = None
        elif args.skip_gt:
            gt_path = None

        INTEL = Intelligence(f'{DATA_PATH}/{file}', reload_path, gt_path,
                             raw=args.raw, max_players=args.n, date=date, prog=bar_lo)

        if len(INTEL.B) == 0:
            par_prev, gt_prev = {}, {}
            priors.clear()
            bar_hi.update()
            continue

        if INTEL.lacks('BR'):
            bar_hi.set_description(f'{date} {"BR":<9}')
            INTEL.best_response()

        if INTEL.lacks('NE'):
            bar_hi.set_description(f'{date} {"NE":<9}')
            INTEL.nash_equilibrium(args.gamma)

        if len(par_prev) > 0:
            INTEL.add_global('tick_prev', par_prev['ticks'])
            INTEL.add_global('tick_pall', par_prev['ticks_all'])
            if INTEL.lacks('YDay'):
                bar_hi.set_description(f'{date} {"YDay":<9}')
                INTEL.responsive_yday(gt_prev)

            if INTEL.lacks('R_BR'):
                bar_hi.set_description(f'{date} {"R_BR":<9}')
                INTEL.responsive_br(par_prev)

            if INTEL.lacks('R_NE'):
                bar_hi.set_description(f'{date} {"R_NE":<9}')
                INTEL.responsive_ne(par_prev, args.gamma)

        if len(priors) > 0:
            inert_params = INTEL.inert_summary(priors, expand)
            INTEL.add_global('tick_inert', list(inert_params['ticks']))
            if INTEL.lacks('I_BR') or args.inert_expansion is not None:
                bar_hi.set_description(f'{date} {"I_BR":<9}')
                INTEL.inert_br(inert_params)

            if INTEL.lacks('I_NE') or args.inert_expansion is not None:
                bar_hi.set_description(f'{date} {"I_NE":<9}')
                INTEL.inert_ne(inert_params, args.gamma)

        bar_hi.set_description(f'{date} {"exporting":<9}')
        INTEL.derive()
        INTEL.export(f'{SAVE_PATH}/nash-{date}.json')

        par_prev = INTEL.game_param()
        gt_prev = INTEL.gt_bar
        priors.appendleft(INTEL.inert_params())
        bar_hi.set_description(f'{date} finished')
        bar_hi.update()

    bar_hi.close()
    bar_lo.close()
    if args.notify:
        sing(['E5', 'D5', 'C5'], extend=2)
