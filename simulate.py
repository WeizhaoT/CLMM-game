import time
import numpy as np
import pandas as pd
import pickle
import traceback
import matplotlib.pyplot as plt
import psutil
import argparse
import multiprocessing as mp
import warnings

from collections import defaultdict, deque
from tqdm import tqdm
from library import LXC, ensure_path, sing, search_dir, DateIter, Pool, convert_price, plot_dynamics, check_int
from typing import Tuple


def get_token_info(pid):
    """
    Retrieves token information for a given pool ID.

    Args:
        pid (int): The pool ID for which to retrieve token information.

    Returns:
    dict: A dictionary containing the following token information:
        - 'T0': Symbol of token0.
        - 'T1': Symbol of token1.
        - 'D0': Decimals of token0.
        - 'D1': Decimals of token1.
        - 'fee': Fee associated with the pool, expressed as a fraction.
    """
    df = pd.read_csv('pairs.csv', index_col='pair_id')
    row = df.loc[pid]
    return {
        'T0': row['token0_symbol'],
        'T1': row['token1_symbol'],
        'D0': int(row['token0_decimals']),
        'D1': int(row['token1_decimals']),
        'fee': int(row['fee']) / 1e4
    }


def get_token_id(row, liq: pd.DataFrame, pos: pd.DataFrame):
    """
    Identifies the token IDs associated with a given transaction row.

    Args:
        row (pd.Series): A row from a DataFrame containing transaction details, including 'transaction_hash', 'event', 
                        'tickLower', 'tickUpper', and 'liquidity'.
        liq (pd.DataFrame): A DataFrame containing liquidity information, indexed by 'transaction_hash'.
        pos (pd.DataFrame): A DataFrame containing position information, indexed by 'token_id'.

    Returns:
        list: A list of token IDs that match the transaction details. Returns [-1] if no matching tokens are found.
    """
    try:
        entries = liq.loc[[row['transaction_hash']]]
    except KeyError:
        return [-1]
    tokens, err_reasons = [], []
    for _, entry in entries.iterrows():
        token = entry['token_id']
        mult = -1 if row['event'] == 'burn' else 1
        row_info = (int(row['tickLower']), int(row['tickUpper']), int(row['liquidity'])*mult)
        pos_info = tuple(pos.loc[token, ['tick_lower', 'tick_upper']]) + (int(entry['liquidity_change']),)
        if row_info == pos_info:
            tokens.append(token)
        else:
            err_reasons.append((row_info, pos_info))
            pass
    if not tokens:
        print(err_reasons)
        warnings.warn(f"Found empty tokens for tx {row['transaction_hash']}", UserWarning)
        return [-1]
    return tokens


def handle_liq_change(row, df_liq, df_pos, memory):
    """
    Handles changes in liquidity for a given transaction row.

    Args:
        row (pd.Series): A row from a DataFrame containing transaction details, including 'transaction_hash', 'event', 
                        'tickLower', 'tickUpper', and 'liquidity'.
        df_liq (pd.DataFrame): A DataFrame containing liquidity information, indexed by 'transaction_hash'.
        df_pos (pd.DataFrame): A DataFrame containing position information, indexed by 'token_id'.
        memory (set): A set used to track processed transactions to avoid duplicate processing.

    Returns:
        tuple: A tuple (dx, dy) representing the change in token amounts due to the liquidity change.
    """
    tl, tu, liq = int(row['tickLower']), int(row['tickUpper']), int(row['liquidity'])
    if liq == 0:
        return 0, 0

    if row['event'] == 'burn':
        liq = -liq

    tokens = get_token_id(row, df_liq, df_pos)

    if len(tokens) == 1:
        return pool.update_position(tokens[0], liq, tl, tu, row['txFrom'], ts)

    dx = dy = 0
    for token in tokens:
        if (token, row['transaction_hash']) not in memory:
            dx, dy = pool.update_position(token, liq, tl, tu, row['txFrom'], ts)
            memory.add((token, row['transaction_hash']))

    return dx, dy


def load(path: str, pid: int):
    """
    Loads and processes data related to positions, liquidity, and events for a given pool ID.

    Args:
        path (str): The directory path where the data files are located.
        pid (int): The pool ID used to identify the specific data files to load.

    Returns:
        tuple: A tuple containing three pandas DataFrames:
            - df_ttl: DataFrame containing event data, indexed by 'index'.
            - df_pos: DataFrame containing position data, indexed by 'token_id'.
            - df_liq: DataFrame containing liquidity data, indexed by 'transaction_hash'.
    """
    timer_start = time.time()
    print(LXC.yellow_lt(f'Loading position  (1/3)... \r'), end='', flush=True)
    df_pos = pd.read_csv(f'{path}/{pid}-Positions.csv', index_col='token_id',
                         usecols=['token_id', 'liquidity', 'tick_lower', 'tick_upper'],
                         dtype={'liquidity': str, 'token_id': int, 'tick_lower': int, 'tick_upper': int})
    print(LXC.yellow_lt(f'Loading liquidity (2/3)... \r'), end='', flush=True)
    df_liq = pd.read_csv(f'{path}/{pid}-Liquidity.csv', index_col='transaction_hash',
                         usecols=['block_timestamp', 'token_id', 'liquidity', 'liquidity_change', 'transaction_hash'],
                         dtype={'block_timestamp': str, 'liquidity': str, 'liquidity_change': str,
                                'transaction_hash': str, 'token_id': int})
    df_liq['block_timestamp'] = pd.to_datetime(df_liq['block_timestamp'])
    print(LXC.yellow_lt(f'Loading event     (3/3)... \r'), end='', flush=True)
    usecols = ['index', 'timestamp', 'event', 'amount0', 'amount1', 'liquidity', 'tickLower', 'tickUpper',
               'sqrtPriceX96', 'token0_price_usd', 'token1_price_usd', 'txFrom', 'transaction_hash']
    type_map = {c: str for c in usecols if not c.startswith('tick')} | \
        {'index': int, 'token0_price_usd': float, 'token1_price_usd': float}
    df_ttl = pd.read_csv(f'{path}/{pid}-Total.csv', index_col='index', dtype=type_map,  usecols=usecols,
                         converters={key: check_int for key in ['tickLower', 'tickUpper']})
    df_ttl['timestamp'] = pd.to_datetime(df_ttl['timestamp'])
    print(LXC.green_lt(f'Loading COMPLETE in {int(time.time() - timer_start)} seconds\n'), flush=True)
    return df_ttl, df_pos, df_liq


def player_pos_delta(df_liq: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> Tuple[dict, dict]:
    """
    Calculates the changes in player positions over a specified time range.

    Args:
        df_liq (pd.DataFrame): A DataFrame containing liquidity information, indexed by 'block_timestamp'.
        start (pd.Timestamp): The start timestamp for the range of interest.
        end (pd.Timestamp): The end timestamp for the range of interest.

    Returns:
        Tuple[dict, dict]: A tuple containing two token_id -> liquidity dictionaries:
            - The first dictionary updates the set of player LPs during this period (from player set of last period).
            - The second dictionary updates the positions by the end of this period (from end of last period).
    """
    df_liq = df_liq.set_index('block_timestamp').truncate(before=start+pd.Timedelta(milliseconds=1), after=end)

    initial, delta_updates, ending_updates, zeros = {}, {}, {}, set()
    for _, row in df_liq.iterrows():
        token_id, liq_change, liq = row['token_id'], int(row['liquidity_change']), int(row['liquidity'])
        if token_id not in initial:
            delta_updates[token_id] = initial[token_id] = liq - liq_change
            if initial[token_id] == 0:
                zeros.add(token_id)
        delta_updates[token_id] = min(delta_updates[token_id], liq)
        ending_updates[token_id] = liq
        if liq == 0:
            ending_updates.pop(token_id)

    for token_id in zeros:
        delta_updates.pop(token_id)

    return {k: v for k, v in sorted(delta_updates.items(), key=lambda pair: -pair[1])}, ending_updates


def plot_simulation_error(liq_idx, swap_idx, errors, figpath):
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), dpi=200)
    axs[0].plot(liq_idx, [errors[i] for i in liq_idx], lw=.5, label='Liquidity')
    axs[0].set_title('Mint/Burn')
    axs[1].plot(swap_idx, [errors[i] for i in swap_idx], lw=.5, label='Swap')
    axs[1].set_title('Swap')

    for ax in axs:
        ax.set_yscale('log')
        ax.set_xlabel(f'Event ID')
        ax.set_ylabel(f'Error')
        ax.grid(True)
    fig.tight_layout()
    fig.savefig(figpath)
    plt.close()


if __name__ == '__main__':
    SNAP_INTV = 50000

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pool', type=int, required=True, help='Pool number')
    parser.add_argument('-r', type=str, help='Restore from checkpoint')
    parser.add_argument('-i', '--image', action='store_true', help='Generate periodic summary plots')
    parser.add_argument('--disable-fee', action='store_true', help='Disable fee calculation')
    parser.add_argument('--notify', action='store_true', help='Notify completion with sound')
    args = parser.parse_args()
    PID = args.pool

    def in_date_range(date: pd.Timestamp):
        return date > pd.Timestamp('2023-12-30 23:59:59')

    process = psutil.Process()
    proc_pool = mp.Pool(mp.cpu_count() - 2)
    jobs = {}

    PATH, INFO, WORK = search_dir(PID), get_token_info(PID), f'dynamics/{PID}'

    ensure_path(WORK, empty=True)
    ensure_path(f'{PATH}/snapshot', empty=False)

    F_TTL, F_POS, F_LIQ = load(PATH, PID)

    init = F_TTL.iloc[0]
    assert init['event'].lower() == 'initialize'

    time_iter = DateIter(init['timestamp'] + pd.DateOffset(day=0, normalize=True) + pd.Timedelta(milliseconds=-1), '1d')

    if args.r:
        with open(f'{args.r}', 'rb') as f:
            checkpoint = pickle.load(f)
            pool: Pool = checkpoint['pool']
            mem_idx, mem_ts = checkpoint['index'], checkpoint['timestamp']
            print(f'Reloading from index {mem_idx}/{len(F_TTL)-1} at {mem_ts}')
    else:
        pool = Pool(
            price=convert_price(int(init['sqrtPriceX96']), 0.),
            time=init['timestamp'],
            usd_price0=init['token0_price_usd'],
            usd_price1=init['token1_price_usd'],
            fee=INFO['fee'],
            decimal0=INFO['D0'],
            decimal1=INFO['D1']
        )
        mem_idx = 0

    liq_idx, swap_idx = [], []
    errors = {}

    bar = tqdm(total=len(F_TTL) - 1, initial=mem_idx, position=0)

    while time_iter.ddl() < F_TTL.iloc[mem_idx+1]['timestamp']:
        time_iter.proceed()

    bad_id, bad_tokens = -1, {}

    stats, eta = defaultdict(int), {}
    last_ending_updates = {}
    cp = time.time()
    memory = set()

    try:
        for i in range(mem_idx+1, len(F_TTL)):
            timer_start = time.time()
            row = F_TTL.iloc[i]
            event, n0, n1, ts = row['event'].lower(), int(row['amount0']), int(row['amount1']), row['timestamp']

            flag_in_range = in_date_range(ts)
            stats[event] += 1

            if event == 'burn':
                n0, n1 = -n0, -n1

            try:
                if event == 'mint' or event == 'burn':
                    dx, dy = handle_liq_change(row, F_LIQ, F_POS, memory)
                    liq_idx.append(i)
                elif event == 'swap':
                    if n0 > 0:
                        n0 *= (1 - INFO['fee'])
                    else:
                        n1 *= (1 - INFO['fee'])

                    dx, dy = pool.swap(convert_price(int(row['sqrtPriceX96'])),
                                       row['token0_price_usd'], row['token1_price_usd'],
                                       compute_fee=(not args.disable_fee) and flag_in_range, timestamp=ts)
                    swap_idx.append(i)
            except:
                bar.close()
                print(LXC.red(traceback.format_exc()))
                print(LXC.red(f'\nError occurred with event {i}, transaction {row["transaction_hash"]}'))
                break

            if event == 'mint' or event == 'burn' or event == 'swap':
                dx, dy = np.round(dx), np.round(dy)
                error_x = abs(dx - n0) / n0 if n0 != 0 else 0
                error_y = abs(dy - n1) / n1 if n1 != 0 else 0
                errors[i] = max(error_x, error_y)

                if event in eta:
                    eta[event].append(time.time() - timer_start)
                else:
                    eta[event] = deque([time.time() - timer_start], maxlen=500)

            flag_settle = i < len(F_TTL) - 1 and F_TTL.iloc[i+1]['timestamp'] > time_iter.ddl()
            if i == len(F_TTL) - 1 or flag_settle:
                start, start_str = time_iter.ddl(), time_iter.ddl_str()
                if flag_in_range:
                    res = pool.report_profit(time_iter.ddl(), path=WORK, datetime=time_iter.ddl_str())
                    if args.image:
                        jobs[time_iter.ddl_str()] = proc_pool.map_async(plot_dynamics, [res])
                else:
                    pool.reset_profit(time_iter.ddl())

                if i < len(F_TTL) - 1:
                    while F_TTL.iloc[i+1]['timestamp'] > time_iter.ddl():
                        time_iter.proceed()
                    if in_date_range(F_TTL.iloc[i+1]['timestamp']):
                        player_updates, ending_updates = player_pos_delta(F_LIQ, start, time_iter.ddl())
                        pool.register_player_pos(player_updates, last_ending_updates)
                        last_ending_updates = ending_updates

            if (not flag_in_range and i % SNAP_INTV == 0) or (flag_in_range and flag_settle and i > SNAP_INTV):
                cp = time.time()
                now = pd.Timestamp.now().strftime('%y-%m-%d_%H-%M-%S')
                with open(f'{PATH}/snapshot/{i:08d}_{now}.pkl', 'wb') as f:
                    pickle.dump({'pool': pool, 'index': i, 'timestamp': row['timestamp']}, f)

            if i % 1 == 0:
                cp_m, cp_s = divmod(int(time.time() - cp), 60)
                cp_str = f'{cp_m}:{cp_s:02d}'
                desc = f'{ts.strftime("%y-%m-%d")} | {len(pool.NFT)} | {len(pool.FT_deltas)} | ' + \
                    f'{int(process.memory_info().rss/2**20):d} MiB | CP {cp_str}'
                bar.set_description(desc if not flag_in_range else LXC.cyan_lt(desc))
            bar.update()
    except KeyboardInterrupt:
        bar.close()
        exit(1)

    bar.close()

    plot_simulation_error(liq_idx, swap_idx, errors, f'{WORK}/error.jpg')

    print('Simulation complete. Summary of event types:')
    keysize = max(len(key) for key in stats) + 1
    print(' ' * (keysize+16) + ' '.join(f'{n:>5d}%' for n in [5, 25, 50, 75, 95]))
    for key, count in sorted(stats.items()):
        msg = f'  {{:<{keysize}s}}: '.format(key) + f'{count:>9d}'
        if key in eta:
            eta_arr = np.array(eta[key]) * 1e3
            quants = np.quantile(eta_arr, [.5, .25, .5, .75, .95])
            msg += ' | ' + ' '.join(f'{q:6.1f}' for q in quants) + ' ms'

        print(msg)

    if args.image and len(jobs) > 0:
        timer_start = time.time()
        print(f'Waiting for plotters to finish (0/0 done, {int(time.time() - timer_start)} s)', end='', flush=True)
        while True:
            ready = [job.ready() for job in jobs.values()]
            try:
                successful = [job.successful() for job in jobs.values()]
            except Exception:
                time.sleep(1)
                print('\rWaiting for plotters to finish '
                      f'({sum(ready)}/{len(ready)} done, {int(time.time() - timer_start)} s)', end='', flush=True)
                continue
            if all(successful):
                print(f'\r({sum(ready)}/{len(ready)} done, {int(time.time() - timer_start)} s)',
                      end=f'{"":>40s}' + '\n', flush=True)
                break

            for date, job in jobs.items():
                if not job.successful():
                    print(date, job._value)
                    job.get()

    proc_pool.close()
    proc_pool.join()
    print(LXC.green_lt(f'Complete!'))

    if args.notify:
        sing([['C4', 'D4#', 'F4#', 'A5']])
