# TL;DR
This repo is a game theoretical analyzer for liquidity providing (LP'ing) on concentrated liquidity market makers ([CLMMs](https://coinmarketcap.com/community/articles/65608f0cc54ab771279066a5/)). It is the core implementation of our [paper](https://arxiv.org/abs/2411.10399) **_Game Theoretic Liquidity Provisioning in Concentrated Liquidity Market Makers_** written by **_Weizhao Tang, Rachid El-Azouzi, Cheng Han Lee, Ethan Chan, and Giulia Fanti_**. If you would like to use this codebase or cite our work, please read details in [references](References). 

The workflow consists of the following steps:
1. Prepare csv tables with the same scheme as in [example-liquidity](example-liquidity).<br>
<span style="color:gray"><sub>They can be crawled from a fully-synced blockchain node, or provided by [Allium](https://www.allium.so/).</sub></span>
   - All blockchain [**events**](example-liquidity/2697585-eth-usdc-fee-100/2697585-Total.csv) associated with a liquidity pool
   - All liquidity [**positions**](example-liquidity/2697585-eth-usdc-fee-100/2697585-Positions.csv) identified by non-fungible tokens (NFTs)
   - All liquidity [**changes**](example-liquidity/2697585-eth-usdc-fee-100/2697585-Liquidity.csv) of the NFT positions

2. Build liquidity providing games with the prepared data.
```
python3 simulate.py -c -p <pool_id>
```
3. Compute strategic actions of each liquidity provider, such as ground truth (GT), Nash equilibrium (NE) and best response (BR).<br>
<span style="color:gray"><sub>Each action comes with the utility and overlap with ground truth.</sub></span>
```
python3 nash.py -p <pool_id>
```
4. Analyze the raw results and extract 1-D essential data for data visualization.
```
python3 analyze.py -p <pool_id>
```
5. Tabular summary and Visualization of the 1-D essential data. <br>
<span style="color:gray"><sub>Use the Jupyter notebook `essence.ipynb`.</sub></span>

# 0. Prerequisites
- Python 3.9<br>
<span style="color:gray"><sub>Newer or older versions may raise problems with package APIs.</sub></span>
- Python Packages
```
pip install pandas numpy scipy matplotlib tqdm scikit-learn sortedcontainers
```

Optional: if you want sound notifications after completing each step (which may take a few minutes or a few hours), install [SoX](https://arielvb.readthedocs.io/en/latest/docs/commandline/sox.html).

# 1. Raw Data Preparation
[```pairs.csv```](pairs.csv) lists some *Uniswap* liquidity pools of popular cryptocurrecy tokens (such as WETH, WBTC, USDC, DAI) on the *Ethereum blockchain*. To extend the data to other tokens, other pool implementations (than Uniswap), or other blockchains, see [TradingStrategy](https://tradingstrategy.ai/trading-view/backtesting). 

Format:
```
liquidity
 └─ /<pool_id>-<pair_slug>
     ├─ /<pool_id>-Total.csv
     ├─ /<pool_id>-Positions.csv
     └─ /<pool_id>-Liquidity.csv
```
Make sure that under `liquidity`, `<pool_id>-*/` matches a unique directory for `pool_id` to be analyzed. 

> [!TIP]
> Verbally, we describe amount of tokens with decimals (e.g., 0.05 ETH). However, all amounts are *integers* in the digital world. In other words, there exists a decimal digit $d$, where
> ```math 
> \text{amount}_{\text{verbal}} = \text{amount}_{\text{digital}} \times 10^{-d}. 
> ```
> $d$ for token0 ($d_X$) and token1 ($d_Y$) are specified in the `token0_decimals` and `token1_decimals` columns in the `pairs.csv` file. Further,  
> ```math
> \begin{align*}
>     \text{price}_{\text{verbal}} &= \text{price}_{\text{digital}} \times 10^{d_X - d_Y}; \\
>     \text{liquidity}_{\text{verbal}} &= \text{liquidity}_{\text{digital}} \times 10^{-(d_X + d_Y)/2}. 
> \end{align*}
> ```

## 1.1. Events
```
<root>/liquidity/<pool_id>-<pair_slug>/<pool_id>-Total.csv
```
On Ethereum, a block contains transactions, where each transaction omits one or more **events**. [Here](https://etherscan.io/tx/0x045238585767783665d4ae283d3e787ece08bc393a4f618ae440dd7e470c82d8#eventlog) is an example. Each event has a mandatory field `event_type`, where types `Initialize`, `Mint`, `Burn`, and `Swap` are most relevant to liquidity pool dynamics.
- `Initialize`: Omitted when a liquidity pool is created. The event specifies the initial pool price. 
  - `sqrtPriceX96` (`uint160`): Initial pool price. 
  ```math
  \text{price}_{\text{digital}} = \left(\mathtt{sqrtPriceX96} \times 2^{-96} \right)^2.
  ```
- `Mint` and `Burn`: Omitted when liquidity provider (LP) adds or removes liquidity. 
  - `tickLower` and `tickUpper` (`int24`): Ticks (of type `int`) specifying the price range of the liquidity position.
  ```math
  \text{price}_{\text{digital}} = 1.0001^{\text{tick}}
  ```
  - `amount` (`uint128`): Amount of *liquidity* minted/burnt. 
  - `amount0` and `amount1` (`uint256`): Amounts of tokens added/removed from the pool; they are derived from `amount`, `tickLower`, and `tickUpper`.
- `Swap`: Omitted when a trader swaps one token for another. 
  - `sqrtPriceX96` (`uint160`): Pool price *after* the trade. 
  - `amount0` and `amount1` (`uint256`): Amounts of tokens added/removed from the pool; they are derived from 1) `sqrtPriceX96`, 2) the pool price *before* the trade, and 3) the liquidity distribution of the pool. 

## 1.2. Positions
```
<root>/liquidity/<pool_id>-<pair_slug>/<pool_id>-Positions.csv
```
A look-up table of all historical liquidity positions as non-fungible tokens (NFTs). We call them as **NFT positions**. Using a table, we store the mapping `token_id -> (tickLower, tickUpper)`. Other columns are unncessary. 

[Allium](https://www.allium.so/) enables direct queries of the table. On the other hand, if you want to crawl it on your own, you can parse `IncreaseLiquidity` and `DecreaseLiquidity` events. 

> When an individual LP adds liquidity, a smart contract issues the LP an NFT that is uniquely identified by `token_id`. It represents the LP's liquidity position with fixed price bounds `tickLower` and `tickUpper`. 
> 
> The amount of liquidity is not fixed. An LP can always add more liquidity to an existing position. However, when an LP removes liquidity, it must decrease liquidity from an existing position without freedom in deciding `tickLower` and `tickUpper`. 
>
> If an LP adds liquidity through a vault service like [Arrakis Finance](https://www.arrakis.finance/), the LP obtains **fungible tokens** instead. Unlike NFTs, these tokens are hard to track, so we consider their owners as non-players of the game. 

## 1.3. Liquidity Changes
```
<root>/liquidity/<pool_id>-<pair_slug>/<pool_id>-Liquidity.csv
```
A timestamped table recording **when** trackable liquidity is increased or decreased to **which NFT position** by **how much**. 

> Again, this table can be obtained from [Allium](https://www.allium.so/) or parsed from `IncreaseLiquidity` and `DecreaseLiquidity` events. It can be joined with the Positions table, but we chose not to in this repo. 

# 2. Game Building

#### Main Purpose
Read blockchain events from the tables and acquire parameters of the game for each day. 

#### Usage
```
python3 simulate.py -p <pool_id> [-r <checkpoint_path>] [-i][--disable-fee]  [--notify]
```
- `-p <pool_id>`: (**Mandatory**) ID of liquidity pool; found in [`pairs.csv`](pairs.csv). 
- `-r <checkpoint_path>`: Resume progress from a checkpoint. Checkpoints are automatically generated under `liquidity/<pool_id>/snapshots`.
- `-i`: Generate a plot of basic information (such as utility-budget relation) for each game. 
- `--disable-fee`: Disable fee calculation in the game. Only set this for quick sanity checks.
- `--notify`: Enable sound notification (must have [SoX](https://arielvb.readthedocs.io/en/latest/docs/commandline/sox.html)) when the simulation is done.

#### Output
Under `dynamics/<pool_id>`, for each date under format `YY-MM-DD`, two files specify the game: 
- `gamedata-<date>.json`: Game data in JSON format.
- `profit-<date>.csv`: All NFT positions that join the game, including their owners, price ranges, liquidity amounts, fee rewards, impermanent losses and utility values.

> [!IMPORTANT] 
> `dynamics/<pool_id>` is a temporary directory. When you run `simulate.py` again, you ___empty___ it and ___lose everything you had before___. Therefore, after completion, you must either manually rename the directory to `dynamics/<pool_id>-archive`, or execute
> ```
> bash push.sh dynamics/<pool_id> archive
> ```
> This safe archive directory is where files are read in further steps. 

#### Notes
1. We hardcoded
   - Time range of game building: 2023-12-31 to 2024-06-30
   - Frequency of game building: 1 day
   - Frequency of checkpointing: 50k events or at the end of each game building
2. Images are generated by multiprocessing. Currently, the fee gradients (marginal fee gains when increasing budget) are not tracked so some subplots are empty. 

# 3. Game Action Computation

#### Main Purpose
Given game parameters, compute the game actions of each LP at each day.

#### Usage

```
python3 nash.py -p <pool_id> [-g <gamma>] [-n <max_num_players>] [--inert-expansion <E>] [-r <keyword>] [--skip-gt] [--notify]
```
- `-p <pool_id>`: (**Mandatory**) ID of liquidity pool; found in [`pairs.csv`](pairs.csv). 
- `-g <gamma>`: $\gamma$ (step size, alternatively $\eta$) of the [relaxation algorithm](https://www.youtube.com/watch?v=BqHbsdTD5w8) for solving Nash equilibrium in concave games. Default is 0.1. If you observe that the V value staying unconverged, retry a lower value of $\gamma$.
- `-n <max_num_players>`: Maximum number of player LPs (higher budget LPs are prioritized). Default is 30. Remaining LPs are categorized as non-players.
- `--inert-expansion <E>`: Expansion factor of the price range in the inert game. As a result, priceUpper/priceLower expands by `E` times. Default is 1.5.
- `--raw`: Specify this when you want to use raw USD prices instead of bent prices in the game. Raw prices of tokens $p_X, p_Y$ are parsed from the [events](#11-events) table. Given pool price $q$, bent prices $\bar p_X, \bar p_Y$ are calculated as follows:
  ```math
  \bar p_X = \sqrt{p_X p_Y q}, \quad \bar p_Y = \sqrt{p_X p_Y q^{-1}}.
  ```
  <sub>**WARNING**: If you don't bend the price, you may encounter *negative* impermanent loss rate and very impractical, opportunistic Nash equilibria, where liquidity crowds in ranges with impermanent gain.</sub>
- `-r <keyword>`: Reload experiment from a keyworded prior experiment. Specify this when you have already fully run `nash.py`, stored them under `game/<pool_id>-<keyword>`, and only want to redo a partial calculation.<br>
<sub>Ground truth actions are recalculated from `dynamics/<pool_id>-archive/profit-<date>.csv` by default.</sub>
- `--skip-gt`: Only set this flag when you are reloading and do not want to reparse the csv file which can be time consuming. 
- `--notify`: Enable sound notification (must have [SoX](https://arielvb.readthedocs.io/en/latest/docs/commandline/sox.html)) when the simulation is done.


#### Output
For each date under format `YY-MM-DD`, `game/<pool_id>/nash-<date>.json` stores the game actions of each LP. 

> [!IMPORTANT] 
> `game/<pool_id>` is also a temporary directory. When you run `nash.py` again, you ___lose everything you had before___. Therefore, after completion, you must either manually rename the directory to `dynamics/<pool_id>-<keyword>`, or execute
> ```
> bash push.sh game/<pool_id> <keyword>
> ```
> This safe directory can be reloaded in the future runs of `nash.py` or read in further steps.

#### Notes
1. In addition to the daily game, we construct two empirical games to simulate the actions of LPs:
   - **Responsive Game**: constructed by *accurate* game parameters from *previous day* 
   - **Inert Game**: constructed by *inaccurate* game parameters from *past week*
> [!TIP]
> Realism v.s. impressionism
2. Currently, we calculate the following actions:
   - Ground Truth (GT): actions parsed from data
   - Nash Equilibrium (NE): actions at equilibrium of the game; computed by the relaxation algorithm
   - Best Response (BR): actions that maximize the utility of each player, assuming other players are sticking to GT
> [!TIP]
> See [material](https://s3.wp.wsu.edu/uploads/sites/1790/2017/11/nash-equilibrium.pdf) for game theoretic concepts. 
   - Yesterday's Actions (YDay): Simply repeating yesterday's actions
   - Nash Equilibrium in Responsive Game (R_NE): actions at Nash equilibrium of the responsive game
   - Best Response in Responsive Game (R_BR): best response actions of the responsive game
   - Nash Equilibrium in Inert Game (I_NE): actions at Nash equilibrium of the inert game
   - Best Response in Inert Game (I_BR): best response actions of the inert game
3. We generally calculate the following metrics for each action:
   - Utility: Utility (net income) of LP who executes the action, assuming other players stick to GT
   - GT Overlap: Overlap between an action and the ground truth action
4. We calculate the following metrics specifically for some actions:
   - (GT) `fee` and `cost`: Fee income and impermanent loss of the LP in the game, breaking down utility
   - (GT) `fee_data` and `cost_data`: Fee income and impermanent loss of the LP parsed from the raw data; generally equal to `fee` and `cost`
   - (NE/R_NE/I_NE) `ne_util`: Utility of LP with Nash Equilibrium action, assuming other players also use NE
  
# 4. Game Action Analysis

#### Main Purpose
Extract main information from the game actions and their metrics as 1-D arrays, and export them into pickle files for future visualization.

#### Usage
```
python3 analyze.py -p <pool_id> --keyword <keyword> [-m <max_plotted_players>] [--notify]
```

- `-p <pool_id>`: (**Mandatory**) ID of liquidity pool; found in [`pairs.csv`](pairs.csv).
- `--keyword <keyword>`: (**Mandatory**) Keyword of action computation. Data will be read from `game/<pool_id>-<keyword>`.
- `-m <max_plotted_players>`: Maximum number of players to be plotted. If not specified, no daily plot is generated. Does not affect the pickled results. 
- `--notify`: Enable sound notification (must have [SoX](https://arielvb.readthedocs.io/en/latest/docs/commandline/sox.html)) after the analysis is done.

#### Output
Pickled file `essence-<keyword>/<pool_id>.pkl`. See class `Essence` [nash_parser.py](nash_parser.py) for the relevant contents.


#### Notes
Some overall plots are generated under `game/<pool_id>-<keyword>`, but they can be ignored.

# 5. Tabular Summary and Visualization

#### Main Purpose
Generate a LaTeX tabular summary and visualization of the game actions and their metrics. 

#### Usage
Walk through the sections in `essence.ipynb`. 

#### Output
- LaTeX tabular summary: `tabletex/table.tex`.
- LaTeX visualization: `essence-<keyword>/*.jpg`.

# Appendix. Tuning Inert Expansion Parameter

#### Purpose
Preview the utility and GT overlap for different inert expansion parameters `E` when constructing the inert game. 

#### Usage
```
python3 inert_test.py -p <pool_id> [-s <skip_prob>] -u <U[0], U[1], ...> -l <L[0], L[1],...> [-a]
```

- `-p <pool_id>`: (**Mandatory**) ID of liquidity pool; found in [`pairs.csv`](pairs.csv).
- `-s <skip_prob>`: Probability of skipping calculation for each day. Default is 0.
- `-u <U[0], U[1],...>`: (**Mandatory**) A list of upper expansion factors. For each item $E_\text{U}$ in list `U`, we calculate the utility and GT overlap for the inert game with upper expansion $E_\text{U}$. By default, lower expansion equals $E_\text{U}$. 
- `-l <L[0], L[1],...>`: NOT IMPLEMENTED. NO EFFECT BY SETTING THIS. 

#### Output
- Pickled data: `inert_test_result/I_BR-<pool_id>.pkl`. 
- json preview: `inert_test_result/I_BR-<pool_id>.json` (only includes means of utility and GT overlap for each inert expansion parameter).

# References
If you use this code in a publication, please cite the following [work](https://arxiv.org/abs/2411.10399):

#### Plain Text:
```
Tang, Weizhao, et al. "Game Theoretic Liquidity Provisioning in Concentrated Liquidity Market Makers." arXiv preprint arXiv:2411.10399 (2024).
```

#### BibTeX: 
```
@article{tang2024game,
  title={Game Theoretic Liquidity Provisioning in Concentrated Liquidity Market Makers},
  author={Tang, Weizhao and El-Azouzi, Rachid and Lee, Cheng Han and Chan, Ethan and Fanti, Giulia},
  journal={arXiv preprint arXiv:2411.10399},
  year={2024}
}
````
