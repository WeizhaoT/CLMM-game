# TL;DR
This repo is a game theoretical analyzer for liquidity providing (LP'ing) on concentrated liquidity market makers ([CLMMs](https://coinmarketcap.com/community/articles/65608f0cc54ab771279066a5/)).

The workflow consists of the following steps:
1. Prepare csv tables with the same scheme as in [example-liquidity](example-liquidity).<br>
<span style="color:gray"><sub>They can be crawled from a fully-synced blockchain node, or provided by [Allium](https://www.allium.so/).</sub></span>
   - All blockchain [**events**](example-liquidity/2697585-eth-usdc-fee-100/2697585-Total.csv) associated with a liquidity pool
   - All liquidity [**positions**](example-liquidity/2697585-eth-usdc-fee-100/2697585-Positions.csv) identified by non-fungible tokens (NFTs)
   - All liquidity [**changes**](example-liquidity/2697585-eth-usdc-fee-100/2697585-Liquidity.csv) of the NFT positions

2. Build liquidity providing games with the prepared data.
```
python3 simulate.py -c -p <pool_number>
```
3. Compute strategic actions of each liquidity provider, such as ground truth (GT), Nash equilibrium (NE) and best response (BR).<br>
<span style="color:gray"><sub>Each action comes with the utility and overlap with ground truth.</sub></span>
```
python3 nash.py -p <pool_number>
```
4. Analyze the raw results and extract 1-D essential data for data visualization.
```
python3 analyze.py -p <pool_number>
```
5. Tabular summary and Visualization of the 1-D essential data. <br>
<span style="color:gray"><sub>Use the Jupyter notebook `essence.ipynb`.</sub></span>

# Prerequisites
- Python 3.9<br>
<span style="color:gray"><sub>Newer or older versions may raise problems with package APIs.</sub></span>
- Python Packages
```
pip install pandas numpy scipy matplotlib tqdm scikit-learn sortedcontainers
```

# Raw Data Preparation
[```pairs.csv```](pairs.csv) lists some *Uniswap* liquidity pools of popular cryptocurrecy tokens (such as WETH, WBTC, USDC, DAI) on the *Ethereum blockchain*. To extend the data to other tokens, other pool implementations (than Uniswap), or other blockchains, see [TradingStrategy](https://tradingstrategy.ai/trading-view/backtesting). 

> ***Beware***: Verbally, we describe amount of tokens with decimals (e.g., 0.05 ETH). However, all amounts are *integers* in the digital world. In other words, there exists a decimal digit $d$, where
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

## Events
On Ethereum, a block contains transactions, where each transaction omits one or more **events**. [Here](https://etherscan.io/tx/0x045238585767783665d4ae283d3e787ece08bc393a4f618ae440dd7e470c82d8#eventlog) is an example. Each event has a mandatory field `event_type`, where types `Initialize`, `Mint`, `Burn`, and `Swap` are most relevant to liquidity pool dynamics.
- `Initialize`: Omitted when a liquidity pool is created. The event specifies the initial pool price. 
  - `sqrtPriceX96` (`uint160`): Initial pool price. 
  ```math
  \text{price}_{\text{digital}} = \left(\texttt{sqrtPriceX96} \times 2^{-96} \right)^2.
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

## Positions

A look-up table of all historical liquidity positions as non-fungible tokens (NFTs). We call them as **NFT positions**. Using a table, we store the mapping `token_id -> (tickLower, tickUpper)`. Other columns are unncessary. 

[Allium](https://www.allium.so/) enables direct queries of the table. On the other hand, if you want to crawl it on your own, you can parse `IncreaseLiquidity` and `DecreaseLiquidity` events. 

> When an individual LP adds liquidity, a smart contract issues the LP an NFT that is uniquely identified by `token_id`. It represents the LP's liquidity position with fixed price bounds `tickLower` and `tickUpper`. 
> 
> The amount of liquidity is not fixed. An LP can always add more liquidity to an existing position. However, when an LP removes liquidity, it must decrease liquidity from an existing position without freedom in deciding `tickLower` and `tickUpper`. 
>
> If an LP adds liquidity through a vault service like [Arrakis Finance](https://www.arrakis.finance/), the LP obtains **fungible tokens** instead. Unlike NFTs, these tokens are hard to track, so we consider their owners as non-players of the game. 

## Liquidity Changes
A timestamped table recording **when** trackable liquidity is increased or decreased to **which NFT position** by **how much**. 

> Again, this table can be obtained from [Allium](https://www.allium.so/) or parsed from `IncreaseLiquidity` and `DecreaseLiquidity` events. It can be joined with the Positions table, but we chose not to in this repo. 