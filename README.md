# TL;DR
This repo is a game theoretical analyzer for liquidity providing (LP'ing) on concentrated liquidity market makers ([CLMMs](https://coinmarketcap.com/community/articles/65608f0cc54ab771279066a5/)).

The workflow consists of the following steps:
1. Prepare csv tables with the same scheme as in [example-liquidity](example-liquidity). 
<span style="color:gray"><sub><sup>They can be crawled from a fully-synced blockchain node, or provided by [Allium](https://www.allium.so/).</sup></sub></span>

   - All blockchain [**events**](example-liquidity/2697585-eth-usdc-fee-100/2697585-Total.csv) associated with a liquidity pool
   - All liquidity [**positions**](example-liquidity/2697585-eth-usdc-fee-100/2697585-Positions.csv) identified by non-fungible tokens (NFTs)
   - All liquidity [**changes**](example-liquidity/2697585-eth-usdc-fee-100/2697585-Liquidity.csv) of the NFT positions

2. Build liquidity providing games with the prepared data.
```
python3 simulate.py -c -p <pool_number>
```
3. Compute strategic actions of each liquidity provider, such as ground truth (GT), Nash equilibrium (NE) and best response (BR). 
<span style="color:gray"><sub><sup>Each action comes with the utility and overlap with ground truth.</sup></sub></span>
```
python3 nash.py -p <pool_number>
```
4. Analyze the raw results and extract 1-D essential data for data visualization.
```
python3 analyze.py -p <pool_number>
```
5. Tabular summary and Visualization of the 1-D essential data. 
<span style="color:gray"><sub><sup>Use the Jupyter notebook ```essence.ipynb```.</sup></sub></span>
