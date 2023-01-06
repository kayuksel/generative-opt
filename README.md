# Democratizing Index Tracking: A GNN-based Meta-Learning Method for Sparse Portfolio Optimization
![](backtest_qc.png)
Investing in stocks is a popular way for individuals to grow their wealth and diversify their investment portfolio, but many exchange-traded funds (ETFs) and mutual funds that offer actively managed index funds are not available to small investors in Europe due to UCITS regulations. An approach, called sparse index tracking, can allow investors to create their own sparse stock portfolio for tracking an index. However, selecting the optimal portfolio from thousands of stocks is a sophisticated and resource-intensive task.
![](gnn_architecture.png)
To address this issue, I have developed a novel population-based optimization method employing a Deep Generative Neural Network trained with policy gradient to sample high-quality candidates. I have compared it against the state-of-the-art evolutionary strategy (Fast CMA-ES) and have found that it is more efficient at finding optimal solutions. Both methods are implemented on GPU using the PyTorch framework and are available in this repository (together with the dataset) for their reproducibility and further improvement.
![](comparison.png)
Before running **main.py**, download **Dataset.pkl** from the below link and place it to this folder:  
https://drive.google.com/file/d/1RVhboDO3u_subUgG1G8rwdY7Ar6Dyrxf/view?usp=sharing
