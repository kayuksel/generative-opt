# Democratizing Index Tracking: A GNN-based Meta-Learning Method for Sparse Portfolio Optimization

Investing in stocks is a popular way for individuals to grow their wealth and diversify their investment portfolio, but many exchange-traded funds (ETFs) and mutual funds that offer actively managed index funds are not available to small investors in Europe due to UCITS regulations. An approach, called sparse index tracking, can allow investors to create their own sparse stock portfolio for tracking an index. However, selecting the optimal portfolio from thousands of stocks can be a sophisticated and resource-intensive task.

To address this issue, I have developed a novel population-based optimization method employing a Deep Generative Neural Network trained with policy gradient to sample high-quality candidates. I have compared the method to the state-of-the-art optimization algorithm (Fast CMA-ES) and have found that it is more efficient at finding optimal solutions. Both methods are implemented on GPU using the PyTorch framework and are available in this repository (together with the dataset) for reproducibility and further improvement.

![](qc_backtest.png)
