# Democratizing Index Tracking for Small Investors in Europe: A Meta-Learning Approach for Sparse Portfolio Optimization 

Investing in stocks is a popular way for individuals to grow their wealth and diversify their investment portfolio, but many exchange-traded funds (ETFs) and mutual funds that offer actively managed index funds are not available to small investors in Europe due to UCITS regulations. An approach, called sparse index tracking, allows investors to create their own portfolio of stocks based on an index. However, selecting the optimal portfolio from thousands of stocks can be a complex and time-consuming task.

To address this issue, I have developed a novel population-based optimization method using a deep generative neural network trained with policy gradient to identify the best portfolio. I have compared the method to a state-of-the-art optimization algorithm and have found that it is more efficient at finding high-quality solutions. The method is implemented on GPU using the PyTorch framework and the data and results are available in this repository for reproducibility and further improvement.

Our goal is to make it easier for small investors in the European Union to track the performance of indexes by providing a more efficient and accessible method for selecting portfolios. We hope that by open-sourcing our implementation and dataset, other researchers and practitioners will be able to build upon and improve my proposed method.
