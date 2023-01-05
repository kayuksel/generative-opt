##########################################
#Kamer Ali Yuksel linkedin.com/in/kyuksel#
##########################################

import numpy as np

syms = ['MSFT', 'AAPL', 'TXN', 'ASML', 'ADBE', 'CRM', 'SBUX', 'MCHP', 'MA', 'ADI', 'SWKS', 'SNPS', 'KLAC']

class MultidimensionalModulatedRegulators(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        #self.SetEndDate(2017, 1, 1)
        self.SetCash(100000)
        self.SetExecution(VolumeWeightedAveragePriceExecutionModel())
 
        self.symbols = []
        for i in range(len(syms)):
            self.symbols.append(Symbol.Create(syms[i], SecurityType.Equity, Market.USA))
            self.Debug(syms[i])
            
        self.SetUniverseSelection(ManualUniverseSelectionModel(self.symbols) )
        self.UniverseSettings.Resolution = Resolution.Hour
        
        self.AddEquity('VGT', Resolution.Hour)
        self.SetBenchmark('VGT')
        
        self.SetBrokerageModel(AlphaStreamsBrokerageModel())

        self.constant_weights = np.array([0.28535828, 0.23993727, 0.12635303, 0.05360154, 0.04681825, 0.04518177, 0.04505873, 0.04194283, 0.03757035, 0.03131177, 0.02168459, 0.0151392 , 0.01004238])
        self.constant_weights = self.constant_weights / np.sum(np.abs(self.constant_weights))

    def OnData(self, data):
                
        rebalance = False
        
        if self.Portfolio.TotalHoldingsValue > 0:
            total = 0.0
            for i, sym in enumerate(self.symbols):
                curr = (self.Securities[sym].Holdings.HoldingsValue/self.Portfolio.TotalPortfolioValue)
                diff = self.constant_weights[i] - curr
                total += np.abs(diff)
                
            if total > 0.05: 
                rebalance = True
                
            if rebalance:
                for i, sym in enumerate(self.symbols):
                    curr = (self.Securities[sym].Holdings.HoldingsValue/self.Portfolio.TotalPortfolioValue)
                    if self.constant_weights[i] < curr:
                        self.SetHoldings(sym, self.constant_weights[i])
                for i, sym in enumerate(self.symbols):
                    curr = (self.Securities[sym].Holdings.HoldingsValue/self.Portfolio.TotalPortfolioValue)                       
                    if self.constant_weights[i] > curr:
                        self.SetHoldings(sym, self.constant_weights[i])
        else:
            for i, sym in enumerate(self.symbols):
                    self.SetHoldings(sym, self.constant_weights[i])
