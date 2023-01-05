##########################################
#Kamer Ali Yuksel linkedin.com/in/kyuksel#
##########################################

import numpy as np

syms = ['AAPL', 'MSFT', 'TXN', 'SNPS', 'MCHP', 'NVDA', 'SBUX', 'ASML', 'QCOM', 'APH', 'KLAC', 'CRM', 'GOOGL']

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

        self.constant_weights = np.array([0.22184507, 0.17495915, 0.13058954, 0.1220931 , 0.07822578, 0.07674023, 0.06398375, 0.04677756, 0.02314778, 0.02249317, 0.0162905 , 0.012325  , 0.01052937])
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