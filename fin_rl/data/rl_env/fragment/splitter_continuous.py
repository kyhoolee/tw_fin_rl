import pandas as pd
from .base_splitter import BaseSplitter
from .trading_dataset import TradingDataset, TradingFragment

class ContinuousSplitter(BaseSplitter):
    def __init__(self, ranges: dict):
        """
        ranges = {
            "train": (start,end),
            "test": (start,end),
            "unseen": (start,end)
        }
        """
        self.ranges = ranges
    
    def split(self, dataset: TradingDataset):
        df = dataset.concat()
        out = {}
        for name, (s,e) in self.ranges.items():
            part = df[(df["open_time"]>=pd.Timestamp(s,tz="UTC")) &
                      (df["open_time"]<=pd.Timestamp(e,tz="UTC"))].copy()
            frag = TradingFragment(df=part, start=part["open_time"].iloc[0], end=part["open_time"].iloc[-1])
            out[name] = TradingDataset(name=f"{dataset.name}_{name}",
                                       fragments=[frag],
                                       symbols=dataset.symbols,
                                       candle_level=dataset.candle_level)
        return out
