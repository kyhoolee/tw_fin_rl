import numpy as np
from .base_splitter import BaseSplitter
from .trading_dataset import TradingDataset

class RegimeSplitter(BaseSplitter):
    def __init__(self, regime_map: dict):
        """
        regime_map = {
            "train": ["bull","sideways"],
            "test":  ["bear"],
            "unseen":["mixed"]
        }
        """
        self.regime_map = regime_map
    
    def split(self, dataset: TradingDataset):
        out = {}
        for name, regimes in self.regime_map.items():
            out[name] = dataset.filter_by_regime(regimes)
        return out

def tag_regimes(df, price_col="Close", window=50):
    """Tag regime đơn giản: bull/bear/sideways dựa trên MA trend."""
    ma = df[price_col].rolling(window).mean()
    trend = df[price_col] - ma
    conds = [(trend > 0.01*df[price_col]),
             (trend < -0.01*df[price_col])]
    choices = ["bull","bear"]
    df["regime"] = np.select(conds, choices, default="sideways")
    return df
