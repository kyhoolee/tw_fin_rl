from typing import Dict
from .trading_dataset import TradingDataset

class BaseSplitter:
    """Interface cho tất cả splitter."""

    def split(self, dataset: TradingDataset) -> Dict[str, TradingDataset]:
        """
        Input: TradingDataset
        Output: {"train": TradingDataset, "test": ..., "unseen": ...}
        """
        raise NotImplementedError
