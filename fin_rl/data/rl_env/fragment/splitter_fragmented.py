from .base_splitter import BaseSplitter
from .trading_dataset import TradingDataset

class FragmentedSplitter(BaseSplitter):
    def __init__(self, assign: dict):
        """
        assign = {"train":[0,1], "test":[2], "unseen":[3]}
        index dựa theo dataset.fragments
        """
        self.assign = assign
    
    def split(self, dataset: TradingDataset):
        out = {}
        for name, idxs in self.assign.items():
            frags = [dataset.fragments[i] for i in idxs]
            out[name] = TradingDataset(name=f"{dataset.name}_{name}",
                                       fragments=frags,
                                       symbols=dataset.symbols,
                                       candle_level=dataset.candle_level)
        return out


class FragmentGenerator:
    @staticmethod
    def slice_fragments(dataset: TradingDataset, ranges: List[tuple]) -> List[TradingFragment]:
        """Cắt 1 dataset continuous thành list fragment theo ranges (start,end)."""
        df = dataset.concat()
        frags = []
        for (s,e) in ranges:
            part = df[(df["open_time"]>=pd.Timestamp(s,tz="UTC")) &
                      (df["open_time"]<=pd.Timestamp(e,tz="UTC"))].copy()
            if not part.empty:
                frags.append(TradingFragment(
                    df=part,
                    start=part["open_time"].iloc[0],
                    end=part["open_time"].iloc[-1]
                ))
        return frags
