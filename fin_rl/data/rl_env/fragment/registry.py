from .splitter_continuous import ContinuousSplitter
from .splitter_fragmented import FragmentedSplitter
from .splitter_regime import RegimeSplitter

SPLITTERS = {
    "continuous": ContinuousSplitter,
    "fragmented": FragmentedSplitter,
    "regime": RegimeSplitter,
}

def get_splitter(name: str):
    if name not in SPLITTERS:
        raise ValueError(f"Unknown splitter {name}")
    return SPLITTERS[name]
