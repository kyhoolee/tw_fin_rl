import argparse, yaml, json
from .registry import get_splitter
from .trading_dataset import TradingDatasetFactory

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config cho splitter")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--candle-level", required=True)
    ap.add_argument("--symbols", nargs="+", required=True)
    args = ap.parse_args()

    # Load dataset gốc
    dataset = TradingDatasetFactory.from_raw(
        data_dir=args.data_dir,
        candle_level=args.candle_level,
        symbols=args.symbols
    )

    # Đọc config YAML
    cfg = yaml.safe_load(open(args.config))
    splitter = get_splitter(cfg["mode"])(**cfg["params"])
    result = splitter.split(dataset)

    # Xuất report nhỏ
    print(json.dumps({k: v.time_ranges() for k,v in result.items()}, indent=2, default=str))
