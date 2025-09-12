import yaml

def load_fee_cfg(path: str, name: str | None = None):
    with open(path, "r") as f:
        y = yaml.safe_load(f)
    if name is None:
        name = y.get("default")
    cfg = y["presets"][name]
    return float(cfg["fee_bps"]), float(cfg.get("slippage_bps", 0.0))