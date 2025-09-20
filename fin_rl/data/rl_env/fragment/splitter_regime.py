import numpy as np
from typing import Dict
from .base_splitter import BaseSplitter
from .trading_dataset import TradingDataset


class FlexibleRegimeSplitter(BaseSplitter):
    def __init__(self, regime_ratios: Dict, mode: str = "fraction",
                 bull_thr=0.02, bear_thr=-0.02, side_thr=0.01,
                 strategy="deterministic", seed=None, verbose=True):
        """
        regime_ratios: cấu hình số lượng/tỉ lệ fragment cho từng split
        mode: "fraction" (tỉ lệ) hoặc "count" (số lượng)
        bull_thr: return_weighted > ngưỡng => bull
        bear_thr: return_weighted < ngưỡng => bear
        side_thr: |return_weighted| < ngưỡng => sideway
        strategy: ["deterministic", "random", "time_stratified"]
        """
        self.regime_ratios = regime_ratios
        self.mode = mode
        self.bull_thr = bull_thr
        self.bear_thr = bear_thr
        self.side_thr = side_thr
        self.strategy = strategy
        self.seed = seed
        self.verbose = verbose

    def _label_fragments(self, dataset: TradingDataset):
        bull_ids, bear_ids, side_ids = [], [], []

        # Fail-fast: dataset phải có cột per-symbol
        need_cols = []
        for sym in dataset.symbols:
            need_cols += [f"Close_{sym}", f"QuoteVolume_{sym}"]

        for i, frag in enumerate(dataset.fragments):
            df = frag.df

            # kiểm tra tối thiểu cho fragment này
            for col in need_cols:
                if col not in df.columns:
                    raise KeyError(
                        f"[FlexibleRegimeSplitter] Fragment {i} missing required column '{col}'. "
                        f"Ensure dataset is WIDE with per-symbol columns."
                    )

            returns_per_symbol = {}
            weights_per_symbol = {}

            rets, weights = [], []
            for sym in dataset.symbols:
                col_close = f"Close_{sym}"
                col_qv    = f"QuoteVolume_{sym}"

                start_p, end_p = df[col_close].iloc[0], df[col_close].iloc[-1]
                r = (end_p / start_p) - 1
                w = float(df[col_qv].sum())

                returns_per_symbol[sym] = {
                    "first": float(start_p),
                    "last": float(end_p),
                    "return": float(r)
                }
                weights_per_symbol[sym] = float(w)

                if w > 0:
                    rets.append(r * w)
                    weights.append(w)

            weighted_return = (sum(rets) / sum(weights)) if weights else 0.0

            frag.meta["returns_per_symbol"] = returns_per_symbol
            frag.meta["weights_per_symbol"] = weights_per_symbol
            frag.meta["return_weighted"] = weighted_return

            # regime classification by weighted return
            if weighted_return > self.bull_thr:
                frag.meta["regime"] = "bull"
                bull_ids.append(i)
            elif weighted_return < self.bear_thr:
                frag.meta["regime"] = "bear"
                bear_ids.append(i)
            elif abs(weighted_return) < self.side_thr:
                frag.meta["regime"] = "sideways"
                side_ids.append(i)
            else:
                frag.meta["regime"] = "sideways"
                side_ids.append(i)

            if self.verbose:
                print(f"[Label] Frag {i} ({frag.start.date()}→{frag.end.date()}): "
                      f"weighted_return={weighted_return:.2%}, regime={frag.meta['regime']}")
                print(f"    df cols = {list(df.columns)[:10]}{' ...' if len(df.columns)>10 else ''}")
                print(f"    meta keys = {list(frag.meta.keys())}")
                for sym, info in returns_per_symbol.items():
                    w = weights_per_symbol[sym]
                    print(f"        {sym}: first={info['first']:.4f}, last={info['last']:.4f}, "
                          f"return={info['return']:.2%}, weight={w:.2e}")

        if self.verbose:
            print(f"[Summary] bull={len(bull_ids)}, bear={len(bear_ids)}, side={len(side_ids)}")

        return {"bull": bull_ids, "bear": bear_ids, "sideways": side_ids}

    def _choose(self, pool, k):
        if k <= 0 or not pool:
            return []
        if self.strategy == "deterministic":
            return pool[:k]
        elif self.strategy == "random":
            rng = np.random.default_rng(self.seed)
            return list(rng.choice(pool, size=min(k, len(pool)), replace=False))
        elif self.strategy == "time_stratified":
            step = max(1, len(pool) // k)
            return pool[::step][:k]
        else:
            raise ValueError(f"Unknown strategy={self.strategy}")

    def split(self, dataset: TradingDataset):
        regime_ids = self._label_fragments(dataset)
        out = {}

        pools = {reg: list(ids) for reg, ids in regime_ids.items()}

        for split_name, ratios in self.regime_ratios.items():
            total = sum(ratios.values())
            norm_ratios = {k: v / total for k, v in ratios.items()} if total > 0 else ratios

            idxs = []
            for regime, val in norm_ratios.items():
                pool = pools[regime]
                if not pool:
                    continue
                if self.mode == "fraction":
                    k = int(val * len(pool))
                elif self.mode == "count":
                    k = min(int(val), len(pool))
                else:
                    raise ValueError(f"Unknown mode={self.mode}")

                chosen = self._choose(pool, k)
                idxs.extend(chosen)
                pools[regime] = [i for i in pool if i not in chosen]

            idxs = sorted(set(idxs))
            frags = [dataset.fragments[i] for i in idxs]
            out[split_name] = TradingDataset(
                name=f"{dataset.name}_{split_name}",
                fragments=frags,
                symbols=dataset.symbols,
                candle_level=dataset.candle_level
            )

            if self.verbose:
                print(f"[FlexibleRegimeSplitter] {split_name}: n_frag={len(frags)}")
                for i in idxs:
                    frag = dataset.fragments[i]
                    print(f"    Frag {i}: regime={frag.meta['regime']}, "
                          f"weighted_return={frag.meta['return_weighted']:.2%}")
                    # có thể in thêm per-symbol nếu muốn
        return out
