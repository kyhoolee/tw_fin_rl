import pandas as pd

def apply_model_masks(df: pd.DataFrame, 
                      max_lookback: int=0, 
                      max_forward: int=0, 
                      embargo_ranges=None, 
                      ts_col="open_time") -> pd.DataFrame:
    """
    Thêm cột usable dựa trên model config:
    - Lookback/forward window
    - Embargo quanh test/unseen
    """
    usable = pd.Series(True, index=df.index)

    if max_lookback > 0:
        usable.iloc[:max_lookback] = False
    if max_forward > 0:
        usable.iloc[-max_forward:] = False

    if embargo_ranges:
        for (s,e,E) in embargo_ranges:  
            mask = (df[ts_col] >= (s - E)) & (df[ts_col] <= (e + E))
            usable[mask] = False

    df = df.copy()
    df["usable"] = usable
    return df
