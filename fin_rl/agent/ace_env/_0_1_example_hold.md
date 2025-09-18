# Example – Hold Strategy under ACE

```
[1. Traditional Evaluation = Biased]
Dataset split:  [ Train+Valid ] --- [ Test ] --- [ Unseen ]
Hold strategy outcome:
    - Depends ONLY on (LastPrice - FirstPrice) of each split
    - If Test happens to end higher → "win"
    - If Unseen ends lower → "loss"
Problem:
    - Entire strategy judged by just 2 candles (start & end)
    - Sharpe / Annualized return meaningless across different split lengths

------------------------------------------------------------

[2. ACE Evaluation = Step-Action Returns]
Each candle t → t+1:
    Action: Hold position
    Reward: Return(t) = Price[t]/Price[t-1] - 1
    State : Current equity + features
Trajectory becomes:
    [+ , - , + , + , - , ...]   (sequence of wins/losses)
Metrics:
    - Step-level win rate
    - Profit factor
    - Expectancy
    - Aggregation: per-day / per-week / per-month

------------------------------------------------------------

[3. Robustness with Fragmented Data]
Dataset:   [ Segment A ]   GAP   [ Segment B ]
ACE handles:
    - Skip returns across GAP (no artificial bridge)
    - Evaluate normally within each segment
    - Carry equity forward across missing data
Result:
    - Easy to construct ENV with mixed regimes
    - No need for one continuous time series containing all regimes

------------------------------------------------------------

[4. Generality & Robotics Analogy]
Robotics goal:
    Reach point B (long-term)
But evaluation requires:
    - Step-level actions (walk, turn, avoid obstacle)
    - Measure robustness under noise/disturbance
    - Path efficiency, not just final arrival

Trading under ACE:
    Long-term goal = Wealth growth
    But evaluation = step-action returns
    - Robust across regimes
    - Comparable across strategies
    - Aggregates naturally to higher horizons
```

---

