# **NLP for RL-based Investment Agent — Work Plan**

### **Motivation**

* Original SARL paper: promising, but has **limitations**

  * Data: old, limited, unclear backtest split
  * Model: mainly policy-based with continuous actions (DPG)
  * State augmentation: news effect assumed *immediate*, unrealistic

---

### **My Planned Improvements**

* **Data Pipeline**

  * Collect richer, up-to-date datasets (stocks + crypto)
  * Explicit split: train / validation / test / *unseen test* across market regimes

* **RL Models**

  * **Continuous**: DPG, DDPG, SAC (fine-grained allocations)
  * **Discrete (1% grid)**: DQN, PPO (stability, risk control)

* **State Augmentation**

  * Combine price/volume + text/news + on-chain/transactional signals
  * Temporal alignment: lag windows, event-based matching
  * Ablation studies: isolate contribution of each signal

---

### **6-Week Milestones**

1. **Weeks 1–2**: Reproducible pipeline, price-only RL baseline
2. **Weeks 3–4**: Add text/transactional features with lag analysis
3. **Weeks 5–6**: Walk-forward evaluation, ablations, risk-aware metrics

---

### **Evaluation Metrics**

* Portfolio Value, Sharpe/Sortino, Calmar, Max Drawdown
* Turnover & transaction cost sensitivity

---

👉 **Goal**: Understand when and how NLP + transaction data improves RL portfolio management.
👉 **Next step**: Discuss asset universe & risk constraints with Prof. Lien.

---




------------------------


Dựa trên tài liệu `4_slide.md` và phân tích chi tiết trong `2_rl_based_portfolio.md`, dưới đây là phiên bản mở rộng và chi tiết hơn của khung kế hoạch trong `4_slide.md`, tích hợp các ý tưởng cải tiến từ `2_rl_based_portfolio.md` và các tài liệu liên quan. Tôi sẽ làm rõ các điểm về nguồn dữ liệu, cải tiến state augmentation, và thử nghiệm các mô hình RL, đồng thời đảm bảo phù hợp với mục tiêu nghiên cứu: hiểu rõ khi nào và làm thế nào NLP + dữ liệu giao dịch cải thiện quản lý danh mục đầu tư dựa trên RL.

---

# **NLP for RL-based Investment Agent — Detailed Work Plan**

## **Motivation**

* **Original SARL Paper (Ye et al., 2020, AAAI-20) Limitations** (from `2_rl_based_portfolio.md`):
  * **Data**: Uses outdated HighTech dataset (Reuters, 2010–2012, 4810 articles for 9 stocks, covering only 72% trading days). Limited diversity, unclear train/validation/test/unseen test splits, and no coverage of crypto markets.
  * **Model**: Primarily policy-based (DPG) with continuous actions, lacking exploration of other RL algorithms (e.g., value-based or actor-critic methods) or discrete action spaces for stability.
  * **State Augmentation**: Assumes immediate news impact on price (at time *t*), which is unrealistic due to temporal misalignment (e.g., anticipation or delayed reactions). Relies on company name matching, ignoring context-specific signals (e.g., earnings vs. minor news) and lacking sentiment intensity or volume-based weighting.
  * **Assumption Weaknesses**: News filtering lacks specificity, no lag analysis, and no incorporation of on-chain/transactional data, limiting robustness in non-stationary markets.

* **Proposed Improvements**:
  * Build a richer, up-to-date dataset covering stocks and crypto with clear market regime-based splits.
  * Enhance state augmentation with context-aware NLP, lag analysis, and on-chain data to better align news/tweets with price movements.
  * Experiment with a diverse set of RL algorithms (continuous and discrete actions) to identify optimal approaches for portfolio management.
  * Leverage insights from related papers (e.g., `Uncertainty-Aware Strategies`, `Mining Intraday Risk Factor Collections`, `101 Formulaic Alphas`) to incorporate uncertainty modeling and sentiment-driven risk factors.

---

## **Detailed Improvements**

### **1. Data Pipeline**

* **Objective**: Create a robust, reproducible pipeline for up-to-date, diverse data covering stocks and cryptocurrencies, with explicit splits to handle market regime shifts.

* **Data Sources** (from `2_rl_based_portfolio.md`):
  * **Stocks**:
    * **Yahoo Finance (yfinance)**: Free, provides daily OHLCV (Open, High, Low, Close, Volume) for stocks (e.g., S&P 500, CSI 300) from 2019–2025. Also includes basic news, though limited in depth.
    * **Alpha Vantage**: Free tier for stock price data and some news, but rate limits require careful request management.
  * **Cryptocurrencies**:
    * **Binance API**: Primary source for high-frequency (1-minute) OHLCV data for major coins (e.g., BTC, ETH, SOL) from 2019–2025. Aggregate to daily for alignment with news/tweets.
    * **CryptoCompare**: Free, provides historical daily news and social media data (2019–2025), aggregated from multiple sources, ideal for sentiment analysis.
    * **CryptoPanic**: Free tier offers daily news with community-voted sentiment scores, covering 2019–2025, suitable for crypto-specific signals.
    * **CoinGecko/CMC**: Limited free tier for recent news, less suitable for historical depth but useful for cross-validation.
    * **X Platform**: Free access to tweets for real-time sentiment (2019–2025). Use search API to filter by coin/stock-specific keywords (e.g., #BTC, $AAPL).
  * **Rate Limit Management**:
    * Split requests by date/asset (e.g., 1 day/coin/request) with 0.1–1s delays.
    * Cache data in CSV files to avoid redundant API calls.
  * **Example Workflow**:
    ```python
    import pandas as pd
    from binance.client import Client
    from cryptocompare import cryptocompare
    import yfinance as yf
    from datetime import datetime
    import time

    def get_stock_data(ticker, start, end):
        df = yf.download(ticker, start=start, end=end, interval='1d')
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]

    def get_crypto_data(symbol, start, end):
        client = Client()
        klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, start, end)
        return pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', ...])

    def get_news_data(coins, start, end, auth_token):
        cryptocompare.set_api_key(auth_token)
        news_data = []
        current_date = start
        while current_date <= end:
            for coin in coins:
                news = cryptocompare.get_news(timestamp=current_date)
                news_data.extend([{'date': current_date, 'coin': coin, 'title': item['title']} for item in news])
            current_date += pd.Timedelta(days=1)
            time.sleep(0.2)
        return pd.DataFrame(news_data)

    start = '2019-01-01'
    end = '2025-09-03'
    stocks = ['AAPL', 'MSFT', 'GOOGL']
    coins = ['BTC', 'ETH', 'SOL']
    stock_data = {ticker: get_stock_data(ticker, start, end) for ticker in stocks}
    crypto_data = {coin: get_crypto_data(f'{coin}USDT', start, end) for coin in coins}
    news_data = get_news_data(coins, start, end, 'YOUR_API_KEY')
    ```

* **Data Splits**:
  * **Train (60%)**: 2019–2022, covering bull and bear markets (e.g., COVID crash, 2021 crypto boom).
  * **Validation (20%)**: 2022–2023, for hyperparameter tuning across volatile periods.
  * **Test (10%)**: 2023–2024, to evaluate generalization in recent market conditions.
  * **Unseen Test (10%)**: 2024–2025, reserved for final evaluation to simulate real-world deployment across unseen regimes (e.g., post-halving crypto markets).
  * Ensure splits capture distinct market regimes (bull, bear, sideways) to test robustness.

* **Relevance to Other Documents**:
  * Aligns with `Mining Intraday Risk Factor Collections` by collecting high-frequency data (1-minute) for intraday risk factors, aggregated to daily for NLP alignment.
  * Supports `101 Formulaic Alphas` by providing diverse data for sentiment-based alpha generation (e.g., combining price and news sentiment).

---

### **2. RL Models**

* **Objective**: Experiment with a comprehensive set of RL algorithms, supporting both continuous and discrete action spaces, to optimize portfolio allocation while balancing stability and risk control.

* **Continuous Action Models** (for fine-grained portfolio weight allocation):
  * **DPG (Deterministic Policy Gradient)**: Baseline from SARL, suitable for continuous actions but sensitive to hyperparameter tuning.
  * **DDPG (Deep Deterministic Policy Gradient)**: Extends DPG with actor-critic architecture, improving stability via target networks (from `MOPO_Model-based_Offline_Policy_Optimization.pdf`).
  * **SAC (Soft Actor-Critic)**: Incorporates entropy regularization for better exploration, ideal for handling non-stationary financial markets (inspired by `MOPO` and `Data-Efficient_Hierarchical_Reinforcement_Learning.pdf`).
  * **Advantages**: Allows precise weight adjustments (e.g., 0.523% vs. 0.524%) for optimal portfolio allocation.
  * **Challenges**: Higher computational cost, potential instability in offline settings due to distributional shift (addressed via conservative methods below).

* **Discrete Action Models** (for stability and risk control):
  * **DQN (Deep Q-Network)**: Value-based, discretizes portfolio weights into 1% grids (e.g., 0%, 1%, ..., 100%). Robust for offline learning (from `Conservative_Q-Learning_for_Offline_Reinforcement_Learning.pdf`).
  * **PPO (Proximal Policy Optimization)**: Actor-critic, balances exploration and stability with clipped objectives, suitable for discrete actions (from `Mining_Intraday_Risk_Factor_Collections.pdf`).
  * **Advantages**: Simplifies action space, reduces risk of extreme allocations, and aligns with `Uncertainty-Aware_Strategies` for robust decision-making.
  * **Implementation**:
    ```python
    import gym
    import numpy as np
    from stable_baselines3 import DQN, PPO, SAC

    class PortfolioEnv(gym.Env):
        def __init__(self, data, n_assets):
            self.action_space = gym.spaces.Discrete(101)  # 0% to 100% in 1% steps
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(n_assets * 5 + 100,))  # Price/volume + NLP state
            self.data = data
            self.n_assets = n_assets

        def step(self, action):
            weights = np.zeros(self.n_assets)
            weights[action // 101] = (action % 101) / 100.0  # Discrete to weight
            reward = self.calculate_portfolio_return(weights)
            return self.get_state(), reward, False, {}

    env = PortfolioEnv(data=merged_df, n_assets=len(stocks + coins))
    model_dqn = DQN('MlpPolicy', env, verbose=1)
    model_ppo = PPO('MlpPolicy', env, verbose=1)
    model_sac = SAC('MlpPolicy', env, verbose=1)
    ```

* **Offline RL Enhancements** (from `Offline_Reinforcement_Learning_with_Implicit_Q-Learning.pdf` and `Conservative_Q-Learning_for_Offline_Reinforcement_Learning.pdf`):
  * **CQL (Conservative Q-Learning)**: Regularizes Q-values to avoid overestimation for out-of-distribution actions, critical for offline financial data.
  * **IQL (Implicit Q-Learning)**: Avoids querying unseen actions by estimating state-value functions with expectile regression, improving generalization.
  * **MOPO (Model-based Offline Policy Optimization)**: Uses a reward penalty based on model uncertainty to handle distributional shift, suitable for offline training with static datasets.

* **Hierarchical RL (HRL)** (from `Data-Efficient_Hierarchical_Reinforcement_Learning.pdf` and `Hierarchical_Actor-Critic.pdf`):
  * Implement a two-level hierarchy:
    * **High-Level Policy**: Selects asset allocation strategies (e.g., momentum, mean-reversion) based on market regimes.
    * **Low-Level Policy**: Executes fine-grained (continuous) or discrete (1% grid) weight adjustments.
  * Use off-policy corrections (e.g., HIRO from `Data-Efficient_Hierarchical_Reinforcement_Learning.pdf`) to stabilize training across levels.
  * Example: High-level policy chooses “long BTC, short ETH” based on sentiment, while low-level policy adjusts weights within constraints.

* **Relevance to Other Documents**:
  * Aligns with `Mining_Intraday_Risk_Factor_Collections` by using HPPO for high-level feature weighting and low-level nonlinear factor generation.
  * Incorporates `Uncertainty-Aware_Strategies` by using CQL/IQL to mitigate model uncertainty in offline RL.

---

### **3. State Augmentation**

* **Objective**: Enhance state representation by combining price/volume, text/news, and on-chain/transactional signals with realistic temporal alignment and context-aware processing.

* **Components**:
  * **Price/Volume**: Daily OHLCV from Binance/yfinance, normalized to relative changes (e.g., *yt = vt+1 / vt*).
  * **Text/News**:
    * Use CryptoCompare/CryptoPanic for crypto news, NewsAPI/yfinance for stocks, and X Platform for tweets.
    * Process with NLP models (e.g., Hugging Face transformers, VADER) to extract sentiment scores and topic relevance (e.g., earnings, product launches).
    * Example:
      ```python
      from transformers import pipeline
      from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

      sentiment_analyzer = pipeline('sentiment-analysis')
      vader = SentimentIntensityAnalyzer()

      def process_news(news_df):
          sentiments = []
          for title in news_df['title']:
              hf_score = sentiment_analyzer(title)[0]['score']
              vader_score = vader.polarity_scores(title)['compound']
              sentiments.append({'hf_score': hf_score, 'vader_score': vader_score})
          return pd.DataFrame(sentiments)

      news_df['sentiment'] = process_news(news_df)
      ```
  * **On-Chain/Transactional Data**:
    * Collect from CryptoCompare or blockchain APIs (e.g., Etherscan for ETH transactions).
    * Features: Transaction volume, wallet activity, gas fees, reflecting market dynamics.
  * **Temporal Alignment**:
    * Implement lag windows (e.g., 1–5 days) to capture delayed or anticipatory price reactions (addressing SARL’s immediate impact assumption).
    * Use event-based matching: Align news with specific events (e.g., earnings dates, crypto halving).
    * Example: If news about “BTC ETF approval” appears, align with price movements within a ±3-day window.
  * **Ablation Studies**:
    * Test contributions of each signal (price, news, on-chain) by training RL models with subsets of features.
    * Evaluate impact of lag windows vs. immediate alignment.

* **Improvements Over SARL**:
  * **Context-Aware Filtering**: Replace company name matching with topic modeling (e.g., LDA, BERT-based clustering) to prioritize relevant news (e.g., earnings over store openings).
  * **Sentiment Intensity**: Weight news by sentiment score and mention frequency, unlike SARL’s uniform treatment.
  * **Lag Analysis**: Model temporal effects explicitly, addressing SARL’s unrealistic *t* to *t+1* assumption.
  * **On-Chain Signals**: Incorporate crypto-specific data, absent in SARL, for richer state representations.

* **Relevance to Other Documents**:
  * Aligns with `101 Formulaic Alphas` by using sentiment as a feature for alpha generation.
  * Supports `Mining_Intraday_Risk_Factor_Collections` by integrating sentiment and on-chain data as risk factors.
  * Draws on `Uncertainty-Aware_Strategies` by using subsampling to quantify model uncertainty in state augmentation.

---

### **6-Week Milestones**

1. **Weeks 1–2: Reproducible Pipeline, Price-Only RL Baseline**
   * **Tasks**:
     * Set up data pipeline for stocks (yfinance) and crypto (Binance, CryptoCompare).
     * Aggregate 1-minute data to daily OHLCV, merge with news/tweets.
     * Implement baseline DQN/PPO models with price/volume states.
     * Define action space (discrete 1% grid) and reward (portfolio return).
   * **Deliverables**: Reproducible data pipeline, baseline RL performance (Sharpe, Max Drawdown).

2. **Weeks 3–4: Add Text/Transactional Features with Lag Analysis**
   * **Tasks**:
     * Integrate news (CryptoPanic, NewsAPI) and tweets (X Platform) with sentiment analysis.
     * Collect on-chain data (e.g., transaction volume) via CryptoCompare/Etherscan.
     * Implement lag windows (1–5 days) and event-based alignment.
     * Train RL models (SAC, CQL, IQL) with augmented states.
   * **Deliverables**: State-augmented RL models, preliminary results on feature impact.

3. **Weeks 5–6: Walk-Forward Evaluation, Ablations, Risk-Aware Metrics**
   * **Tasks**:
     * Perform walk-forward evaluation on test/unseen test sets (2023–2025).
     * Conduct ablation studies to isolate contributions of price, news, and on-chain signals.
     * Evaluate using risk-aware metrics (Sharpe, Sortino, Calmar, Max Drawdown, turnover sensitivity).
     * Experiment with HRL (HPPO, HIRO) for high-level strategy selection.
   * **Deliverables**: Final performance report, ablation study results, recommendations for asset universe and risk constraints.

---

### **Evaluation Metrics**

* **Portfolio Metrics**:
  * **Portfolio Value**: Cumulative return over test periods.
  * **Sharpe/Sortino Ratio**: Risk-adjusted returns, emphasizing downside risk (Sortino).
  * **Calmar Ratio**: Return per unit of max drawdown.
  * **Max Drawdown**: Worst-case loss to assess risk exposure.
* **Trading Metrics**:
  * **Turnover**: Measure trading frequency to evaluate transaction cost impact.
  * **Transaction Cost Sensitivity**: Test performance under varying cost assumptions (e.g., 0.1–0.5% per trade).
* **Relevance to Documents**:
  * Aligns with `101 Formulaic Alphas` by focusing on low-turnover, sentiment-driven strategies.
  * Supports `Uncertainty-Aware_Strategies` by incorporating risk-aware metrics to handle model uncertainty.

---

### **Goal and Next Steps**

* **Goal**: Quantify the impact of NLP (news/tweets) and on-chain data on RL-based portfolio management, identifying when and how these signals improve performance over price/volume alone.
* **Next Steps**:
  * Discuss with Prof. Lien to finalize the asset universe (e.g., top 10 stocks from S&P 500, top 5 cryptocurrencies).
  * Define risk constraints (e.g., max 20% drawdown, max 10% allocation per asset).
  * Validate data pipeline and baseline RL models with initial results.

---

### **Additional Notes**

* **Alignment with Other Documents**:
  * **Stable Offline Value Function Learning**: Use KROPE’s bisimulation-based representations to stabilize offline RL training, especially for CQL/IQL models.
  * **Efficient Off-Policy Meta-RL**: Incorporate probabilistic context variables to model market regime shifts, enhancing adaptability in non-stationary markets.
  * **Hierarchical RL**: Leverage HPPO (from `Mining_Intraday_Risk_Factor_Collections`) and HIRO (from `Data-Efficient_Hierarchical_Reinforcement_Learning`) for multi-level decision-making.
  * **Uncertainty-Aware Strategies**: Apply subsampling to quantify model uncertainty, ensuring robust performance across market conditions.

* **Implementation Tips**:
  * Use Stable-Baselines3 or RLlib for RL model implementation to save time.
  * Leverage Hugging Face for advanced NLP (e.g., BERT for topic modeling).
  * Store intermediate data (CSV, parquet) to streamline experimentation.

Nếu bạn cần thêm code chi tiết (e.g., triển khai RL environment, NLP pipeline) hoặc muốn thảo luận sâu hơn về một phần cụ thể (e.g., HRL setup, lag analysis), hãy cho tôi biết!