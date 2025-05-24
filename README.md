# 📈 Stock Price Prediction Using Random Forest

This project uses **machine learning** and **technical analysis** to predict stock closing prices based on historical market data. It employs a **Random Forest Regressor** for prediction and provides intuitive **visualizations** for enhanced data interpretation. Developed and executed using Visual Studio Code.

---

## 🔍 Overview

* 📊 Retrieves 5 years of historical stock data from Yahoo Finance using the `yfinance` API
* 🧮 Calculates **technical indicators**: Simple Moving Averages (SMA) and Relative Strength Index (RSI)
* 🧠 Trains a **Random Forest Regression** model to forecast next-day stock prices
* 📈 Produces visualizations for:

  * Price trends & trading volume
  * Technical indicators (SMA, RSI)
  * Feature correlation
  * Actual vs predicted stock prices
  * Model error distribution

---

## 🚀 Getting Started

💻 Development Environment

Developed and tested using Visual Studio Code
Python version: 3.7+

### 📦 Installation

Ensure you have Python 3.7+ installed. Then install the necessary dependencies:

```bash
pip install numpy pandas yfinance scikit-learn matplotlib seaborn
```

### ▶️ How to Run

```bash
python stock_predictor.py
```

You will be prompted to enter a valid stock ticker (e.g., `AAPL` for Apple Inc.).

---

## 📂 Features

### 📥 Data Collection (Expanded)

This project uses **real-time financial data** provided by [Yahoo Finance](https://finance.yahoo.com/) through the `yfinance` library. Below is a detailed breakdown:

#### ✅ Key Aspects of Data Collection

* **Ticker Input**: Accepts any valid public stock ticker (e.g., `MSFT`, `AMZN`, `NFLX`)
* **Time Range**: Retrieves 5 years of **daily interval** data
* **Fields Collected**:

  * `Open` – Price at market open
  * `High` – Intraday high price
  * `Low` – Intraday low price
  * `Close` – Price at market close
  * `Adj Close` – Adjusted for dividends/splits
  * `Volume` – Shares traded

#### 🔄 Data Preprocessing

* **Missing values**: Cleaned using forward-fill or drop strategies
* **Datetime formatting**: Converted to `datetime64` for index alignment
* **Feature engineering**: Creates new columns for SMA and RSI indicators
* **Label shifting**: Creates a `target` column representing the **next-day close price**, used for supervised learning

#### 📦 Example Code

```python
import yfinance as yf

ticker = 'AAPL'
df = yf.download(ticker, period='5y', interval='1d')

# Create target variable (next day close)
df['Target'] = df['Close'].shift(-1)
```

This forms the foundation for building technical indicators and feeding data into the machine learning pipeline.

---

### 📊 Technical Indicators

Used in both feature engineering and visualization:

* `SMA_20`: 20-day Simple Moving Average – captures short-term trends
* `SMA_50`: 50-day SMA – reflects medium-term movement
* `RSI`: Relative Strength Index – detects momentum shifts and potential reversals

These are used as **predictive features** and also visualized to show trading signals.

---

### 🤖 Machine Learning Model

**Random Forest Regressor** is chosen for its robustness and ability to handle non-linear relationships:

* Ensemble of 100+ decision trees
* Reduces overfitting through bagging
* Does not require feature scaling
* Predicts the **next day's closing price**

#### 📈 Evaluation Metrics

* **Mean Squared Error (MSE)** – Penalizes large errors
* **R² Score** – Indicates how well predictions explain the variance in data

---

### 📊 Visualization Tools

Uses `matplotlib` and `seaborn` to create clear, readable charts:

* 📉 Stock price history over time
* 📊 Volume bar chart overlay
* 📈 Trend lines for SMA
* 🔥 RSI momentum plot
* 🔍 Heatmap showing correlation between features
* 🟢 Scatter plot: Actual vs Predicted price
* 🔴 Histogram: Error distribution

---

📈 Example Output
yaml
Copy
Edit
Model Evaluation:
  Mean Squared Error: 2.87
  R² Score: 0.942

Predicted next day closing price for TSLA: $173.22
Visual outputs are saved or displayed interactively using Visual Studio Code.

## 🧠 Future Improvements

* Implement **deep learning models** (e.g., LSTM, GRU)
* Use **multi-day forecasting** (5–10 day lookahead)
* Add **fundamental data** (PE ratios, earnings reports)
* Incorporate **sentiment analysis** from Twitter or news feeds
* Create a **Streamlit web app** for real-time predictions

---

## 📌 Disclaimer

This project is for **educational and research purposes only**. It does **not constitute financial advice**, and no trading or investment decisions should be based on its output.

---

## 🙌 Acknowledgements

* [Yahoo Finance](https://finance.yahoo.com/) – data provider
* [`yfinance`](https://github.com/ranaroussi/yfinance) – API wrapper
* `scikit-learn` – ML framework
* `matplotlib` & `seaborn` – data visualization
