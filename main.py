from fastapi import FastAPI
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf

app = FastAPI()

# ================================
# CORS
# ================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# LOAD MODEL
# ================================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ================================
# ROOT
# ================================
@app.get("/")
def home():
    return {"message": "API is working"}

# ================================
# PREDICTION API (IMPROVED)
# ================================
@app.get("/predict/{symbol}")
def predict(symbol: str):
    try:
        df = yf.download(symbol, period="60d", interval="1d", progress=False)

        if df is None or df.empty or len(df) < 30:
            return {"error": f"No data found for symbol: {symbol}"}

        # Feature engineering
        df["Return_5"] = df["Close"].pct_change(5)
        df["Return_20"] = df["Close"].pct_change(20)

        df["MA_10"] = df["Close"].rolling(10).mean()
        df["MA_20"] = df["Close"].rolling(20).mean()
        df["MA_ratio"] = df["MA_10"] / df["MA_20"]

        df["Volatility"] = df["Close"].pct_change().rolling(10).std()
        df["Volume_Spike"] = df["Volume"] / df["Volume"].rolling(10).mean()

        df = df.dropna()

        latest = df.iloc[-1]

        features = [[
            latest["Return_5"],
            latest["Return_20"],
            latest["MA_ratio"],
            latest["Volatility"],
            latest["Volume_Spike"]
        ]]

        # Model prediction
        scaled = scaler.transform(features)
        raw_prob = model.predict_proba(scaled)[0][1]

        # 🔥 FIX: Amplify probability spread
        prob = float(raw_prob)
        prob = (prob - 0.45) * 3
        prob = max(0, min(1, prob))  # clamp 0–1

        # 🔥 Improved signal logic
        if prob > 0.6:
            signal = "STRONG BUY"
        elif prob > 0.5:
            signal = "BUY"
        elif prob > 0.4:
            signal = "HOLD"
        else:
            signal = "SELL"

        return {
            "symbol": symbol,
            "probability": round(prob, 3),
            "raw_probability": round(float(raw_prob), 3),
            "signal": signal
        }

    except Exception as e:
        return {"error": str(e)}

# ================================
# LOAD & CLEAN PERFORMANCE DATA
# ================================
def load_performance_data():
    df = pd.read_csv("performance.csv")

    df.columns = [col.lower() for col in df.columns]

    if "date" not in df.columns:
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    df["strategy"] = df["strategy"].ffill()
    df["nifty"] = df["nifty"].ffill()

    df = df.dropna(subset=["strategy", "nifty"])

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    return df

perf_df = load_performance_data()

# ================================
# PERFORMANCE API
# ================================
@app.get("/performance")
def get_performance():
    df = pd.read_csv("performance.csv")
    df = df.sort_values("date")
    return {"data": df.to_dict(orient="records")}