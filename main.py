from fastapi import FastAPI
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

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
# ✅ FINAL PREDICT (MANUAL INPUT)
# ================================
@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])

        df_scaled = scaler.transform(df)
        raw_prob = model.predict_proba(df_scaled)[0][1]

        # 🔥 Improved probability scaling
        prob = float(raw_prob)

        # 🔥 Signal logic
        if prob > 0.6:
            signal = "STRONG BUY"
        elif prob > 0.5:
            signal = "BUY"
        elif prob > 0.4:
            signal = "HOLD"
        else:
            signal = "SELL"

        return {
            "probability": round(prob, 3),
            "raw_probability": round(float(raw_prob), 3),
            "signal": signal
        }

    except Exception as e:
        return {"error": str(e)}

# ================================
# PERFORMANCE API
# ================================
@app.get("/performance")
def get_performance():
    df = pd.read_csv("performance.csv")
    df = df.sort_values("date")
    return {"data": df.to_dict(orient="records")}
