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
# PREDICTION API
# ================================
@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    df_scaled = scaler.transform(df)
    prob = model.predict_proba(df_scaled)[0][1]
    return {"probability": float(prob)}

# ================================
# LOAD & CLEAN PERFORMANCE DATA
# ================================
def load_performance_data():
    df = pd.read_csv("performance.csv")

    # Ensure correct column names
    df.columns = [col.lower() for col in df.columns]

    # Rename if needed
    if "date" not in df.columns:
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})

    # Convert to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Sort properly
    df = df.sort_values("date")

    # Handle missing values
    df["strategy"] = df["strategy"].ffill()
    df["nifty"] = df["nifty"].ffill()

    # Drop any remaining NaN
    df = df.dropna(subset=["strategy", "nifty"])

    # Convert date to string for JSON
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    return df

# Load once at startup
perf_df = load_performance_data()

# ================================
# PERFORMANCE API
# ================================
@app.get("/performance")
def get_performance():
    return {"data": perf_df.to_dict(orient="records")}