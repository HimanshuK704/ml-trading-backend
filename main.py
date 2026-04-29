from fastapi import FastAPI
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (safe for project)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.get("/")
def home():
    return {"message": "API is working"}


@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    
    df_scaled = scaler.transform(df)
    prob = model.predict_proba(df_scaled)[0][1]
    
    return {"probability": float(prob)}

perf_df = pd.read_csv("performance.csv")

@app.get("/performance")
def get_performance():
    return {"data": perf_df.to_dict(orient="records")}