import os
os.environ["KERAS_BACKEND"] = "jax"
import numpy as np
import pandas as pd
import keras
from tqdm import tqdm

from src.dataGenius.process_data import FinancialProcessor 

print("📦 Loading DEEPFIN V3 Autoregressive model...")
model = keras.models.load_model("weights/deepfin_v3.keras")

# ==========================================
# V3 SECTOR MAP (SIC CODES)
# ==========================================
# In a full production environment, you would load this from a master CSV.
# For now, we seed the giants to ensure the V3 Sector Routing categorizes them perfectly.
sic_map = {
    "JPM": 6021, "BAC": 6021, "WFC": 6021, "C": 6021,  # Commercial Banks (Idx 7)
    "GS": 6211, "MS": 6211,                            # Investment Banks (Idx 7)
    "AAPL": 3571, "MSFT": 7372, "GOOGL": 7370,         # Tech / Software (Idx 3 & 8)
    "AMZN": 5961, "WMT": 5331, "HD": 5211              # Retail (Idx 6)
}

# 1. Gather all available tickers
available_tickers = [d for d in os.listdir("data") if os.path.isdir(os.path.join("data", d))]
available_tickers = [t for t in available_tickers if t.isupper()]
print(f"🔍 Found {len(available_tickers)} total companies.")

results = []

print("⚙️ Running GLOBAL V3 Backtest (Sector-Aware)...")
for ticker in tqdm(available_tickers):
    try:
        # Fetch the SIC code, default to 0 (Standard Corp) if not in our map
        company_sic = sic_map.get(ticker, 0)
        
        # Step A: Get 2024 Data
        actual_2024_res = FinancialProcessor(ticker, 2024, sic=company_sic).process_ticker()
        if actual_2024_res is None:
            continue
            
        raw = actual_2024_res["raw_data"]
        
        # Keep the Data Error filter to ignore parsing bugs
        if raw["Revenue"] <= 0.1: 
            continue
            
        # Step B: Gather Input History (2021-2023)
        history = []
        for year in [2021, 2022, 2023]:
            res = FinancialProcessor(ticker, year, sic=company_sic).process_ticker()
            if res is not None and "tensor" in res:
                history.append(res["tensor"])
            else:
                raise ValueError("Missing history")

        # Step C: Run Inference with Instance Normalization
        actual_2024_rev = raw["Revenue"] / 1e9
        input_seq = np.array(history).reshape(1, 3, -1)
        
        # Dynamically scale this specific company against its own history
        scaler_max = np.max(np.abs(input_seq), axis=1, keepdims=True)
        scaler_max[scaler_max == 0.0] = 1.0
        
        input_scaled = input_seq / scaler_max
        input_scaled = np.clip(input_scaled, -10.0, 10.0)
        
        prediction_raw = model.predict(input_scaled, verbose=0)
        
        if ticker in ["JPM", "BAC", "WFC", "C", "GS"]:
            print(f"Debug: {ticker} prediction raw output: {prediction_raw}")
            
        # Extract the V3 prediction (Shape is now [Batch, 1, 23])
        predicted_rev_scaled = prediction_raw[0, 0, 0] 
        max_revenue_dollars = scaler_max[0, 0, 0] 
        
        predicted_2024_rev = (predicted_rev_scaled * max_revenue_dollars) / 1e9
        
        # Step D: Metrics
        error_dollars = predicted_2024_rev - actual_2024_rev
        error_percentage = abs(error_dollars / actual_2024_rev) * 100

        results.append({
            "Ticker": ticker,
            "Actual 2024 Rev ($B)": actual_2024_rev,
            "AI Predicted 2024 Rev ($B)": predicted_2024_rev,
            "Error ($B)": error_dollars,
            "Accuracy (%)": max(0, 100 - error_percentage)
        })

    except Exception as e:
        pass # Skip incomplete sequences

# 2. Final Report Generation
df_results = pd.DataFrame(results)

if len(df_results) == 0:
    print("\n❌ No valid results generated.")
    exit()

df_results = df_results.sort_values(by="Actual 2024 Rev ($B)", ascending=False)

print("\n" + "="*85)
print(f" 🏆 DEEPFIN V3 GLOBAL BACKTEST ({len(df_results)} COMPANIES)")
print("="*85)

# Calculate Accuracy Metrics
median_acc = df_results["Accuracy (%)"].median()
mean_acc = df_results["Accuracy (%)"].mean()

print(f"Macro Median Accuracy: {median_acc:.2f}%")
print(f"Macro Mean Accuracy:   {mean_acc:.2f}%")
print("-" * 85)

# Automatically hunt down the big banks to check the hypothesis
print("🏦 ACCURACY FOR TOP FINANCIAL GIANTS (JPM, BAC, WFC, C, GS):")
banks = ["JPM", "BAC", "WFC", "C", "GS"]
bank_results = df_results[df_results["Ticker"].isin(banks)]
if not bank_results.empty:
    print(bank_results.to_string(index=False, float_format="%.2f"))
else:
    print("Data for top banks not found in the current pipeline output.")

# Identify the absolute "Golden Companies" where the model is a genius
matches_99 = df_results[df_results["Accuracy (%)"] > 99]
matches_95 = df_results[df_results["Accuracy (%)"] > 95]
matches_90 = df_results[df_results["Accuracy (%)"] > 90]
print("-" * 85)
print(f"⭐ Model achieved >99% accuracy for {len(matches_99)} companies.")
print(f"⭐ Model achieved >95% accuracy for {len(matches_95)} companies.")
print(f"⭐ Model achieved >90% accuracy for {len(matches_90)} companies.")
print("="*85)

df_results.to_csv("assets/v3_global_backtest.csv", index=False)