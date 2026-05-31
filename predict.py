import os
# Force Keras to use the JAX backend for GPU acceleration BEFORE importing Keras
os.environ["KERAS_BACKEND"] = "jax"

import matplotlib
matplotlib.use('Agg') # Force non-interactive backend BEFORE importing plt
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import keras
from tqdm import tqdm

# Ensure this points to your actual FinancialProcessor import
from src.dataGenius.process_data import FinancialProcessor 

# ==========================================
# 1. LOAD THE MODEL AND SCALER
# ==========================================
print("📦 Loading DEEPFIN model and scalers...")
model = keras.models.load_model("weights/deepfin_v2.keras")
scaler_mean = np.load("data/scaler_mean.npy")
scaler_scale = np.load("data/scaler_scale.npy")

# Prevent NaN Division by Zero
scaler_scale[scaler_scale == 0.0] = 1.0

os.makedirs("assets", exist_ok=True)

# ==========================================
# 2. GATHER HISTORICAL (INPUT) DATA
# ==========================================
history = []
history_extended = []
extended_years_loaded = []
ticker = "AAPL"

print(f"🔍 Extracting {ticker}'s financial states via C++ Engine...")

# T_in: 3 years of input sequence (2022, 2023, 2024)
for year in tqdm([2022, 2023, 2024]):
    res = FinancialProcessor(ticker, year).process_ticker()
    if res is not None and "tensor" in res:
        history.append(res["tensor"])
    else:
        raise ValueError(f"Failed to process {ticker} for year {year}. Did you download the SEC data?")

# Extract extended historical data for 2024, 2025, 2026 for comparison testing
print(f"📊 Extracting extended historical data for 2024-2026...")
for year in tqdm([2024, 2025, 2026]):
    res = FinancialProcessor(ticker, year).process_ticker()
    if res is not None and "tensor" in res:
        history_extended.append(res["tensor"])
        extended_years_loaded.append(year)
    else:
        print(f"⚠️ Warning: Could not process {ticker} for year {year}. Skipping...")

# Convert to NumPy arrays
input_seq = np.array(history).reshape(1, 3, -1)
extended_seq = np.array(history_extended).reshape(1, len(history_extended), -1) if history_extended else None

# ==========================================
# 3. BASELINE AI PREDICTION (JAX JIT-COMPILED)
# ==========================================
print("🧠 Running Baseline Neural Forecast...")
input_scaled = (input_seq - scaler_mean) / scaler_scale
prediction_raw = model.predict(input_scaled)
prediction = (prediction_raw * scaler_scale) + scaler_mean

# ==========================================
# 4. QUANTIVESTA™ $5 BILLION CAUSAL PERTURBATION
# ==========================================
print("⚡ Injecting $5,000,000,000 CAPEX Perturbation into 2024...")
sim_input = input_scaled.copy()
sim_input[0, 2, 4] += (5000000000 / scaler_scale[4]) 

sim_prediction_raw = model.predict(sim_input)
sim_prediction = (sim_prediction_raw * scaler_scale) + scaler_mean

# ==========================================
# 5. DATA EXTRACTION & TERMINAL TABLE 
# ==========================================
# T_in: 3 years of input sequence (2022, 2023, 2024)
for year in tqdm([2022, 2023, 2024]):
    res = FinancialProcessor(ticker, year).process_ticker()
    if res is not None and "tensor" in res:
        history.append(res["tensor"])
        
        # 🛑 ADD THIS DEBUG LINE:
        print(f"\n[DEBUG {year}] Raw Python Rev: {res['raw_data']['Revenue']} | Tensor[0]: {res['tensor'][0]}")
        
    else:
        raise ValueError(f"Failed to process {ticker} for year {year}.")
hist_rev = input_seq[0, :, 0] / 1e9
base_pred_rev = prediction[0, :, 0] / 1e9
sim_pred_rev = sim_prediction[0, :, 0] / 1e9

historical_years = [2022, 2023, 2024]
forecast_years = [2024, 2025, 2026, 2027] # 2024 overlaps to connect the lines

# Connect the forecast lines back to the final historical point (2024)
baseline_line = [hist_rev[-1]] + list(base_pred_rev)
simulated_line = [hist_rev[-1]] + list(sim_pred_rev)

# Print Raw Value Table
results_df = pd.DataFrame({
    "Fiscal Year": historical_years + forecast_years[1:],
    "Data Source": ["Historical", "Historical", "Historical", "AI Forecast", "AI Forecast", "AI Forecast"],
    "Baseline Rev ($B)": list(hist_rev) + list(base_pred_rev),
    "Simulated Rev ($B)": list(hist_rev) + list(sim_pred_rev)
})
results_df["Delta ($B)"] = results_df["Simulated Rev ($B)"] - results_df["Baseline Rev ($B)"]
pd.options.display.float_format = '{:,.2f}'.format

print("\n" + "="*85)
print(f" QUANTIVESTA™ RAW VALUE FORECAST: {ticker} (+$5B CAPEX INJECTION IN 2024)")
print("="*85)
print(results_df.to_string(index=False))
print("="*85 + "\n")

# ==========================================
# 6. GENERATE VISUALIZATIONS
# ==========================================
print("📊 Generating Visualizations...")

# --- PLOT 1: Baseline vs Simulation ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(historical_years, hist_rev, marker='o', color='#4a4a4a', linewidth=2.5, markersize=8, label='Historical Revenue')
ax.plot(forecast_years, baseline_line, marker='s', color='#1f77b4', linewidth=2.5, linestyle='--', markersize=8, label='Baseline AI Forecast')
ax.plot(forecast_years, simulated_line, marker='^', color='#2ca02c', linewidth=2.5, linestyle='--', markersize=8, label='Simulated (+$5B CAPEX)')
ax.axvline(x=2024, color='black', linestyle=':', linewidth=2, label='Inference Point (T_0 = 2024)')

ax.set_title(f'QUANTIVESTA™: {ticker} AI Forecast vs Causal Perturbation', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Fiscal Year', fontsize=12)
ax.set_ylabel('Revenue (Billions $)', fontsize=12)
ax.set_xticks(historical_years + forecast_years[1:]) 
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='upper left', fontsize=10, frameon=True, shadow=True)

fig.tight_layout()
save_path = f'assets/{ticker}_prediction.png'
fig.savefig(save_path, dpi=300) 
plt.close(fig) 
print(f"📈 Primary graph successfully saved as '{save_path}'")

# --- PLOT 2: Historical vs AI Validation ---
if extended_seq is not None and len(extended_years_loaded) > 0:
    # 1. Map the AI forecast to specific years so we don't misalign the timeline
    forecast_map = {
        2024: hist_rev[-1],       # Anchor point
        2025: base_pred_rev[0],   # Year 1 pred
        2026: base_pred_rev[1],   # Year 2 pred
        2027: base_pred_rev[2]    # Year 3 pred
    }
    
    # 2. Extract actual historical revenue
    extended_hist_rev = extended_seq[0, :, 0] / 1e9
    
    # 3. Pull only the forecasted values that match the years we successfully loaded
    aligned_forecast = [forecast_map[year] for year in extended_years_loaded if year in forecast_map]
    
    # Create comparison plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    ax2.plot(extended_years_loaded, extended_hist_rev, marker='o', color="#8b1b1b", 
             linewidth=2.5, markersize=8, label='Actual Historical Revenue')
    ax2.plot(extended_years_loaded, aligned_forecast, marker='s', color='#1f77b4', 
             linewidth=2.5, linestyle='--', markersize=8, label='Baseline AI Forecast')
    
    ax2.set_title(f'QUANTIVESTA™: {ticker} Actual Validation vs AI Forecast', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Fiscal Year', fontsize=12)
    ax2.set_ylabel('Revenue (Billions $)', fontsize=12)
    ax2.set_xticks(extended_years_loaded)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper left', fontsize=10, frameon=True, shadow=True)
    
    fig2.tight_layout()
    comparison_path = f'assets/{ticker}_historical_comparison.png'
    fig2.savefig(comparison_path, dpi=300)
    plt.close(fig2)
    print(f"📈 Validation comparison graph saved as '{comparison_path}'")