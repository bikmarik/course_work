import os
# Force Keras to use the JAX backend for GPU acceleration
os.environ["KERAS_BACKEND"] = "jax"

import matplotlib
matplotlib.use('Agg') # Safe headless plotting (Must be before pyplot)
import matplotlib.pyplot as plt

import keras
from keras import layers
import numpy as np

# ==========================================
# 1. LOAD & SCRUB THE TRAINING DATA
# ==========================================
print("📦 Loading DEEPFIN V3 Data...")
X_raw = np.load("data/X_train.npy").astype(np.float32)
Y_raw = np.load("data/Y_train.npy").astype(np.float32)

print(f"Original Shape -> X: {X_raw.shape}, Y: {Y_raw.shape}")

# 1. Standard Scrubber (Catches actual NaN and Inf)
X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
Y_raw = np.nan_to_num(Y_raw, nan=0.0, posinf=0.0, neginf=0.0)

# 2. THE AGGRESSIVE TRUNCATOR 
MAX_POSSIBLE_DOLLARS = 1e14

X_raw = np.clip(X_raw, -MAX_POSSIBLE_DOLLARS, MAX_POSSIBLE_DOLLARS)
Y_raw = np.clip(Y_raw, -MAX_POSSIBLE_DOLLARS, MAX_POSSIBLE_DOLLARS)

# ==========================================
# 2. INSTANCE NORMALIZATION (Per-Company Scaling)
# ==========================================
print("⚖️ Applying Instance Normalization...")

# Find the maximum value per company, per feature across their 3-year history
scaler_max = np.max(np.abs(X_raw), axis=1, keepdims=True)

# Prevent division by zero
scaler_max[scaler_max == 0.0] = 1.0

# Scale each company entirely by its OWN historical size
X_scaled = X_raw / scaler_max
Y_scaled = Y_raw / scaler_max 

# Safe clip to allow up to 10x growth without exploding gradients
X_scaled = np.clip(X_scaled, -10.0, 10.0)
Y_scaled = np.clip(Y_scaled, -10.0, 10.0)

# ==========================================
# 3. DEFINE THE AUTOREGRESSIVE ENGINE (3 -> 1)
# ==========================================
print("\n🧠 Building V3 Autoregressive Architecture...")
n_features = X_raw.shape[2] 

model = keras.Sequential([
    layers.Input(shape=(3, n_features)),
    
    # 1. Feature Extraction: Look at the 3-year curve
    layers.LSTM(128, activation='tanh', return_sequences=True),
    layers.Dropout(0.2),
    
    # 2. State Aggregation: Condense the 3 years into a single "thought vector"
    # return_sequences=False forces it to output a single dense state
    layers.LSTM(64, activation='tanh', return_sequences=False),
    layers.Dropout(0.2),
    
    # 3. Predict the exact next year's features (Output shape: [Batch, Features])
    layers.Dense(n_features),
    
    # 4. Reshape to perfectly match Y_train's (Batch, 1, Features) format
    layers.Reshape((1, n_features))
])

# THE SHIELD: clipnorm=1.0 physically prevents exploding gradients 
optimizer = keras.optimizers.AdamW(learning_rate=0.001, clipnorm=1.0)
model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
model.summary()

# ==========================================
# 4. TRAIN WITH EARLY STOPPING
# ==========================================
print("\n🔥 Training DEEPFIN V3...")

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=15, 
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_scaled, 
    Y_scaled, 
    epochs=100, 
    batch_size=32, 
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# ==========================================
# 5. SAVE WEIGHTS
# ==========================================
os.makedirs("weights", exist_ok=True)
os.makedirs("assets", exist_ok=True)

# Saved as V3 to prevent overwriting your old Seq2Seq weights
model.save("weights/deepfin_v3.keras")

print("\n✅ V3 Autoregressive Model saved perfectly.")

# ==========================================
# 6. VISUALIZE TRAINING LOSS
# ==========================================
print("📊 Rendering Training Convergence Graph...")

plt.clf() 
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(history.history['loss'], label='Training Loss (MSE)', color='#d62728', linewidth=2.5)

if 'val_loss' in history.history:
    ax.plot(history.history['val_loss'], label='Validation Loss', color='#1f77b4', linewidth=2.5)

ax.set_title('DEEPFIN V3: Training Convergence', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss (Scaled MSE)', fontsize=12)
ax.legend(loc='upper right', frameon=True, shadow=True)
ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('assets/training_loss.png', dpi=300)
plt.close(fig) 

print("📈 Training loss graph fully rendered and saved to 'assets/training_loss.png'")