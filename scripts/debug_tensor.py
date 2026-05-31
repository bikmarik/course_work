import numpy as np

print("🕵️‍♂️ DEEPFIN Data Detective Initiated...")

# Load the tensors
X = np.load("data/X_train.npy")
Y = np.load("data/Y_train.npy")

def scan_tensor(tensor, name):
    if np.isnan(tensor).any() or np.isinf(tensor).any():
        print(f"\n❌ CORRUPTION DETECTED IN {name}:")
        for i in range(tensor.shape[2]):
            nans = np.isnan(tensor[:, :, i]).sum()
            infs = np.isinf(tensor[:, :, i]).sum()
            if nans > 0 or infs > 0:
                print(f"  -> Feature [{i}]: {nans} NaNs, {infs} Infinities")
    else:
        print(f"\n✅ {name} is perfectly clean.")

scan_tensor(X, "X_train")
scan_tensor(Y, "Y_train")