import os
import h5py
import pickle
import numpy as np
import pandas as pd
from typing import Tuple

def load_graph_data(pkl_filename: str) -> Tuple[list, dict, np.ndarray]:
    """Load METR-LA dependency graph (adjacency matrix)."""
    with open(pkl_filename, 'rb') as f:
        sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding='latin1')
    return sensor_ids, sensor_id_to_ind, adj_mx

def load_h5_data(h5_filename: str) -> np.ndarray:
    """Load METR-LA traffic speed dataset from HDF5."""
    with h5py.File(h5_filename, 'r') as f:
        data = f['df']['block0_values'][:]
    return data

def generate_dataset(data: np.ndarray, seq_len: int, pre_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Apply sliding window mechanism for historical input and future prediction."""
    x, y = [], []
    num_samples = data.shape[0] - seq_len - pre_len + 1
    for i in range(num_samples):
        x.append(data[i : i + seq_len, :])
        y.append(data[i + seq_len : i + seq_len + pre_len, :])
    return np.array(x), np.array(y)

def preprocess():
    """Main preprocessing logic pipeline conformant to rules/tech.md."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, 'data', 'raw', 'metr-la')
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    h5_path = os.path.join(raw_dir, 'METR-LA.h5')
    pkl_path = os.path.join(raw_dir, 'adj_METR-LA.pkl')
    
    # Validation: Ensure raw datasets exist before progressing
    if not os.path.exists(h5_path) or not os.path.exists(pkl_path):
        print(f"Error: Raw dataset not found at {raw_dir}")
        print("Please download and place 'metr-la.h5' and 'adj_METR-LA.pkl' there.")
        return

    # 1. Load Data
    data = load_h5_data(h5_path)
    sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(pkl_path)
    
    # 2. Temporal Split (70/10/20 Rule)
    num_samples = data.shape[0]
    num_train = int(num_samples * 0.7)
    num_val = int(num_samples * 0.1)
    
    train_data = data[:num_train]
    val_data = data[num_train:num_train + num_val]
    test_data = data[num_train + num_val:]
    
    # 3. Z-Score Normalization (Trained strictly on train_data to avoid data leakage)
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)
    std[std == 0] = 1.0  # Prevent potential division by zero
    
    train_norm = (train_data - mean) / std
    val_norm = (val_data - mean) / std
    test_norm = (test_data - mean) / std
    
    # 4. Generate Sliding Windows (12 input timesteps, 12 target timesteps target)
    seq_len = 12
    pre_len = 12
    
    x_train, y_train = generate_dataset(train_norm, seq_len, pre_len)
    x_val, y_val = generate_dataset(val_norm, seq_len, pre_len)
    x_test, y_test = generate_dataset(test_norm, seq_len, pre_len)
    
    # 5. Persist Output Artefacts
    np.savez_compressed(os.path.join(processed_dir, 'train.npz'), x=x_train, y=y_train)
    np.savez_compressed(os.path.join(processed_dir, 'val.npz'), x=x_val, y=y_val)
    np.savez_compressed(os.path.join(processed_dir, 'test.npz'), x=x_test, y=y_test)
    
    with open(os.path.join(processed_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump({'mean': mean, 'std': std}, f)
        
    with open(os.path.join(processed_dir, 'adj_mx.pkl'), 'wb') as f:
        pickle.dump(adj_mx, f)
        
    # 6. Output Reporting
    print("Preprocessing successful.")
    print(f"Train Shape: Input X={x_train.shape}, Target Y={y_train.shape}")
    print(f"Val Shape: Input X={x_val.shape}, Target Y={y_val.shape}")
    print(f"Test Shape: Input X={x_test.shape}, Target Y={y_test.shape}")

if __name__ == "__main__":
    preprocess()
