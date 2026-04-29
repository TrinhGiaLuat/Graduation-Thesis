import os
import sys
import numpy as np
import pandas as pd
import pickle

# Fix Windows console UTF-8 printing
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def generate_time_features(num_samples, num_nodes):
    """Tạo 2 đặc trưng thời gian: time_of_day và day_of_week"""
    steps_per_day = 288 # 24 giờ * 60 phút / 5 phút
    
    # Tạo chuỗi thời gian
    time_of_day = (np.arange(num_samples) % steps_per_day).astype(np.float32)
    day_of_week = ((np.arange(num_samples) // steps_per_day) % 7).astype(np.float32)
    
    # Broadcast (Nhân bản) cho tất cả các trạm
    # Hình dáng: (num_samples, num_nodes, 1)
    tod_expanded = np.tile(time_of_day[:, np.newaxis, np.newaxis], (1, num_nodes, 1))
    dow_expanded = np.tile(day_of_week[:, np.newaxis, np.newaxis], (1, num_nodes, 1))
    
    return tod_expanded, dow_expanded

def build_adjacency_matrix(csv_path, num_nodes):
    """Xây dựng ma trận kề bằng Gaussian Kernel từ khoảng cách"""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[!] Lỗi đọc file CSV: {e}")
        return np.eye(num_nodes)
    
    # Tạo ma trận chi phí (khoảng cách)
    dist_mx = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    dist_mx[:] = np.inf
    
    # Đọc cạnh
    for _, row in df.iterrows():
        i, j = int(row['from']), int(row['to'])
        if i < num_nodes and j < num_nodes:
            dist_mx[i, j] = row['cost']
    
    # Gaussian Kernel: W_ij = exp(- (dist_ij / std)^2)
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    
    adj_mx = np.exp(-np.square(dist_mx / std))
    
    # Đường chéo chính bằng 1 (Self-loop)
    np.fill_diagonal(adj_mx, 1.0)
    
    # Các khoảng cách inf -> trọng số 0
    adj_mx[adj_mx < 1e-5] = 0.0
    
    return adj_mx

def generate_dataset(data, seq_len, pre_len):
    """Sliding Window: (seq_x) -> (seq_y)"""
    x, y = [], []
    num_samples = data.shape[0]
    for i in range(num_samples - seq_len - pre_len + 1):
        x.append(data[i : i + seq_len])
        y.append(data[i + seq_len : i + seq_len + pre_len])
    
    return np.array(x), np.array(y)

def main():
    print("=" * 60)
    print(" [~] KHOI DONG TIEN XU LY DU LIEU PEMS04 (MULTIMODAL)")
    print("=" * 60)
    
    # Cấu hình đường dẫn
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, 'data', 'raw', 'pems04')
    processed_dir = os.path.join(base_dir, 'data', 'processed', 'pems04')
    os.makedirs(processed_dir, exist_ok=True)
    
    npz_path = os.path.join(raw_dir, 'PEMS04.npz')
    csv_path = os.path.join(raw_dir, 'PEMS04.csv')
    
    # ---------------------------------------------------------
    # 1. ĐỌC VÀ NÂNG CẤP ĐẶC TRƯNG (FEATURE ENGINEERING)
    # ---------------------------------------------------------
    print("[*] 1. Đọc dữ liệu gốc và Feature Engineering...")
    data_raw = np.load(npz_path)['data']  # (16992, 307, 3)
    num_samples, num_nodes, num_features = data_raw.shape
    
    print(f"    -> Shape gốc: {data_raw.shape}")
    
    tod_expanded, dow_expanded = generate_time_features(num_samples, num_nodes)
    data_5d = np.concatenate([data_raw, tod_expanded, dow_expanded], axis=-1)
    
    print(f"    -> Shape sau khi nạp đặc trưng thời gian: {data_5d.shape}")
    
    # ---------------------------------------------------------
    # 2. CHIA TẬP DỮ LIỆU (70/10/20)
    # ---------------------------------------------------------
    print("\n[*] 2. Chia tập dữ liệu (70% Train, 10% Val, 20% Test)...")
    train_split = int(num_samples * 0.7)
    val_split = int(num_samples * 0.8)
    
    train_data = data_5d[:train_split]
    val_data = data_5d[train_split:val_split]
    test_data = data_5d[val_split:]
    
    print(f"    -> Mẫu Train: {train_data.shape[0]}")
    print(f"    -> Mẫu Val  : {val_data.shape[0]}")
    print(f"    -> Mẫu Test : {test_data.shape[0]}")
    
    # ---------------------------------------------------------
    # 3. CHUẨN HÓA (NORMALIZATION)
    # ---------------------------------------------------------
    print("\n[*] 3. Chuẩn hóa dữ liệu...")
    # Chỉ tính mean, std cho 3 đặc trưng đầu (Flow, Occ, Speed) của tập Train
    mean = np.mean(train_data[..., :3], axis=(0, 1), keepdims=True)
    std = np.std(train_data[..., :3], axis=(0, 1), keepdims=True)
    
    # Tránh chia cho 0
    std[std == 0] = 1.0
    
    # Hàm Helper để chuẩn hóa
    def normalize_data(data_subset):
        norm = np.copy(data_subset)
        # Z-Score cho 3 đặc trưng đầu
        norm[..., :3] = (data_subset[..., :3] - mean) / std
        # Min-Max cho thời gian về [0, 1]
        norm[..., 3] = data_subset[..., 3] / 288.0
        norm[..., 4] = data_subset[..., 4] / 7.0
        return norm

    train_norm = normalize_data(train_data)
    val_norm = normalize_data(val_data)
    test_norm = normalize_data(test_data)
    
    # ---------------------------------------------------------
    # 4. SLIDING WINDOW (12-12)
    # ---------------------------------------------------------
    print("\n[*] 4. Áp dụng Sliding Window (seq_x=12, seq_y=12)...")
    seq_len = 12
    pre_len = 12
    
    x_train, y_train = generate_dataset(train_norm, seq_len, pre_len)
    x_val, y_val = generate_dataset(val_norm, seq_len, pre_len)
    x_test, y_test = generate_dataset(test_norm, seq_len, pre_len)
    
    print(f"    -> Shape của tập Train sau Sliding Window: X={x_train.shape}, Y={y_train.shape}")
    print(f"    -> Shape của tập Val   sau Sliding Window: X={x_val.shape}, Y={y_val.shape}")
    print(f"    -> Shape của tập Test  sau Sliding Window: X={x_test.shape}, Y={y_test.shape}")
    
    # ---------------------------------------------------------
    # 5. XÂY DỰNG MA TRẬN KỀ
    # ---------------------------------------------------------
    print("\n[*] 5. Tính toán Ma trận kề (Adjacency Matrix)...")
    adj_mx = build_adjacency_matrix(csv_path, num_nodes)
    print(f"    -> Shape Ma trận kề: {adj_mx.shape}")
    
    # ---------------------------------------------------------
    # 6. LƯU KẾT QUẢ
    # ---------------------------------------------------------
    print("\n[*] 6. Ghi xuất file (Saving)...")
    np.savez_compressed(os.path.join(processed_dir, 'train.npz'), x=x_train, y=y_train)
    np.savez_compressed(os.path.join(processed_dir, 'val.npz'), x=x_val, y=y_val)
    np.savez_compressed(os.path.join(processed_dir, 'test.npz'), x=x_test, y=y_test)
    
    # Chỉ lưu Mean/Std của 3 đặc trưng vật lý để phục vụ giải chuẩn hóa sau này
    scaler_dict = {
        'mean': mean.squeeze(),
        'std': std.squeeze()
    }
    with open(os.path.join(processed_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler_dict, f)
        
    with open(os.path.join(processed_dir, 'adj_mx.pkl'), 'wb') as f:
        pickle.dump(adj_mx, f)
        
    print(f"    -> Đã lưu các tập NPZ vào: {processed_dir}")
    print("=" * 60)
    print(" [x] HOÀN TẤT TIỀN XỬ LÝ CHO PEMS04!")

if __name__ == '__main__':
    main()
