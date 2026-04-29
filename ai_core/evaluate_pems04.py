import os
import sys
import torch
import numpy as np
import pickle
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset

# Cấu hình trỏ tuyệt đối môi trường nội bộ
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.graph_wavenet import GraphWaveNet

# Fix Windows console UTF-8 printing
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def calculate_metrics(target, prediction):
    """Hàm tính MAE, RMSE, MAPE cho một tập dữ liệu cụ thể"""
    mae = np.mean(np.abs(target - prediction))
    rmse = np.sqrt(np.mean(np.square(target - prediction)))
    
    # Tính MAPE tránh chia cho 0 (chỉ tính với lưu lượng > 0)
    mask = target > 0.5
    if np.sum(mask) == 0:
        mape = 0.0
    else:
        mape = np.mean(np.abs(target[mask] - prediction[mask]) / target[mask]) * 100
        
    return mae, rmse, mape

def evaluate_pems04():
    print("=" * 85)
    print(" [~] BAT DAU QUA TRINH DANH GIA MO HINH PEMS04 (GNN) ")
    print("=" * 85)

    # 1. Cấu hình đường dẫn và thiết bị
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'checkpoints', 'model_pems04_best.pt')
    scaler_path = os.path.join(base_dir, 'data', 'processed', 'pems04', 'scaler.pkl')
    test_data_path = os.path.join(base_dir, 'data', 'processed', 'pems04', 'test.npz')
    adj_path = os.path.join(base_dir, 'data', 'processed', 'pems04', 'adj_mx.pkl')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Thiết bị sử dụng: {device}")

    # 2. Tải Ma trận kề
    if not os.path.exists(adj_path):
        print(f"[!] Lỗi: Không tìm thấy file {adj_path}")
        return
    with open(adj_path, 'rb') as f:
        adj_mx = pickle.load(f)
    adj_tensor = torch.tensor(adj_mx, dtype=torch.float32).to(device)

    # 3. Tải mô hình
    print("[*] Khoi tao mo hinh GraphWaveNet (307 Nodes, 5 In_Dim, 12 Out_Dim)...")
    model = GraphWaveNet(num_nodes=307, in_dim=5, out_dim=12, supports=[adj_tensor], addaptadj=True).to(device)
    
    if not os.path.exists(model_path):
        print(f"[!] Lỗi: Không tìm thấy file trọng số tại {model_path}")
        return
        
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"    -> Da nạp thành công bộ trọng số: model_pems04_best.pt")
    except Exception as e:
        print(f"[!] Lỗi khi nạp trọng số: {e}")
        return

    # 4. Tải Scaler
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        mean = scaler['mean'][0] # Mean của Flow
        std = scaler['std'][0]   # Std của Flow
    except Exception as e:
        print(f"[!] Lỗi khi nạp scaler: {e}")
        return

    # 5. Tải dữ liệu Test
    try:
        data = np.load(test_data_path)
        x_test = data['x']  # (Samples, 12, 307, 5)
        y_test = data['y']  # (Samples, 12, 307, 5)
    except Exception as e:
        print(f"[!] Lỗi khi nạp dữ liệu test: {e}")
        return

    # 6. Chạy Inference
    batch_size = 64
    tensor_x = torch.FloatTensor(x_test)
    tensor_y = torch.FloatTensor(y_test)
    dataloader = DataLoader(TensorDataset(tensor_x, tensor_y), batch_size=batch_size, shuffle=False)

    preds_list = []
    truths_list = []
    
    print(f"[*] Dang chay du bao tren toan bo tap Test ({x_test.shape[0]} mau)...")
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            
            # CHUYEN DOI TENSOR (PERMUTE) NHU LUC TRAIN
            # (Batch, 12, 307, 5) -> (Batch, 5, 307, 12)
            x_input = batch_x.permute(0, 3, 2, 1)
            
            output = model(x_input) # Output: (Batch, 307, 12)
            
            # Lay target Flow (index 0) và đưa về (Batch, 307, 12)
            target = batch_y[..., 0].permute(0, 2, 1)
            
            preds_list.append(output.cpu().numpy())
            truths_list.append(target.numpy())

    preds = np.concatenate(preds_list, axis=0) # (Samples, 307, 12)
    truths = np.concatenate(truths_list, axis=0) # (Samples, 307, 12)

    # 7. GIAI CHUAN HOA
    preds_real = (preds * std) + mean
    truths_real = (truths * std) + mean
    preds_real = np.maximum(preds_real, 0)
    truths_real = np.maximum(truths_real, 0)

    # 8. HIEN THI KET QUA CHI TIET
    print("\n" + "=" * 85)
    print(" [x] KET QUA DANH GIA MO HINH PEMS04 - MULTI-HORIZON ")
    print("=" * 85)
    print(f"{'Horizon':<15} | {'MAE':<10} | {'RMSE':<10} | {'MAPE (%)':<10} | {'R-squared':<10}")
    print("-" * 85)

    # Tinh toán cho các mốc cụ thể (Lưu ý: Index 2 = 15p, Index 5 = 30p, Index 11 = 60p)
    horizons = [2, 5, 11]
    horizon_names = ["15 Phút", "30 Phút", "60 Phút"]
    
    for idx, h_name in zip(horizons, horizon_names):
        h_mae, h_rmse, h_mape = calculate_metrics(truths_real[:, :, idx], preds_real[:, :, idx])
        h_r2 = r2_score(truths_real[:, :, idx].flatten(), preds_real[:, :, idx].flatten())
        print(f"{h_name:<15} | {h_mae:<10.2f} | {h_rmse:<10.2f} | {h_mape:<10.2f} | {h_r2:<10.4f}")

    # Tính toán trung bình toàn bộ (Overall)
    o_mae, o_rmse, o_mape = calculate_metrics(truths_real, preds_real)
    o_r2 = r2_score(truths_real.flatten(), preds_real.flatten())
    print("-" * 85)
    print(f"{'TRUNG BINH':<15} | {o_mae:<10.2f} | {o_rmse:<10.2f} | {o_mape:<10.2f} | {o_r2:<10.4f}")
    print("=" * 85)
    print("[*] Luu y: MAE tinh bang don vi Luu luong (Flow/5min).\n")

if __name__ == "__main__":
    evaluate_pems04()
