import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pickle

# Cấu hình trỏ tuyệt đối môi trường nội bộ
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.graph_wavenet import GraphWaveNet

# Fix Windows console UTF-8 printing
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# CẤU HÌNH THÔNG SỐ (HYPERPARAMETERS)
# ==========================================
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
PATIENCE = 15 # Early Stopping

NUM_NODES = 307
IN_DIM = 5    # Flow, Occ, Speed, TimeOfDay, DayOfWeek
OUT_DIM = 12  # Dự đoán 12 mốc thời gian

def load_pems04_data(data_path, batch_size):
    """
    Tiến trình nạp bộ dataset PeMS04.
    Raw Shape: X=(N, 12, 307, 5), Y=(N, 12, 307, 5)
    """
    print(f"[*] Nạp dữ liệu từ: {data_path}")
    train_data = np.load(os.path.join(data_path, 'train.npz'))
    val_data = np.load(os.path.join(data_path, 'val.npz'))

    # Load Adjacency Matrix
    with open(os.path.join(data_path, 'adj_mx.pkl'), 'rb') as f:
        adj_mx = pickle.load(f)

    # Convert thành Tensor
    x_train = torch.FloatTensor(train_data['x'])
    y_train = torch.FloatTensor(train_data['y'])
    
    x_val = torch.FloatTensor(val_data['x'])
    y_val = torch.FloatTensor(val_data['y'])

    # Tạo DataLoader
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, adj_mx

def main():
    print("=" * 60)
    print(" [~] KHOI DONG TIEN TRINH HUAN LUYEN (TRAIN PEMS04) ")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Thiết bị (Device): {device}")

    # 1. Nạp dữ liệu
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'processed', 'pems04')
    checkpoint_dir = os.path.join(base_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    train_loader, val_loader, adj_mx = load_pems04_data(data_path, BATCH_SIZE)
    adj_tensor = torch.tensor(adj_mx, dtype=torch.float32).to(device)

    # 2. Khởi tạo mô hình GraphWaveNet
    print(f"[*] Khởi tạo GraphWaveNet: Nodes={NUM_NODES}, In_Dim={IN_DIM}, Out_Dim={OUT_DIM}")
    model = GraphWaveNet(
        num_nodes=NUM_NODES, 
        in_dim=IN_DIM, 
        out_dim=OUT_DIM, 
        supports=[adj_tensor], 
        addaptadj=True
    ).to(device)

    # 3. Khai báo Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss() # MAE Loss

    # 4. Cấu hình luồng huấn luyện
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = os.path.join(checkpoint_dir, 'model_pems04_best.pt')

    print("\n[*] Bắt đầu huấn luyện...")
    for epoch in range(EPOCHS):
        # ----------------- TRAIN -----------------
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # BIẾN ĐỔI TENSOR VÀO (CRITICAL STEP)
            # Từ (Batch, 12, 307, 5) -> (Batch, 5, 307, 12)
            batch_x_permuted = batch_x.permute(0, 3, 2, 1)
            
            # CHỌN TENSOR ĐÍCH (Target)
            # Ta chỉ dự đoán đặc trưng chính (Flow - index 0). (Batch, 12, 307, 5) -> (Batch, 12, 307)
            batch_y_target = batch_y[..., 0]
            
            optimizer.zero_grad()
            
            # Forward Pass -> Output shape: (Batch, 307, 12)
            output = model(batch_x_permuted)
            
            # Đối chiếu với Target: Cần permute Target về (Batch, 307, 12)
            loss = criterion(output, batch_y_target.permute(0, 2, 1))
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_loader.dataset)

        # ----------------- VALIDATION -----------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                # Biến đổi Tensor
                batch_x_permuted = batch_x.permute(0, 3, 2, 1)
                batch_y_target = batch_y[..., 0]
                
                output = model(batch_x_permuted)
                loss = criterion(output, batch_y_target.permute(0, 2, 1))
                val_loss += loss.item() * batch_x.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        # ----------------- IN LOG & EARLY STOPPING -----------------
        print(f"[Epoch {epoch+1:03d}/{EPOCHS}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            # print(f"  -> Đã lưu Checkpoint tốt nhất!")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\n[!] Cảnh báo: Kích hoạt Early Stopping tại Epoch {epoch+1}.")
                break
                
    print("\n" + "=" * 60)
    print(" [x] HOÀN TẤT HUẤN LUYỆN!")
    print(f" [*] Model tốt nhất đã được lưu tại: {best_model_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
