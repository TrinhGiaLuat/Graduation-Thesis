import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Cấu hình trỏ tuyệt đối môi trường nội bộ
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.graph_wavenet import GraphWaveNet

def load_data(data_path, batch_size):
    """
    Tiến trình nạp bộ dataset lịch sử và hoán vị (Transpose) tạo vùng Tensor tương thích kiến trúc GNN.
    TRỌNG YẾU TỪ BÁO CÁO: Reshape trục thời gian/không gian 
    (N, 12, 207) -> (Batch, 1, 207, 12).
    """
    train_data = np.load(os.path.join(data_path, 'train.npz'))
    val_data = np.load(os.path.join(data_path, 'val.npz'))
    test_data = np.load(os.path.join(data_path, 'test.npz'))

    def reshape_tensor(x, y):
        # Lớp x Shape Input: (N, 12, 207) -> Chèn dimension -> (N, 1, 12, 207) -> (N, 1, 207, 12)
        x_tensor = torch.FloatTensor(x).unsqueeze(1).transpose(2, 3) 
        y_tensor = torch.FloatTensor(y)
        return x_tensor, y_tensor

    x_train, y_train = reshape_tensor(train_data['x'], train_data['y'])
    x_val, y_val = reshape_tensor(val_data['x'], val_data['y'])
    x_test, y_test = reshape_tensor(test_data['x'], test_data['y'])

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def train():
    """
    Hội đồng khởi chạy hệ thống huấn luyện vòng đời Mô hình Graph WaveNet 
    bằng phương pháp Đạo hàm nghịch (Backpropagation) và Early Stopping.
    """
    # 1. Nhận diện thiết bị Device (Fallback Configuration)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Bắt đầu quy trình huấn luyện đồ thị mạng Nơ-ron trên thiết bị: {device}")

    # 2. Hyperparameters Config (Chuẩn định mức thực tiễn học máy)
    EPOCHS = 100
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    PATIENCE = 15
    NUM_NODES = 207

    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    checkpoint_dir = os.path.join(base_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 3. Kéo luồng DataLoader (Hạ tầng Mini-batch gradient descent)
    train_loader, val_loader, test_loader = load_data(processed_dir, BATCH_SIZE)

    # 4. Thiếp lập sơ đồ Kiến trúc Object - Trình cực tiểu hoá sai số
    model = GraphWaveNet(num_nodes=NUM_NODES, in_dim=1, out_dim=12, addaptadj=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
    
    # Đo lường tính trượt khoảng cách thực tiễn cực kì rủi ro sinh lỗi biên - Sử dụng Huber Loss
    criterion = nn.HuberLoss() 

    # 5. Phân vùng sự kiện dừng khẩn cấp (Early Stopping Flag logic)
    best_val_loss = float('inf')
    epochs_no_improve = 0

    print("Thiết lập luồng duyệt qua các chu kỳ (Epochs)...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            # Quá trình lan truyền tín hiệu (Forward Pass)
            output = model(batch_x)
            
            # Tính toán Vector dốc (Backward Pass & Optimization)
            loss = criterion(output, batch_y.permute(0, 2, 1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_loader.dataset)

        # Trích lọc đo lường Validation Loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x)
                loss = criterion(output, batch_y.permute(0, 2, 1))
                val_loss += loss.item() * batch_x.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        print(f"[Epoch {epoch+1:03d}/{EPOCHS}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Ghi nhận kho State_Dict cực tiểu rủi ro (Best Checkpoint Override)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'model_best.pt'))
            print("  -> Snapshot mô hình đạt trạng thái cực tiểu cục bộ được sao lưu bảo mật.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Cảnh báo tự động: Kích hoạt cờ Early Stopping do lỗi hệ thống hội tụ tại {PATIENCE} chu kỳ.")
                break

if __name__ == "__main__":
    train()
