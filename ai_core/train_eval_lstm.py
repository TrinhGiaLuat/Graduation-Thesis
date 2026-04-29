import os
import torch
import torch.nn as nn
import numpy as np
import pickle
from torch.utils.data import DataLoader, TensorDataset
import time
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# =========================================================================
# KIẾN TRÚC SEQ2SEQ LSTM DÀNH CHO DỰ BÁO CHUỖI THỜI GIAN
# =========================================================================
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers=1):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, hidden, cell):
        # x: (batch_size, 1, output_dim)
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.fc(output)
        return prediction, hidden, cell

class Seq2SeqLSTM(nn.Module):
    def __init__(self, num_nodes, in_dim=1, hidden_dim=64, out_len=12, num_layers=1):
        super(Seq2SeqLSTM, self).__init__()
        self.num_nodes = num_nodes
        self.out_len = out_len
        # Xem toàn bộ 307 trạm như 307 features của 1 bước thời gian
        input_dim = num_nodes * in_dim 
        
        self.encoder = Encoder(input_dim, hidden_dim, num_layers)
        self.decoder = Decoder(input_dim, hidden_dim, num_layers)

    def forward(self, x):
        # x: (batch, seq_len, nodes, in_dim) -> (batch, seq_len, nodes * in_dim)
        batch_size, seq_len, num_nodes, in_dim = x.size()
        x = x.view(batch_size, seq_len, -1)
        
        # Encoder nén thông tin
        hidden, cell = self.encoder(x)
        
        # Mồi cho Decoder (lấy bước cuối cùng của input)
        decoder_input = x[:, -1:, :] # (batch, 1, nodes * in_dim)
        
        outputs = []
        for t in range(self.out_len):
            prediction, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs.append(prediction)
            decoder_input = prediction # Auto-regressive
            
        outputs = torch.cat(outputs, dim=1) # (batch, out_len, nodes * in_dim)
        outputs = outputs.view(batch_size, self.out_len, num_nodes, in_dim)
        return outputs.squeeze(-1) # (batch, out_len, nodes)

# =========================================================================
# HÀM ĐÁNH GIÁ & HUẤN LUYỆN
# =========================================================================
def calculate_metrics(target, prediction):
    mae = np.mean(np.abs(target - prediction))
    rmse = np.sqrt(np.mean(np.square(target - prediction)))
    mask = target > 0.5
    mape = np.mean(np.abs(target[mask] - prediction[mask]) / target[mask]) * 100 if np.sum(mask) > 0 else 0
    return mae, rmse, mape

def train_eval_lstm():
    print("=" * 70)
    print(" [~] HUẤN LUYỆN LSTM (ENCODER-DECODER) CHO DỰ BÁO GIAO THÔNG ")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Đang sử dụng thiết bị: {device}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Tải dữ liệu
    print("[*] Đang kiểm tra vị trí dữ liệu...")
    
    # Logic tìm file thông minh: Ưu tiên tìm ngay tại thư mục hiện tại (Dành cho Kaggle/Flat folder)
    if os.path.exists(os.path.join(base_dir, 'train.npz')):
        data_dir = base_dir
        print("    -> Tìm thấy dữ liệu tại thư mục gốc.")
    else:
        # Nếu không thấy thì tìm theo cấu trúc thư mục dự án cũ
        data_dir = os.path.join(base_dir, 'data', 'processed', 'pems04')
        print(f"    -> Tìm kiếm dữ liệu tại: {data_dir}")
    
    train_data = np.load(os.path.join(data_dir, 'train.npz'))
    val_data = np.load(os.path.join(data_dir, 'val.npz'))
    test_data = np.load(os.path.join(data_dir, 'test.npz'))
    
    with open(os.path.join(data_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    mean_flow, std_flow = scaler['mean'][0], scaler['std'][0]

    # Chỉ dùng Flow (feature 0)
    x_train = torch.FloatTensor(train_data['x'][:, :, :, 0:1])
    y_train = torch.FloatTensor(train_data['y'][:, :, :, 0])
    x_val = torch.FloatTensor(val_data['x'][:, :, :, 0:1])
    y_val = torch.FloatTensor(val_data['y'][:, :, :, 0])
    x_test = torch.FloatTensor(test_data['x'][:, :, :, 0:1])
    y_test = torch.FloatTensor(test_data['y'][:, :, :, 0])

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=64, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=64, shuffle=False)

    # 2. Khởi tạo mô hình
    model = Seq2SeqLSTM(num_nodes=307, in_dim=1, hidden_dim=256, out_len=12, num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.L1Loss() # MAE Loss thường tốt hơn MSE cho dự báo lưu lượng

    # 3. Huấn luyện có Early Stopping
    epochs = 50 # Số vòng lặp chuẩn
    patience = 5 # Số vòng chịu đựng nếu Val Loss không giảm
    best_val_loss = float('inf')
    counter = 0

    print("[*] Bắt đầu quá trình huấn luyện...")
    for epoch in range(epochs):
        start_time = time.time()
        
        # --- Train ---
        model.train()
        train_loss = 0
        for b_x, b_y in train_loader:
            b_x, b_y = b_x.to(device), b_y.to(device)
            optimizer.zero_grad()
            output = model(b_x)
            loss = criterion(output, b_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # --- Validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for b_x, b_y in val_loader:
                b_x, b_y = b_x.to(device), b_y.to(device)
                output = model(b_x)
                loss = criterion(output, b_y)
                val_loss += loss.item()
                
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        time_taken = time.time() - start_time
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {time_taken:.1f}s")
        
        # --- Early Stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # Lưu lại model tốt nhất (Lưu ý: Không bắt buộc lưu đĩa để tiết kiệm thời gian, lưu memory là đủ)
            best_model_state = model.state_dict()
        else:
            counter += 1
            if counter >= patience:
                print(f"[!] Early stopping kích hoạt tại epoch {epoch+1}")
                break

    # 4. Lưu model tốt nhất ra file
    torch.save(best_model_state, os.path.join(base_dir, 'lstm_best_model.pt'))
    print(f"[*] Đã lưu trọng số mô hình tốt nhất vào: {os.path.join(base_dir, 'lstm_best_model.pt')}")

    # 5. Đánh giá trên tập Test
    print("[*] Đang nạp lại trọng số tốt nhất và đánh giá trên tập Test...")
    model.load_state_dict(best_model_state)
    model.eval()
    
    all_preds, all_truths = [], []
    with torch.no_grad():
        for b_x, b_y in test_loader:
            b_x = b_x.to(device)
            output = model(b_x).cpu().numpy()
            all_preds.append(output)
            all_truths.append(b_y.numpy())

    preds = np.concatenate(all_preds, axis=0) * std_flow + mean_flow
    truths = np.concatenate(all_truths, axis=0) * std_flow + mean_flow

    mae, rmse, mape = calculate_metrics(truths, preds)

    print("\n" + "=" * 40)
    print(" KẾT QUẢ MÔ HÌNH LSTM (SEQ2SEQ):")
    print(f" - MAE:  {mae:.4f}")
    print(f" - RMSE: {rmse:.4f}")
    print(f" - MAPE: {mape:.2f}%")
    print("=" * 40)

    # 6. Lưu kết quả ra file txt
    with open(os.path.join(base_dir, 'lstm_results.txt'), 'w', encoding='utf-8') as f:
        f.write("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH LSTM (SEQ2SEQ)\n")
        f.write("="*40 + "\n")
        f.write(f"MAE:  {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MAPE: {mape:.2f}%\n")
        f.write("="*40 + "\n")
    print(f"[*] Đã lưu kết quả đánh giá vào: {os.path.join(base_dir, 'lstm_results.txt')}")

if __name__ == "__main__":
    train_eval_lstm()
