import os
import sys
import torch

# Fix Windows console UTF-8 printing
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import pickle
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models.graph_wavenet import GraphWaveNet

def evaluate_model():
    print("=" * 60)
    print(" [~] BAT DAU QUA TRINH KIEM THU MO HINH GRAPH WAVENET ")
    print("=" * 60)

    # 1. Cấu hình đường dẫn và thiết bị
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'checkpoints', 'model_best.pt')
    scaler_path = os.path.join(base_dir, 'data', 'processed', 'scaler.pkl')
    test_data_path = os.path.join(base_dir, 'data', 'processed', 'test.npz')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Thiết bị tính toán (Device): {device}")

    # 2. Tải mô hình
    print("[*] Nạp kiến trúc mô hình (Loading Model)...")
    model = GraphWaveNet(num_nodes=207, in_dim=1, out_dim=12, addaptadj=True).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("    -> Nạp thành công trọng số: model_best.pt")
    except Exception as e:
        print(f"[!] Lỗi nạp mô hình: {e}")
        return

    # 3. Tải Scaler (Bộ chuẩn hóa)
    print("[*] Nạp bộ giải chuẩn hóa (Scaler)...")
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        mean, std = scaler['mean'], scaler['std']
    except Exception as e:
        print(f"[!] Lỗi nạp scaler: {e}")
        return

    # 4. Tải dữ liệu Test
    print("[*] Nạp dữ liệu Test (Test Dataset)...")
    try:
        data = np.load(test_data_path)
        x_test = data['x']  # (Samples, 12, 207)
        y_test = data['y']  # (Samples, 12, 207)
        print(f"    -> Tổng số mẫu kiểm thử: {x_test.shape[0]}")
    except Exception as e:
        print(f"[!] Lỗi nạp test data: {e}")
        return

    # 5. Khởi tạo Batching để tránh tràn RAM
    batch_size = 64
    tensor_x = torch.FloatTensor(x_test)
    tensor_y = torch.FloatTensor(y_test)
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Giai đoạn gộp và lưu Cache để tránh việc phải chạy lại AI lâu
    temp_preds_path = os.path.join(base_dir, 'preds_temp.npy')
    temp_truths_path = os.path.join(base_dir, 'truths_temp.npy')

    if os.path.exists(temp_preds_path) and os.path.exists(temp_truths_path):
        print("[*] Tìm thấy kết quả suy luận cũ (Cache). Bỏ qua bước chạy AI...")
        preds = np.load(temp_preds_path)
        truths = np.load(temp_truths_path)
    else:
        print(f"[*] Tiến hành suy luận (Inference) theo từng batch (Size: {batch_size})...")
        preds_list = []
        truths_list = []
        total_batches = len(dataloader)
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(dataloader):
                batch_x = batch_x.to(device)
                tensor_in = batch_x.unsqueeze(1).transpose(2, 3) 
                output = model(tensor_in)  
                preds_list.append(output.cpu().numpy())
                truths_list.append(batch_y.numpy())
                if (i + 1) % 1 == 0 or (i + 1) == total_batches:
                    print(f"    -> Đang xử lý: {i + 1}/{total_batches} batches ({(i + 1)/total_batches*100:.1f}%)")
        
        preds = np.concatenate(preds_list, axis=0)  # (Samples, 207, 12) hoặc (Samples, 12, 207)
        truths = np.concatenate(truths_list, axis=0)
        # Lưu lại để nếu có lỗi công thức phía dưới bạn không phải đợi 15p nữa
        np.save(temp_preds_path, preds)
        np.save(temp_truths_path, truths)

    print("[*] Hoàn tất suy luận! Tiến hành xử lý giải chuẩn hóa (Denormalization)...")
    
    # SỬA LỖI ĐỒNG BỘ HÌNH DÁNG (SHAPE):
    # Đảm bảo preds và truths có cùng chiều (Samples, 12, 207)
    if preds.shape[1] == 207: # Nếu preds là (Samples, 207, 12) theo cấu trúc GNN cũ
        preds = np.transpose(preds, (0, 2, 1))

    # Lúc này cả preds và truths đều lả (Samples, 12, 207)
    mean_reshaped = mean.reshape(1, 1, 207)
    std_reshaped = std.reshape(1, 1, 207)

    preds_real = (preds * std_reshaped) + mean_reshaped
    truths_real = (truths * std_reshaped) + mean_reshaped

    # Đưa các giá trị âm (do sai số nhẹ của hàm loss) về 0 bằng maximum
    preds_real = np.maximum(preds_real, 0)
    truths_real = np.maximum(truths_real, 0)


    # 6. Tính toán các chỉ số (Metrics)
    print("\n" + "=" * 60)
    print(" [x] KET QUA DANH GIA (TEST METRICS) ")
    print("=" * 60)

    # -- MAE (Mean Absolute Error)
    mae = np.mean(np.abs(preds_real - truths_real))
    
    # -- RMSE (Root Mean Squared Error)
    rmse = np.sqrt(np.mean(np.square(preds_real - truths_real)))
    
    # -- MAPE (Mean Absolute Percentage Error)
    # Loại bỏ các mẫu truth == 0 để tránh chia cho số không (Divide by Zero)
    mask = truths_real > 0.1  # Do vận tốc thực tế, ta loại > 0.1 mph để an toàn
    if np.sum(mask) == 0:
        mape = 0.0
    else:
        mape = np.mean(np.abs(preds_real[mask] - truths_real[mask]) / truths_real[mask]) * 100

    print(f"[*] MAE  (Sai so tuyet doi trung binh)  : {mae:.2f} (mph)")
    print(f"[*] RMSE (Sai so binh phuong trung binh): {rmse:.2f} (mph)")
    print(f"[*] MAPE (Sai so phan tram tuyet doi)   : {mape:.2f} %")
    print("=" * 60)

if __name__ == "__main__":
    evaluate_model()
