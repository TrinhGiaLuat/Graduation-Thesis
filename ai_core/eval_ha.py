import os
import numpy as np
import pickle

def calculate_metrics(target, prediction):
    """Tính MAE, RMSE, MAPE chuẩn học thuật"""
    mae = np.mean(np.abs(target - prediction))
    rmse = np.sqrt(np.mean(np.square(target - prediction)))
    mask = target > 0.5
    mape = np.mean(np.abs(target[mask] - prediction[mask]) / target[mask]) * 100 if np.sum(mask) > 0 else 0.0
    return mae, rmse, mape

def evaluate_ha_pro():
    print("=" * 60)
    print(" [~] ĐÁNH GIÁ MÔ HÌNH HA (Lịch sử theo Thứ & Giờ) ")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data', 'processed', 'pems04')
    
    # 1. Tải Scaler
    with open(os.path.join(data_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    mean_flow, std_flow = scaler['mean'][0], scaler['std'][0]

    # 2. Tải dữ liệu
    print("[*] Đang tải dữ liệu Train và Test...")
    train_data = np.load(os.path.join(data_dir, 'train.npz'))
    test_data = np.load(os.path.join(data_dir, 'test.npz'))

    y_train = train_data['y'] # (Samples, 12, 307, 5)
    y_test = test_data['y']

    # 3. Gom nhóm lịch sử (Training Phase của HA)
    print("[*] Đang xây dựng Ma trận Lịch sử (Historical Matrix)...")
    # Chúng ta dùng bước thời gian đầu tiên (t=0) của mỗi sample làm mốc
    train_flow = y_train[:, 0, :, 0] * std_flow + mean_flow  # (Samples, 307)
    train_tod = np.round(y_train[:, 0, 0, 3], 4)             # Time of day (Samples,)
    train_dow = np.round(y_train[:, 0, 0, 4], 4)             # Day of week (Samples,)

    # Dictionary lưu trữ: (day_of_week, time_of_day) -> list các flow (để tính trung bình)
    history_dict = {}
    for i in range(len(train_flow)):
        dow = train_dow[i]
        tod = train_tod[i]
        key = (dow, tod)
        if key not in history_dict:
            history_dict[key] = []
        history_dict[key].append(train_flow[i])

    # Tính trung bình (Mean) cho mỗi nhóm
    ha_model = {}
    for key, values in history_dict.items():
        ha_model[key] = np.mean(values, axis=0) # (307,)

    # 4. Dự báo (Testing Phase)
    print("[*] Đang tiến hành dự báo trên tập Test...")
    test_truth = y_test[:, :, :, 0] * std_flow + mean_flow # (Samples, 12, 307)
    test_preds = np.zeros_like(test_truth)

    missing_keys = 0
    for i in range(y_test.shape[0]):
        for t in range(12):
            dow = np.round(y_test[i, t, 0, 4], 4)
            tod = np.round(y_test[i, t, 0, 3], 4)
            key = (dow, tod)
            
            if key in ha_model:
                test_preds[i, t, :] = ha_model[key]
            else:
                # Nếu không tìm thấy chính xác Thứ+Giờ (hiếm gặp), fallback lấy trung bình của Giờ đó trên tất cả các Thứ
                tod_only_keys = [k for k in ha_model.keys() if k[1] == tod]
                if tod_only_keys:
                    avg_tod = np.mean([ha_model[k] for k in tod_only_keys], axis=0)
                    test_preds[i, t, :] = avg_tod
                else:
                    # Fallback cuối cùng: trung bình toàn bộ train (rất hiếm)
                    test_preds[i, t, :] = np.mean(train_flow, axis=0)
                missing_keys += 1

    if missing_keys > 0:
        print(f"[!] Cảnh báo: Có {missing_keys} bước không khớp lịch sử chính xác, dùng Fallback.")

    # 5. Tính Metrics
    mae, rmse, mape = calculate_metrics(test_truth, test_preds)

    print("\n" + "=" * 40)
    print(" KẾT QUẢ MÔ HÌNH HA (Chuẩn):")
    print(f" - MAE:  {mae:.4f}")
    print(f" - RMSE: {rmse:.4f}")
    print(f" - MAPE: {mape:.2f}%")
    print("=" * 40)

if __name__ == "__main__":
    evaluate_ha_pro()
