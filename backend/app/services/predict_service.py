import os
import sys
import torch
import pickle
import numpy as np

# Thiết lập hệ thống chèn phân mục đường dẫn để kích hoạt liên kết Backend - AI Core
# Xử lý cả môi trường Windows nội bộ và môi trường Docker Container
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ai_core_windows = os.path.join(base_dir, 'ai_core')
ai_core_docker = '/ai_core'

ai_core_path = ai_core_docker if os.path.exists(ai_core_docker) else ai_core_windows

if ai_core_path not in sys.path:
    sys.path.append(ai_core_path)

import importlib.util

try:
    # Tránh xung đột Namespace với file models.py của Backend bằng vòng cung (Dynamic Module Import)
    gw_path = os.path.join(ai_core_path, 'models', 'graph_wavenet.py')
    spec = importlib.util.spec_from_file_location("graph_wavenet_ai", gw_path)
    gw_module = importlib.util.module_from_spec(spec)
    sys.modules["graph_wavenet_ai"] = gw_module
    spec.loader.exec_module(gw_module)
    GraphWaveNet = gw_module.GraphWaveNet
except Exception as e:
    raise ImportError(f"[Nghiêm Trọng] Đứt gãy luồng tham chiếu hệ thống trí tuệ nhân tạo: {str(e)}")

class PredictService:
    """
    Cấu trúc Dịch vụ Cốt lõi (Core Inference Service).
    Khai triển mô hình Graph WaveNet cung ứng phương thức truy vấn dự báo cho Endpoint.
    """
    def __init__(self):
        # Mặc định cấu hình tương thích thiết bị CUDA tối đa hoá tính toán song song
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_nodes = 207
        self.seq_len = 12
        
        # Liên kết tài nguyên đã qua huẩn luyện tại phân vùng Data Layer
        self.model_path = os.path.join(ai_core_path, 'checkpoints', 'model_best.pt')
        self.scaler_path = os.path.join(ai_core_path, 'data', 'processed', 'scaler.pkl')
        
        self.model = None
        self.scaler = None
        
        self._load_dependencies()

    def _load_dependencies(self):
        """
        Nạp cấu trúc đồ thị không gian và trích dẫn bộ trọng số Pre-Trained.
        """
        if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
            print("[PredictService] Cảnh báo: Định vị tệp tin Checklist / Trọng số AI thất bại.")
            return
            
        try:
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
                
            self.model = GraphWaveNet(num_nodes=self.num_nodes, in_dim=1, out_dim=12, addaptadj=True).to(self.device)
            state_dict = torch.load(self.model_path, map_location=self.device)
            
            # Đồng bộ không gian bộ nhớ State Dict vào Instance Object
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
        except Exception as e:
            print(f"[PredictService] Xảy ra ngoại lệ trong đường truyền đọc Model Pytorch: {str(e)}")

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Phương thức kích tiến suy diễn dựa vào mảng lưu lượng vận tốc giao thông lịch sử.
        
        Args:
            input_data (np.ndarray): Khuôn mẫu dữ liệu gốc Shape -> (12, 207)
            
        Returns:
            np.ndarray: Dữ liệu vận tốc dự báo thực Shape -> (12, 207)
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Tiến trình dự báo chưa sẵn sàng do khuyết mô hình/Scalar Parameter.")
            
        if input_data.shape != (self.seq_len, self.num_nodes):
            raise ValueError(f"Khối lượng chiều dị kì. Expected {(self.seq_len, self.num_nodes)}, received {input_data.shape}")

        # 1. Pipeline Normalization (Trấn áp nhiễu loạn ngoại lai)
        mean = self.scaler['mean']
        std = self.scaler['std']
        norm_data = (input_data - mean) / std
        
        # 2. Hoán vị định dạng cho Tensor (Reshape Space-Time Axis)
        # Hình thái nguồn (12, 207) -> Tịnh tiến -> (1, 1, 207, 12) theo hệ quy chiếu PyTorch
        tensor_in = torch.FloatTensor(norm_data).unsqueeze(0).unsqueeze(0).transpose(2, 3).to(self.device)

        # 3. Mở cổng kết xuất kết quả AI (Khử tính năng Gradient backprop)
        with torch.no_grad():
            output_tensor = self.model(tensor_in)  
            
        # Hình thái trả về từ GraphWaveNet là (Batch=1, Nodes=207, Seq=12)
        # Squeeze dim 0 -> (207, 12)
        pred_norm = output_tensor.squeeze(0).cpu().numpy()

        # 4. Denormalization
        # Do pred_norm là (207, 12) mà std/mean là (207,), ta phải reshape std/mean thành (207, 1) để numpy tự động Broadcast sang dimension còn lại.
        std_reshaped = std.reshape(207, 1)
        mean_reshaped = mean.reshape(207, 1)
        
        pred_real = (pred_norm * std_reshaped) + mean_reshaped
        
        # Triệt tiêu sai phân dự đoán dẫn tới vận tốc âm
        pred_real = np.maximum(pred_real, 0)
        
        # Format đầu ra yêu cầu là (12, 207) -> Transpose lại
        pred_real = np.transpose(pred_real, (1, 0))
        
        return pred_real
