import os
import sys
import torch
import pickle
import logging
import datetime
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

logger = logging.getLogger("predict_service")

class PredictService:
    """
    Cấu trúc Dịch vụ Cốt lõi (Core Inference Service).
    Khai triển mô hình Graph WaveNet cung ứng phương thức truy vấn dự báo cho Endpoint.
    """
    def __init__(self):
        # Mặc định cấu hình tương thích thiết bị CUDA tối đa hoá tính toán song song
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_nodes = 307
        self.in_dim = 5
        self.seq_len = 12
        
        # Liên kết tài nguyên đã qua huấn luyện cho PeMS04
        self.model_path = os.path.join(ai_core_path, 'checkpoints', 'model_pems04_best.pt')
        self.scaler_path = os.path.join(ai_core_path, 'data', 'processed', 'pems04', 'scaler.pkl')
        self.adj_path = os.path.join(ai_core_path, 'data', 'processed', 'pems04', 'adj_mx.pkl')
        
        self.model = None
        self.scaler = None
        self.adj_mx = None
        
        self._load_dependencies()

    def _load_dependencies(self):
        """
        Nạp cấu trúc đồ thị không gian và trích dẫn bộ trọng số Pre-Trained PeMS04.
        """
        if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
            logger.warning(f"Không tìm thấy tệp tin model/scaler tại {ai_core_path}")
            return
            
        try:
            # 1. Nạp Scaler
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # 2. Nạp Ma trận kề
            with open(self.adj_path, 'rb') as f:
                self.adj_mx = pickle.load(f)
            adj_tensor = torch.tensor(self.adj_mx, dtype=torch.float32).to(self.device)
            
            # 3. Khởi tạo và nạp Model
            self.model = GraphWaveNet(
                num_nodes=self.num_nodes, 
                in_dim=self.in_dim, 
                out_dim=self.seq_len, 
                supports=[adj_tensor],
                addaptadj=True
            ).to(self.device)
            
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            logger.info(f"Đã nạp thành công mô hình PeMS04 (307 nodes)")
        except Exception as e:
            logger.error(f"Lỗi khi nạp tài nguyên AI: {str(e)}")

    def predict(self, input_data: np.ndarray, timestamp: float = None) -> np.ndarray:
        """
        Dự báo lưu lượng giao thông cho 307 trạm trong 60 phút tới.
        Hỗ trợ đầu vào linh hoạt: (12, 307) hoặc (12, 307, 5).
        """
        if self.model is None:
            return None

        try:
            # 1. Tiền xử lý: Kiểm tra shape đầu vào
            if len(input_data.shape) == 3 and input_data.shape[2] == 5:
                # Nếu đã có đủ 5 features (từ tập Test gửi sang)
                full_raw = input_data.copy()
            else:
                # Nếu chỉ có 1 feature Flow (từ dữ liệu live gửi sang)
                full_raw = np.zeros((self.seq_len, self.num_nodes, 5))
                full_raw[..., 0] = input_data # Flow
                
                # Sinh đặc trưng thời gian
                dt = datetime.datetime.fromtimestamp(timestamp) if timestamp else datetime.datetime.now()
                tod = (dt.hour * 60 + dt.minute) // 5
                dow = dt.weekday()
                full_raw[..., 3] = tod / 288.0
                full_raw[..., 4] = dow / 7.0

            # 2. Chuẩn hóa Z-score cho 3 kênh vật lý đầu tiên
            mean = self.scaler['mean'].reshape(1, 1, 3)
            std = self.scaler['std'].reshape(1, 1, 3)
            
            full_input = full_raw.copy()
            full_input[..., :3] = (full_raw[..., :3] - mean) / std
            
            # 4. Convert sang Tensor và Permute về (1, 5, 307, 12)
            tensor_in = torch.FloatTensor(full_input).unsqueeze(0).permute(0, 3, 2, 1).to(self.device)

            # 5. Inference
            with torch.no_grad():
                output_tensor = self.model(tensor_in) # (1, 307, 12)
            
            # 6. Giải chuẩn hóa (Denormalization) cho Flow (index 0)
            target_mean = self.scaler['mean'][0]
            target_std = self.scaler['std'][0]
            
            pred_norm = output_tensor.squeeze(0).cpu().numpy() # (307, 12)
            pred_real = (pred_norm * target_std) + target_mean
            pred_real = np.maximum(pred_real, 0)
            
            return pred_real.T # Trả về (12, 307)

        except Exception as e:
            logger.error(f"Lỗi trong quá trình dự báo: {str(e)}")
            return None
