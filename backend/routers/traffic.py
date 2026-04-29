"""
Module traffic.py (Router)
--------------------------
Khai báo các API Endpoints phục vụ truy xuất dữ liệu không gian, đồ thị giao thông và dự báo time-series.
Cơ chế Lazy-Load an toàn: Nếu Model AI chưa khởi động, fallback về dữ liệu giả lập thay vì crash Server.
"""
import os
import sys
import logging
import random
from datetime import datetime, timedelta
from typing import List, Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from database import get_db
from models import Station, RoadSegment, TrafficRecord
from schemas import StationResponse, TrafficRecordResponse

logger = logging.getLogger("api_traffic")
router = APIRouter(prefix="/api", tags=["Traffic Data"])

# ============================================================
# Lazy-Load AI Module (Import tại runtime để Backend luôn khởi động được)
# ============================================================
_ai_cache = {}  # Singleton cache tránh load lại mỗi request

def _try_load_ai():
    """
    Cơ chế nạp trì hoãn (Lazy-Load) Mô hình GNN PeMS04. 
    Không gọi tại startup giúp Server luôn khởi động thành công.
    Trả về (service, sensor_map, test_x) hoặc (None, None, None) nếu thất bại.
    """
    if "loaded" in _ai_cache:
        return _ai_cache.get("svc"), _ai_cache.get("map"), _ai_cache.get("data")
    
    _ai_cache["loaded"] = True  # Đánh dấu đã thử để không retry vô hạn
    try:
        import numpy as np
        import pickle

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ai_core_path = os.path.join(base_dir, 'ai_core')
        sys.path.insert(0, ai_core_path)

        from app.services.predict_service import PredictService

        # Trỏ vào file PeMS04
        adj_path = os.path.join(ai_core_path, 'data', 'processed', 'pems04', 'adj_mx.pkl')
        npz_path = os.path.join(ai_core_path, 'data', 'processed', 'pems04', 'test.npz')

        if not os.path.exists(adj_path) or not os.path.exists(npz_path):
            logger.warning("Không tìm thấy file dữ liệu AI PeMS04. Chạy chế độ Fallback.")
            return None, None, None

        # Tự động map ID từ 0 đến 306 vì adj_mx của PeMS04 là ma trận 307x307
        sensor_id_to_ind = {str(i): i for i in range(307)}

        test_data = np.load(npz_path)
        svc = PredictService()

        _ai_cache["svc"] = svc
        _ai_cache["map"] = sensor_id_to_ind
        _ai_cache["data"] = test_data['x']
        logger.info("Nạp Mô hình GNN PeMS04 thành công.")
        return svc, sensor_id_to_ind, test_data['x']

    except Exception as e:
        logger.warning(f"Không thể nạp Mô hình GNN: {e}. Fallback về dữ liệu giả lập.")
        return None, None, None


def _generate_fallback_records(station_id: int) -> List[TrafficRecordResponse]:
    """Sinh dữ liệu giả lập hình sin để Frontend vẽ được biểu đồ khi Model chưa sẵn sàng."""
    import math
    start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    records = []
    for i in range(24):
        # Mô phỏng lưu lượng hình sin (buổi sáng và chiều tối đông hơn)
        base = 400 + 300 * math.sin((i - 7) * math.pi / 12)
        volume = round(max(50.0, base + random.uniform(-50, 50)), 2)
        records.append(TrafficRecordResponse(
            id=i + 1,
            station_id=station_id,
            timestamp=start_time + timedelta(hours=i),
            volume=volume,
            avg_speed=round(volume / 10, 2),
            is_prediction=bool(i >= 12)
        ))
    return records


@router.get("/stations", response_model=List[StationResponse])
async def get_all_stations(db: AsyncSession = Depends(get_db)):
    """Trích xuất danh sách trạm giao thông (Graph Nodes)."""
    logger.info("GET /api/stations")
    try:
        result = await db.execute(select(Station))
        return result.scalars().all()
    except Exception as e:
        logger.error(f"Lỗi SQL: {str(e)}")
        raise HTTPException(status_code=500, detail="Không thể tải danh sách trạm.")


@router.get("/graph", response_model=List[Any])
async def get_traffic_graph(db: AsyncSession = Depends(get_db)):
    """Phân rã cấu trúc Edges của Đồ thị Giao thông."""
    logger.info("GET /api/graph")
    try:
        result = await db.execute(select(RoadSegment))
        edges = result.scalars().all()
        return [
            {"id": e.id, "source_station_id": e.source_station_id,
             "target_station_id": e.target_station_id, "distance": e.distance}
            for e in edges
        ]
    except Exception as e:
        logger.error(f"Lỗi Graph: {str(e)}")
        raise HTTPException(status_code=500, detail="Không thể tải dữ liệu đồ thị.")


@router.get("/traffic/snapshot")
async def get_traffic_snapshot(horizon: int = 12, timestep: int = None, db: AsyncSession = Depends(get_db)):
    """
    Endpoint Batch Inference: Dự báo lưu lượng giao thông cho toàn bộ 307 trạm
    trong MỘT lần forward pass duy nhất qua mô hình Graph WaveNet.

    QUAN TRỌNG: Route này PHẢI được khai báo TRƯỚC /traffic/{station_id}
    để FastAPI không nhầm "snapshot" là một station_id kiểu integer.

    Args:
        horizon (int): Mốc thời gian dự báo.
            - 3  → 15 phút | 6  → 30 phút | 12 → 60 phút
    """
    logger.info(f"GET /api/traffic/snapshot?horizon={horizon}")

    if not (1 <= horizon <= 12):
        raise HTTPException(
            status_code=400,
            detail=f"Tham số 'horizon' không hợp lệ: {horizon}. Phải nằm trong khoảng [1, 12]."
        )

    svc, sensor_map, data_x = _try_load_ai()
    if svc is None or data_x is None:
        raise HTTPException(status_code=503, detail="Model AI chưa sẵn sàng. Vui lòng thử lại sau.")

    try:
        import numpy as np

        # BƯỚC 1: Lấy mẫu từ tập Test dựa trên timestep để giả lập thời gian thực
        # Nếu không truyền timestep, dùng mẫu cuối cùng để đảm bảo tính nhất quán
        if timestep is not None:
            sample_idx = timestep % data_x.shape[0] # Lặp lại nếu vượt quá số lượng mẫu
        else:
            sample_idx = data_x.shape[0] - 1  # Mẫu cuối (cố định mặc định)
            
        input_seq  = data_x[sample_idx]

        # Giải chuẩn hóa 3 kênh vật lý để predict() pipeline hoạt động đúng
        mean = svc.scaler['mean'].reshape(1, 1, 3)
        std  = svc.scaler['std'].reshape(1, 1, 3)
        past_real = input_seq.copy()
        past_real[:, :, :3] = (input_seq[:, :, :3] * std) + mean

        # BƯỚC 2: Batch Inference — 1 forward pass → output (12, 307)
        pred_all = svc.predict(past_real)
        if pred_all is None:
            raise HTTPException(status_code=503, detail="Inference thất bại.")

        # BƯỚC 3: Trích xuất đúng horizon yêu cầu → (307,)
        pred_at_horizon = pred_all[horizon - 1, :]

        # BƯỚC 4: Đóng gói kết quả
        result = []
        for node_idx in range(307):
            flow_value = float(np.maximum(pred_at_horizon[node_idx], 0.0))
            result.append({
                "station_id":     node_idx + 1,
                "station_id_str": str(node_idx),
                "predicted_flow": round(flow_value, 2)
            })

        logger.info(f"Snapshot horizon={horizon} hoàn tất: {len(result)} trạm.")
        
        # -------------------------------------------------------------
        # BƯỚC 5: Ghi Log Lịch Sử Giả Lập ra file CSV (Giai đoạn 4)
        # -------------------------------------------------------------
        if timestep is not None:
            import os
            import csv
            from datetime import datetime
            
            log_dir = "/app/logs"
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "simulation_logs.csv")
            
            # Tính toán giờ ảo (giống Frontend: 08:00 AM + (step * 5 phút))
            start_minutes = 8 * 60
            total_minutes = start_minutes + (timestep * 5)
            h = (total_minutes // 60) % 24
            m = total_minutes % 60
            virtual_time = f"{h:02d}:{m:02d}"
            
            file_exists = os.path.isfile(log_file)
            
            # Mở file chế độ append
            with open(log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Timestamp_Thực", "Giờ_Giả_Lập", "Step", "Horizon", "Station_ID", "Dự_Báo_Lưu_Lượng"])
                
                # Để tránh file phình to quá nhanh, ta có thể chỉ ghi Top 10 trạm nóng nhất
                # Nhưng nhà quản lý cần Data toàn mạng, nên ghi đủ 307 trạm.
                rows = [
                    [
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                        virtual_time, 
                        timestep, 
                        horizon, 
                        r["station_id"], 
                        r["predicted_flow"]
                    ]
                    for r in result
                ]
                writer.writerows(rows)

            # BƯỚC 6: Ghi Log vào Database (PostgreSQL)
            from models import PredictionLog
            db_logs = [
                PredictionLog(
                    station_id=r["station_id"],
                    virtual_time=virtual_time,
                    timestep=timestep,
                    horizon=horizon,
                    predicted_flow=r["predicted_flow"],
                    created_at=datetime.now()
                )
                for r in result
            ]
            db.add_all(db_logs)
            await db.commit()

        return result
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"Lỗi Batch Inference Snapshot: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi Batch Inference: {str(e)}")


@router.get("/traffic/{station_id}", response_model=List[TrafficRecordResponse])
async def get_station_traffic(station_id: int, timestep: int = None, db: AsyncSession = Depends(get_db)):
    """
    Endpoint Dự báo GNN: Trả về 24 điểm dữ liệu (12 lịch sử + 12 tương lai).
    Fallback về dữ liệu giả lập nếu Model AI chưa sẵn sàng.
    """
    logger.info(f"GET /api/traffic/{station_id}")
    try:
        station = await db.get(Station, station_id)
        if not station:
            raise HTTPException(status_code=404, detail="Không tìm thấy trạm.")

        # Thử nạp AI, nếu fail thì dùng fallback
        svc, sensor_map, data_x = _try_load_ai()

        if svc is None or sensor_map is None or data_x is None:
            logger.info(f"Sử dụng Fallback data cho trạm {station_id}")
            return _generate_fallback_records(station_id)

        import numpy as np
        sensor_id_str = station.station_id_str
        if sensor_id_str not in sensor_map:
            return _generate_fallback_records(station_id)

        target_idx = sensor_map[sensor_id_str]
        # Dùng timestep để đồng bộ với giả lập thời gian thực của toàn thành phố
        if timestep is not None:
            sample_idx = timestep % data_x.shape[0]
        else:
            sample_idx = data_x.shape[0] - 1  # Mẫu cuối
            
        input_seq = data_x[sample_idx]  # (12, 307, 5) - Đây là dữ liệu đã chuẩn hoá (Normalized)

        # Khôi phục về đơn vị đời thực
        # PeMS04 có 5 kênh, nhưng scaler chỉ có cho 3 kênh vật lý đầu tiên
        past_real = input_seq.copy()
        
        mean = svc.scaler['mean']
        std = svc.scaler['std']
        
        # Reshape mean/std để broadcast với (12, 307, 3)
        m = mean.reshape(1, 1, 3)
        s = std.reshape(1, 1, 3)
        
        # Chỉ giải chuẩn hóa cho 3 kênh đầu (Flow, Occ, Speed)
        past_real[:, :, :3] = (input_seq[:, :, :3] * s) + m
        
        # Dự báo từ AI (Service này đã nhận diện 307 trạm và 5 features)
        pred_real = svc.predict(past_real)
        
        if pred_real is None:
            return _generate_fallback_records(station_id)

        # Trích xuất dữ liệu của trạm mục tiêu (Lưu lượng - Kênh index 0)
        past_values = past_real[:, target_idx, 0]
        future_values = pred_real[:, target_idx] # PredictService trả về (12, 307) đã là Flow

        start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        records = []
        for i, val in enumerate(list(past_values) + list(future_values)):
            records.append(TrafficRecordResponse(
                id=i + 1, station_id=station_id,
                timestamp=start_time + timedelta(hours=i),
                volume=round(max(0.0, float(val)), 2),
                avg_speed=round(max(0.0, float(val)), 2),
                is_prediction=bool(i >= 12)
            ))
        return records

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"Lỗi Inference trạm {station_id}: {str(e)}")
        for line in traceback.format_exc().split('\\n'):
            logger.error(line)
        # Fallback thay vì crash
        return _generate_fallback_records(station_id)


@router.get("/traffic/snapshot")
async def get_traffic_snapshot(horizon: int = 12):
    """
    Endpoint Batch Inference: Dự báo lưu lượng giao thông cho toàn bộ 307 trạm
    trong MỘT lần forward pass duy nhất qua mô hình Graph WaveNet.

    Nguyên lý hoạt động (Batch Inference):
        - Thay vì gọi model 307 lần (một lần cho mỗi trạm), endpoint này đưa
          toàn bộ tensor đầu vào của 307 node vào model CÙNG LÚC.
        - Model trả về output shape (1, 307, 12) — tức là 12 horizon cho 307 trạm
          chỉ trong 1 lần tính toán duy nhất. Đây là ưu thế then chốt của GNN.
        - Endpoint sau đó trích xuất giá trị tại horizon được yêu cầu và
          giải chuẩn hóa về đơn vị xe thực tế.

    Args:
        horizon (int): Mốc thời gian dự báo.
            - 3  → 15 phút (timestep thứ 3, tức 3 × 5 phút)
            - 6  → 30 phút (timestep thứ 6)
            - 12 → 60 phút (timestep thứ 12)

    Returns:
        List[dict]: Mảng 307 phần tử, mỗi phần tử gồm:
            - station_id (int): ID nội bộ của trạm trong DB.
            - station_id_str (str): Index node (0–306).
            - predicted_flow (float): Lưu lượng dự báo (số xe/5 phút), đã giải chuẩn hóa.

    Raises:
        HTTPException 400: Nếu horizon không hợp lệ (ngoài khoảng 1–12).
        HTTPException 503: Nếu Model AI chưa sẵn sàng.
    """
    logger.info(f"GET /api/traffic/snapshot?horizon={horizon}")

    # Validate tham số horizon
    if not (1 <= horizon <= 12):
        raise HTTPException(
            status_code=400,
            detail=f"Tham số 'horizon' không hợp lệ: {horizon}. Phải nằm trong khoảng [1, 12]."
        )

    # Thử nạp AI Service (sử dụng lại Singleton Cache từ _try_load_ai())
    svc, sensor_map, data_x = _try_load_ai()

    if svc is None or data_x is None:
        raise HTTPException(
            status_code=503,
            detail="Model AI chưa sẵn sàng. Vui lòng thử lại sau."
        )

    try:
        import numpy as np

        # =====================================================================
        # BƯỚC 1: CHUẨN BỊ TENSOR ĐẦU VÀO (BATCH INPUT PREPARATION)
        # Lấy ngẫu nhiên 1 mẫu dữ liệu từ tập Test làm input 12 bước quá khứ.
        # Shape: (12, 307, 5) — 12 timestep, 307 nodes, 5 features
        # =====================================================================
        sample_idx = random.randint(0, data_x.shape[0] - 1)
        input_seq = data_x[sample_idx]  # (12, 307, 5) - đang ở dạng đã chuẩn hóa

        # Giải chuẩn hóa 3 kênh vật lý (Flow, Occ, Speed) về đơn vị thực
        # để predict() có thể chuẩn hóa lại theo đúng pipeline đã train
        mean = svc.scaler['mean'].reshape(1, 1, 3)
        std  = svc.scaler['std'].reshape(1, 1, 3)
        past_real = input_seq.copy()
        past_real[:, :, :3] = (input_seq[:, :, :3] * std) + mean

        # =====================================================================
        # BƯỚC 2: BATCH INFERENCE — 1 LẦN FORWARD PASS CHO 307 TRẠM
        # predict() nội bộ sẽ:
        #   - Chuẩn hóa lại đầu vào (Z-score)
        #   - Permute tensor về (1, 5, 307, 12) theo yêu cầu của GraphWaveNet
        #   - Chạy model.forward() → output (1, 307, 12)
        #   - Giải chuẩn hóa kênh Flow về đơn vị xe thực tế
        #   - Trả về numpy array shape (12, 307)
        # =====================================================================
        pred_all = svc.predict(past_real)  # (12, 307) — tất cả 307 trạm, 12 horizons

        if pred_all is None:
            raise HTTPException(status_code=503, detail="Inference thất bại.")

        # =====================================================================
        # BƯỚC 3: TRÍCH XUẤT MỐC THỜI GIAN YÊU CẦU (HORIZON EXTRACTION)
        # horizon=3 → index 2 (15 phút), horizon=6 → index 5 (30 phút), v.v.
        # pred_at_horizon shape: (307,) — giá trị flow của 307 trạm tại mốc t
        # =====================================================================
        pred_at_horizon = pred_all[horizon - 1, :]  # (307,)

        # =====================================================================
        # BƯỚC 4: ĐÓNG GÓI KẾT QUẢ (RESPONSE SERIALIZATION)
        # station_id_str = str(i) tương ứng với DB ID = i + 1 (do RESTART IDENTITY)
        # =====================================================================
        result = []
        for node_idx in range(307):
            flow_value = float(np.maximum(pred_at_horizon[node_idx], 0.0))
            result.append({
                "station_id":     node_idx + 1,      # DB Primary Key (1-indexed)
                "station_id_str": str(node_idx),      # Node index (0-indexed)
                "predicted_flow": round(flow_value, 2)
            })

        logger.info(f"Snapshot horizon={horizon} hoàn tất: {len(result)} trạm.")
        return result

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"Lỗi Batch Inference Snapshot: {str(e)}")
        for line in traceback.format_exc().split('\\n'):
            logger.error(line)
        raise HTTPException(status_code=500, detail=f"Lỗi nội bộ khi chạy Batch Inference: {str(e)}")
