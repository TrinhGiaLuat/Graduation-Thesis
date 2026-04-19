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
    Cơ chế nạp trì hoãn (Lazy-Load) Mô hình GNN. 
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

        pkl_path = os.path.join(ai_core_path, 'data', 'raw', 'metr-la', 'adj_METR-LA.pkl')
        npz_path = os.path.join(ai_core_path, 'data', 'processed', 'test.npz')

        if not os.path.exists(pkl_path) or not os.path.exists(npz_path):
            logger.warning("Không tìm thấy file dữ liệu AI. Chạy chế độ Fallback.")
            return None, None, None

        with open(pkl_path, 'rb') as f:
            _, sensor_id_to_ind, _ = pickle.load(f, encoding='latin1')

        test_data = np.load(npz_path)
        svc = PredictService()

        _ai_cache["svc"] = svc
        _ai_cache["map"] = sensor_id_to_ind
        _ai_cache["data"] = test_data['x']
        logger.info("Nạp Mô hình GNN thành công.")
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


@router.get("/traffic/{station_id}", response_model=List[TrafficRecordResponse])
async def get_station_traffic(station_id: int, db: AsyncSession = Depends(get_db)):
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
        sample_idx = random.randint(0, data_x.shape[0] - 1)
        input_seq = data_x[sample_idx]  # (12, 207) - Đây là dữ liệu đã chuẩn hoá (Normalized)

        mean = svc.scaler['mean']
        std = svc.scaler['std']
        
        # Khôi phục về vận tốc đời thực (mph) NGAY LẬP TỨC
        past_real = (input_seq * std) + mean
        
        # CHÚ Ý: Phải truyền past_real (Raw Data) vào service vì bên trong service đã tự động có code Normalization rồi!
        # Nếu truyền input_seq thì sẽ bị chuẩn hoá 2 lần dẫn tới các con số tiệm cận 0 (3-5 mph)
        pred_real = svc.predict(past_real)
        past_values = past_real[:, target_idx]
        future_values = pred_real[:, target_idx]

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
