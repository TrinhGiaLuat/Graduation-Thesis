"""
Module traffic.py (Router)
--------------------------
Khai báo các API Endpoints phục vụ việc truy xuất dữ liệu không gian, đồ thị cấu trúc và chuỗi thời gian giao thông.
Đóng vai trò như Cổng giao tiếp dữ liệu (Data Access Layer) truyền tải giữa ứng dụng Frontend hoặc Model GNN.
"""
import logging
from typing import List, Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

# Sử dụng Absolute/Relative import map từ App module
from database import get_db
from models import Station, RoadSegment, TrafficRecord
from schemas import StationResponse, TrafficRecordResponse

logger = logging.getLogger("api_traffic")

router = APIRouter(prefix="/api", tags=["Traffic Data"])

@router.get("/stations", response_model=List[StationResponse])
async def get_all_stations(db: AsyncSession = Depends(get_db)):
    """
    Trích xuất danh sách trạm giao thông tĩnh (Graph Nodes) đang có trong bộ dữ liệu đồ thị nền tảng.
    """
    logger.info("Mở luồng cung cấp API lấy danh sách trạm (GET /api/stations).")
    try:
        result = await db.execute(select(Station))
        stations = result.scalars().all()
        return stations
    except Exception as e:
        logger.error(f"Xảy ra lỗi khi truy xuất danh sách trạm: {str(e)}")
        raise HTTPException(status_code=500, detail="Lỗi DB: Không thể tải danh sách trạm giao thông tại thời điểm này.")

@router.get("/graph", response_model=List[Any])
async def get_traffic_graph(db: AsyncSession = Depends(get_db)):
    """
    Phân rã cấu trúc các đoạn đường liên kết (Edges), đóng vai trò thiết lập ma trận kề (Adjacency Matrix)
    hoặc kết xuất dữ liệu cho map layer phía Frontend.
    """
    logger.info("Mở luồng cung cấp API cấu trúc liên kết mạng Graph (GET /api/graph).")
    try:
        result = await db.execute(select(RoadSegment))
        edges = result.scalars().all()
        
        # Serialize dưới dạng JSON thuần túy do không định nghĩa schema strict
        return [
            {
                "id": edge.id,
                "source_station_id": edge.source_station_id,
                "target_station_id": edge.target_station_id,
                "distance": edge.distance
            }
            for edge in edges
        ]
    except Exception as e:
        logger.error(f"Lỗi khi trích xuất kết cấu Graph Edges: {str(e)}")
        raise HTTPException(status_code=500, detail="Không thể kết xuất dữ liệu đồ thị mạng lưới đường bộ.")

@router.get("/traffic/{station_id}", response_model=List[TrafficRecordResponse])
async def get_station_traffic_history(station_id: int, db: AsyncSession = Depends(get_db)):
    """
    Kết xuất chuỗi series dữ liệu giao thông (Time-series) của một Nút (Station) định kỳ trong lịch sử 24 giờ.
    
    Args:
        station_id (int): Khoá chính định danh thực thể Station.
    """
    logger.info(f"Yêu cầu trích xuất dữ liệu mảng Time-series trạm {station_id} được xử lý.")
    try:
        # Lấy tối đa 24 bản ghi của trạm, sort theo thời điểm tăng dần để vẽ biếu đồ
        stmt = select(TrafficRecord).where(
            TrafficRecord.station_id == station_id
        ).order_by(TrafficRecord.timestamp.asc()).limit(24)
        
        result = await db.execute(stmt)
        records = result.scalars().all()
        
        if not records:
            # Ngăn lỗi fail silently, raise mã 404
            raise HTTPException(status_code=404, detail=f"Cơ sở dữ liệu không ghi nhận dữ liệu cho trạm {station_id}.")
            
        return records
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Phát hiện lỗi không gian bộ nhớ CSDL khi trích lịch sử trạm {station_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Gặp lỗi hệ thống nội bộ khi tải luồng dữ liệu 24H qua.")
