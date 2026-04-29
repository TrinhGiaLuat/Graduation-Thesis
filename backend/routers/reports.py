from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from database import get_db
from models import PredictionLog, Station
import os

router = APIRouter(tags=["Reports"])

@router.get("/reports/summary")
async def get_report_summary(db: AsyncSession = Depends(get_db)):
    """
    Trích xuất báo cáo tổng quan từ bảng prediction_logs.
    """
    try:
        # 1. Tổng số bản ghi log (Số điểm dữ liệu đã dự báo)
        result_total = await db.execute(select(func.count(PredictionLog.id)))
        total_predictions = result_total.scalar() or 0

        # 2. Tổng lưu lượng tích lũy của toàn mạng lưới
        result_flow = await db.execute(select(func.sum(PredictionLog.predicted_flow)))
        total_flow = result_flow.scalar() or 0

        # 3. Top 5 trạm thường xuyên ùn tắc nhất (dựa trên trung bình lưu lượng)
        # Sử dụng func.avg để tính lưu lượng trung bình mỗi trạm
        stmt_top = (
            select(PredictionLog.station_id, Station.name, func.avg(PredictionLog.predicted_flow).label("avg_flow"))
            .join(Station, PredictionLog.station_id == Station.id)
            .group_by(PredictionLog.station_id, Station.name)
            .order_by(desc("avg_flow"))
            .limit(5)
        )
        result_top = await db.execute(stmt_top)
        top_stations = [
            {"station_id": row.station_id, "name": row.name, "avg_flow": round(row.avg_flow, 2)}
            for row in result_top.fetchall()
        ]

        # 4. Trích xuất xu hướng lưu lượng theo thời gian (giới hạn 50 mốc thời gian gần nhất)
        stmt_trend = (
            select(PredictionLog.virtual_time, func.sum(PredictionLog.predicted_flow).label("total_flow"))
            .group_by(PredictionLog.virtual_time)
            .order_by(PredictionLog.virtual_time)
        )
        result_trend = await db.execute(stmt_trend)
        # Lấy 50 mốc cuối cùng để vẽ biểu đồ
        trend_data = [
            {"time": row.virtual_time, "flow": round(row.total_flow, 2)}
            for row in result_trend.fetchall()
        ][-50:]

        return {
            "total_predictions": total_predictions,
            "total_flow": round(total_flow, 2),
            "top_stations": top_stations,
            "trend_data": trend_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reports/history")
async def get_report_history(db: AsyncSession = Depends(get_db)):
    """
    Lấy danh sách các bản ghi cảnh báo (Lưu lượng > 500) mới nhất
    hoặc 50 bản ghi lưu lượng cao nhất để nhà quản lý xem lại.
    """
    try:
        stmt = (
            select(
                PredictionLog.virtual_time,
                PredictionLog.horizon,
                PredictionLog.predicted_flow,
                Station.name.label("station_name")
            )
            .join(Station, PredictionLog.station_id == Station.id)
            .where(PredictionLog.predicted_flow > 400) # Chỉ lấy các trạm có nguy cơ cao
            .order_by(desc(PredictionLog.virtual_time), desc(PredictionLog.predicted_flow))
            .limit(50)
        )
        result = await db.execute(stmt)
        
        history = []
        for row in result.fetchall():
            history.append({
                "time": row.virtual_time,
                "horizon": row.horizon,
                "station": row.station_name,
                "flow": round(row.predicted_flow, 2),
                "is_critical": row.predicted_flow > 500
            })
            
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/reports/reset")
async def reset_report_data(db: AsyncSession = Depends(get_db)):
    """
    Xóa sạch dữ liệu lịch sử trong DB và file CSV.
    """
    try:
        # 1. Xóa trong Database
        from sqlalchemy import delete
        await db.execute(delete(PredictionLog))
        await db.commit()

        # 2. Xóa file CSV
        log_file = "/app/logs/simulation_logs.csv"
        if os.path.exists(log_file):
            os.remove(log_file)
            
        return {"message": "Đã xóa sạch dữ liệu lịch sử thành công."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
