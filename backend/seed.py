"""
Module seed.py
--------------
Script khởi tạo dữ liệu mẫu (Seeder) cho hệ thống cơ sở dữ liệu.
Tạo giả lập 5 trạm giao thông ngẫu nhiên, kết nối chúng thành một đồ thị không gian (Graph) 
và nạp dữ liệu lưu lượng chuỗi thời gian (Time-series Traffic Volume) trong 24 giờ qua.
Giới hạn: Dữ liệu ảo chỉ dùng cho mục đích kiểm thử cấu trúc pipeline theo yêu cầu đặc biệt.
"""
import asyncio
import random
from datetime import datetime, timedelta
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from database import AsyncSessionLocal
from models import Station, RoadSegment, TrafficRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("db_seeder")

async def clear_existing_data(session: AsyncSession):
    """ Xóa sạch dữ liệu trong các bảng để tránh trùng lặp khi chạy seeder nhiều lần. """
    logger.info("Đang làm sạch dữ liệu hiện tồn trong CSDL...")
    # Vì đã thiết lập ondelete="CASCADE", ta chỉ cần xóa root (Station) là CSDL tự dọn các bảng con
    await session.execute(text("TRUNCATE TABLE stations RESTART IDENTITY CASCADE;"))
    await session.commit()
    logger.info("Đã xóa sạch dữ liệu cũ.")

async def seed_data():
    """ Hàm thực thi chính: Sinh trạm giao thông, tạo cạnh Graph và đổ dữ liệu 24h. """
    async with AsyncSessionLocal() as session:
        try:
            await clear_existing_data(session)

            logger.info("Bắt đầu khởi tạo 5 trạm giao thông ảo (Graph Nodes)...")
            stations_data = [
                Station(station_id_str="ST-001", name="Ngã tư Cầu Giấy", lat=21.0289, lng=105.7980),
                Station(station_id_str="ST-002", name="Ngã tư Kim Mã", lat=21.0315, lng=105.8155),
                Station(station_id_str="ST-003", name="Ngã tư Nguyễn Chí Thanh", lat=21.0185, lng=105.8075),
                Station(station_id_str="ST-004", name="Ngã tư Điện Biên Phủ", lat=21.0310, lng=105.8360),
                Station(station_id_str="ST-005", name="Ngã tư Chùa Bộc", lat=21.0070, lng=105.8275)
            ]
            
            session.add_all(stations_data)
            await session.commit()
            
            for st in stations_data:
                await session.refresh(st)

            logger.info("Bắt đầu tạo mạng lưới đoạn đường liên kết (Graph Edges)...")
            edges_data = [
                RoadSegment(source_station_id=stations_data[0].id, target_station_id=stations_data[1].id, distance=2.5),
                RoadSegment(source_station_id=stations_data[1].id, target_station_id=stations_data[0].id, distance=2.6),
                RoadSegment(source_station_id=stations_data[1].id, target_station_id=stations_data[2].id, distance=1.8),
                RoadSegment(source_station_id=stations_data[2].id, target_station_id=stations_data[1].id, distance=1.8),
                RoadSegment(source_station_id=stations_data[2].id, target_station_id=stations_data[4].id, distance=3.2),
                RoadSegment(source_station_id=stations_data[1].id, target_station_id=stations_data[3].id, distance=2.1),
                RoadSegment(source_station_id=stations_data[3].id, target_station_id=stations_data[1].id, distance=2.2),
            ]
            session.add_all(edges_data)
            
            logger.info("Sinh chuỗi dữ liệu giao thông (Time-series) ảo trong 24 giờ qua...")
            records_data = []
            now = datetime.now()
            
            # Đổ dữ liệu về 23 giờ trong quá khứ cộng giờ hiện tại (24 mẫu) cho từng trạm
            for st in stations_data:
                for i in range(24):
                    record_time = now - timedelta(hours=i)
                    vol = random.uniform(100, 1000)
                    speed = random.uniform(20, 60) if vol < 600 else random.uniform(10, 30)
                    
                    record = TrafficRecord(
                        station_id=st.id,
                        timestamp=record_time.replace(minute=0, second=0, microsecond=0),
                        volume=vol,
                        avg_speed=speed,
                        is_prediction=False
                    )
                    records_data.append(record)
                    
            session.add_all(records_data)
            await session.commit()

            logger.info("Đã hoàn tất nạp chuỗi dữ liệu Seed vào hệ thống PostgreSQL.")
        except Exception as e:
            logger.error(f"Xảy ra ngoại lệ trong tiến trình đẩy Database Seed: {str(e)}")
            await session.rollback()

if __name__ == "__main__":
    asyncio.run(seed_data())
