import os
import sys
import pickle
import asyncio
import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Kế thừa liên kết môi trường tuyệt đối đến Root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from database import AsyncSessionLocal
from models import Station, RoadSegment

async def clear_existing_data(session: AsyncSession):
    """
    Kích hoạt tiến trình làm sạch vùng CSDL theo chuẩn TRUNCATE nguyên khối.
    """
    await session.execute(text("TRUNCATE TABLE stations RESTART IDENTITY CASCADE;"))
    await session.commit()

async def seed_metr_la():
    """
    Tiến trình đồng bộ hoá tập dữ liệu thực tế METR-LA vào hệ thống PostgreSQL.
    Ánh xạ 207 Nút giao thông (Stations) và thiết lập cấu trúc cạnh đồ thị (Road Segments).
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Kéo bộ tọa độ CSV gốc để sinh Nút đo lường
    csv_path = os.path.join(base_dir, 'ai_core', 'data', 'raw', 'metr-la', 'graph_sensor_locations.csv')
    # Kéo Ma trận nguyên gốc tích chập để ánh xạ Indexing Id
    raw_pkl_path = os.path.join(base_dir, 'ai_core', 'data', 'raw', 'metr-la', 'adj_METR-LA.pkl')

    if not os.path.exists(csv_path) or not os.path.exists(raw_pkl_path):
        print("Lỗi I/O: Không tìm nhận được vị trí tệp dữ liệu phân vùng METR-LA.")
        return

    async with AsyncSessionLocal() as session:
        try:
            await clear_existing_data(session)

            # 1. Phân rã cấu trúc tệp toạ độ trạm thu (CSV Dataset)
            print("Đang kiến tạo bộ Cấu hình Tọa độ cho 207 Trạm cảm biến...")
            df_locations = pd.read_csv(csv_path, names=['sensor_id', 'lat', 'lng'])
            
            stations_dict = {}
            for index, row in df_locations.iterrows():
                # Bỏ qua tệp tin chứa Header
                if str(row['sensor_id']).lower().strip() == 'sensor_id':
                    continue
                    
                st = Station(
                    station_id_str=str(row['sensor_id']).strip(),
                    name=f"LA Sensor {row['sensor_id']}",
                    lat=float(row['lat']),
                    lng=float(row['lng'])
                )
                session.add(st)
                # Cache Index nhằm cung cấp ForeignKey cấp tốc cho bảng Edges
                stations_dict[str(row['sensor_id']).strip()] = st
                
            await session.commit()
            
            for st in stations_dict.values():
                await session.refresh(st)
                
            # 2. Xử lý đồ thị hướng Giao thông (Adjacency Matrix Translation)
            print("Tiến hành cấu trúc siêu liên hợp Đồ thị Adjacency Matrix...")
            with open(raw_pkl_path, 'rb') as f:
                # Đảm bảo duy trì byte formatting cho tệp Pickle phiên bản cũ
                sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding='latin1')
            
            edges_to_insert = []
            num_nodes = adj_mx.shape[0]
            
            # Loại bỏ vòng xoay trạm tụ (self-loop) và lọc trọng điểm trên .5
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j and adj_mx[i, j] > 0.5:
                        src_str = str(sensor_ids[i])
                        tgt_str = str(sensor_ids[j])
                        
                        if src_str in stations_dict and tgt_str in stations_dict:
                            edge = RoadSegment(
                                source_station_id=stations_dict[src_str].id,
                                target_station_id=stations_dict[tgt_str].id,
                                distance=float(adj_mx[i, j])
                            )
                            edges_to_insert.append(edge)
                            
            session.add_all(edges_to_insert)
            await session.commit()

            print(f"Báo cáo tiến trình: Tổng nạp {len(stations_dict)} Node Trạm đo và {len(edges_to_insert)} Cạnh giao thông.")
        
        except Exception as e:
            print(f"Dừng khẩn cấp tiến trình Seeding: {str(e)}")
            await session.rollback()

if __name__ == "__main__":
    asyncio.run(seed_metr_la())
