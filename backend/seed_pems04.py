import os
import sys
import pickle
import asyncio
import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Kế thừa liên kết môi trường tuyệt đối đến Root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from database import AsyncSessionLocal
from models import Station, RoadSegment

async def clear_existing_data(session: AsyncSession):
    """Làm sạch CSDL cũ"""
    await session.execute(text("TRUNCATE TABLE stations RESTART IDENTITY CASCADE;"))
    await session.commit()

async def seed_pems04():
    """
    Đồng bộ hóa 307 trạm PeMS04 vào PostgreSQL.
    Sử dụng thuật toán Force-Directed Layout (Fruchterman-Reingold đơn giản hóa)
    để sắp xếp vị trí các trạm dựa trên cấu trúc ma trận kề, sau đó ánh xạ
    vào tọa độ khu vực San Francisco.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    adj_path = os.path.join(base_dir, 'ai_core', 'data', 'processed', 'pems04', 'adj_mx.pkl')

    if not os.path.exists(adj_path):
        print(f"[!] Lỗi: Không tìm thấy file ma trận kề tại {adj_path}")
        return

    async with AsyncSessionLocal() as session:
        try:
            await clear_existing_data(session)
            print("[*] Đã dọn dẹp dữ liệu cũ. Bắt đầu nạp 307 trạm PeMS04...")

            # 1. Thuật toán sắp xếp vị trí dựa trên cấu trúc đồ thị (Spring Layout)
            print("[*] Đang tính toán vị trí trạm dựa trên cấu trúc Graph (Force-directed layout)...")
            with open(adj_path, 'rb') as f:
                adj_mx = pickle.load(f)
            
            # Sử dụng thuật toán Fruchterman-Reingold đơn giản hóa
            # Khởi tạo vị trí ngẫu nhiên ban đầu
            np.random.seed(42)
            pos = np.random.rand(307, 2)
            iterations = 50
            k = np.sqrt(1.0 / 307) # Lực đẩy lý tưởng
            
            # Chạy mô phỏng lực để các node tự sắp xếp theo ma trận kề
            for _ in range(iterations):
                # Tính lực đẩy giữa tất cả các cặp node (để không bị dính chùm)
                delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
                dist = np.linalg.norm(delta, axis=-1) + 1e-9
                repulsion = (k**2 / dist)[..., np.newaxis] * delta
                pos += np.sum(repulsion, axis=1) * 0.01
                
                # Tính lực hút giữa các node có liên kết (dựa trên adj_mx)
                attraction = (adj_mx[..., np.newaxis] * delta * dist[..., np.newaxis] / k)
                pos -= np.sum(attraction, axis=1) * 0.01
                
                # Giữ trong khung [0, 1]
                pos = (pos - pos.min(axis=0)) / (pos.max(axis=0) - pos.min(axis=0))

            # Ánh xạ khung xương này vào tọa độ San Francisco
            # SF Bounding Box: Lat(37.70, 37.82), Lng(-122.50, -122.38)
            lat_min, lat_max = 37.70, 37.82
            lng_min, lng_max = -122.50, -122.38
            
            stations_dict = {}
            for i in range(307):
                lat = lat_min + pos[i, 1] * (lat_max - lat_min)
                lng = lng_min + pos[i, 0] * (lng_max - lng_min)
                
                st = Station(
                    station_id_str=str(i),
                    name=f"SF Sensor {i}",
                    lat=float(lat),
                    lng=float(lng)
                )
                session.add(st)
                stations_dict[i] = st
            
            await session.commit()
            print("[*] Đã kiến tạo khung xương mạng lưới 307 trạm tại San Francisco.")
            for i in range(307):
                await session.refresh(stations_dict[i])

            # 2. Xử lý các cạnh (Road Segments)
            print("[*] Đang thiết lập các liên kết giao thông (Graph Edges)...")
            
            edges_to_insert = []
            num_nodes = adj_mx.shape[0]
            
            # Chỉ lấy các liên kết có trọng số kề > 0.1 (để bản đồ đỡ bị rối)
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j and adj_mx[i, j] > 0.1:
                        edge = RoadSegment(
                            source_station_id=stations_dict[i].id,
                            target_station_id=stations_dict[j].id,
                            distance=float(adj_mx[i, j])
                        )
                        edges_to_insert.append(edge)
            
            session.add_all(edges_to_insert)
            await session.commit()

            print(f"[v] HOÀN TẤT: Đã nạp {len(stations_dict)} Trạm SF và {len(edges_to_insert)} Liên kết đồ thị.")
        
        except Exception as e:
            print(f"[!] Lỗi Seeding: {str(e)}")
            await session.rollback()

if __name__ == "__main__":
    asyncio.run(seed_pems04())
