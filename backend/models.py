"""
Module models.py
----------------
Định nghĩa các bảng cơ sở dữ liệu (ORM Models) phục vụ mô hình 24H-GNN dự báo lưu lượng giao thông.
Mọi thực thể đều kết nối tới Base map metadata của cơ chế SQLAlchemy (2.0 standard).
"""
from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Boolean
from sqlalchemy.orm import relationship

# Thu nạp class Base tĩnh từ luồng điều khiển CSDL
from database import Base

class Station(Base):
    """
    Thực thể mô tả Trạm đo giao thông / Nút giao trên đồ thị không gian (Spatial Graph).
    """
    __tablename__ = "stations"

    id = Column(Integer, primary_key=True, index=True)
    station_id_str = Column(String, unique=True, index=True, nullable=False, doc="Mã định danh thực tế của trạm thu thập")
    name = Column(String, nullable=True, doc="Tên gọi tham chiếu nếu có")
    lat = Column(Float, nullable=False, doc="Vĩ tuyến")
    lng = Column(Float, nullable=False, doc="Kinh tuyến")

    # Bản đồ chéo tới danh sách đoạn đường (Graph Edges) cũng như lưu lượng đo được
    outgoing_edges = relationship("RoadSegment", foreign_keys="[RoadSegment.source_station_id]", back_populates="source_station")
    incoming_edges = relationship("RoadSegment", foreign_keys="[RoadSegment.target_station_id]", back_populates="target_station")
    records = relationship("TrafficRecord", back_populates="station", cascade="all, delete-orphan")

class RoadSegment(Base):
    """
    Thực thể biểu diễn Đoạn đường hoặc Cạnh trọng số trong Đồ thị giao thông,
    liên kết hai trạm (Station) theo chiều nguồn -> đích.
    """
    __tablename__ = "road_segments"

    id = Column(Integer, primary_key=True, index=True)
    # Ràng buộc Foreign Key với thuật toán xoá thác (Cascading delete)
    source_station_id = Column(Integer, ForeignKey("stations.id", ondelete="CASCADE"), nullable=False, doc="ID của trạm xuất phát")
    target_station_id = Column(Integer, ForeignKey("stations.id", ondelete="CASCADE"), nullable=False, doc="ID của trạm ngõ tới")
    distance = Column(Float, nullable=False, doc="Hệ số ma sát / Khoảng cách cự ly giữa hai trạm")

    # Mối tương quan 2 chiều (Bidirectional mapping) 
    source_station = relationship("Station", foreign_keys=[source_station_id], back_populates="outgoing_edges")
    target_station = relationship("Station", foreign_keys=[target_station_id], back_populates="incoming_edges")

class TrafficRecord(Base):
    """
    Cấu trúc bản ghi lưu lượng và vận tốc xe cộ ở các thời điểm cụ thể phục vụ mô hình huấn luyện/đo lường dự báo.
    """
    __tablename__ = "traffic_records"

    id = Column(Integer, primary_key=True, index=True)
    station_id = Column(Integer, ForeignKey("stations.id", ondelete="CASCADE"), nullable=False)
    timestamp = Column(DateTime, index=True, nullable=False, doc="Mốc thời gian lấy mẫu dữ liệu")
    volume = Column(Float, nullable=False, doc="Lưu thông thể tích xe cụ thể")
    avg_speed = Column(Float, nullable=True, doc="Vận tốc trung bình của tập phương tiện (tuỳ chọn)")
    is_prediction = Column(Boolean, default=False, nullable=False, doc="Phân mảnh cơ sở Ground-truth vs Dự đoán từ Model AI")

    # Tham chiếu gián tiếp đến Metadata cửa Station
    station = relationship("Station", back_populates="records")
