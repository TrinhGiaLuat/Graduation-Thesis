"""
Module schemas.py
-----------------
Khai báo khuôn mẫu validation (Pydantic Models V2) dùng để Serialization and Deserialization cấu trúc dữ liệu JSON vào/ra ứng dụng FastAPI.
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from datetime import datetime

class StationBase(BaseModel):
    """ Lớp Schema nền tảng đối với thông tin trạm không gian """
    station_id_str: str = Field(..., description="Mã chuỗi định danh thực tế (sensor ID)")
    name: Optional[str] = Field(None, description="Tên danh xưng điểm đo đạc")
    lat: float = Field(..., description="Toạ độ không gian chuẩn quốc tế Vĩ độ (Latitude)")
    lng: float = Field(..., description="Toạ độ không gian Kinh độ (Longitude)")

class StationCreate(StationBase):
    """ Schema xác nhận dữ liệu khi API tạo mới Node/Station """
    pass

class StationResponse(StationBase):
    """ Phân giải Schema cho yêu cầu response của API, tiêm thêm ID định tuyến của CSDL """
    id: int
    
    # Kích hoạt chế độ ORM Support cho Pydantic v2
    model_config = ConfigDict(from_attributes=True)

class TrafficRecordBase(BaseModel):
    """ Lớp Schema cơ sở chuẩn hoá Series Thời gian giao thông """
    timestamp: datetime = Field(..., description="Mốc thời gian quy ước")
    volume: float = Field(..., description="Số mật độ phương tiện được đo đếm")
    avg_speed: Optional[float] = Field(None, description="Vận tốc trung vị phương tiện")
    is_prediction: bool = Field(False, description="Tín hiệu kiểm soát (True nếu được sinh ra từ module AI, False là Raw Data)")

class TrafficRecordResponse(TrafficRecordBase):
    """ Schema phản hồi kết xuất của đối tượng lưu lượng API """
    id: int
    station_id: int

    model_config = ConfigDict(from_attributes=True)
