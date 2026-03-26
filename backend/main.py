"""
Module main.py
--------------
Điểm vào (Entry Point) của ứng dụng Máy chủ FastAPI.
Phụ trách công tác khởi tạo Application, khai báo CORS middleware và thiết lập các API endpoint cơ sở (kiểm tra sức khoẻ hệ thống).
"""
import logging
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

# Import module kết nối database cùng cơ chế Dependency Injection
from database import get_db

# Thiết lập hệ thống logging trung tâm
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("api_main")

# Khởi tạo thể hiện chính của ứng dụng FastAPI
app = FastAPI(
    title="24H-GNN Traffic Prediction API",
    description="Hệ thống API RESTful phục vụ kiến trúc dự báo giao thông đô thị dựa trên Graph Neural Networks (GNN).",
    version="1.0.0"
)

# Cấu hình chính sách CORS (Cross-Origin Resource Sharing)
# Ở môi trường Dev, cho phép Frontend React (localhost:3000) gọi API mà không bị chặn
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
async def ping():
    """
    Endpoint kiểm tra trạng thái sức khoẻ máy chủ (Health Check).
    Đóng vai trò xác minh Backend đang hoạt động và đáp ứng request.
    
    Returns:
        dict: Phản hồi bao gồm trạng thái phân tích 'ok' và lời chào.
    """
    logger.info("Nhận lưu lượng kiểm tra sức khỏe hệ thống từ GET /ping.")
    try:
        return {"status": "ok", "message": "Backend 24H-GNN đang hoạt động!"}
    except Exception as e:
        logger.error(f"Ngoại lệ không mong muốn phát sinh tại /ping: {str(e)}")
        raise HTTPException(status_code=500, detail="Đã xảy ra lỗi máy chủ nội bộ bất ngờ")

@app.get("/test-db")
async def test_db_connection(db: AsyncSession = Depends(get_db)):
    """
    Endpoint thử nghiệm kết nối CSDL PostgreSQL bằng cách triệu xuất một truy vấn tĩnh nhẹ nhàng.
    
    Args:
        db (AsyncSession): Phiên giao dịch CSDL tiêm vào qua dependency injection.
        
    Returns:
        dict: Mã trạng thái, thông điệp phân giải và phiên bản engine.
    """
    logger.info("Hệ thống khởi chạy quy trình kiểm tra Database qua GET /test-db.")
    try:
        # Thực thi truy vấn cơ sở nhằm xác minh trạng thái đường truyền
        result = await db.execute(text("SELECT version();"))
        db_version = result.scalar()
        
        logger.info(f"Hoạt động cơ sở dữ liệu trơn tru. Version: {db_version}")
        return {
            "status": "success",
            "message": "Kết nối thành công tới Database PostgreSQL!",
            "db_version": db_version
        }
    except Exception as e:
        logger.error(f"Đường truyền Database trục trặc trong quá trình vận hành ping: {str(e)}")
        # Cần raise HTTP exception thay đổi luồng thực thi (không được fail silently)
        raise HTTPException(status_code=500, detail=f"Không thể thiết lập kết nối Database: {str(e)}")
