"""
Module main.py
--------------
Điểm vào (Entry Point) của ứng dụng Máy chủ FastAPI.
Phụ trách công tác khởi tạo Application, khai báo CORS middleware, khởi chạy Database Schema theo Lifecycle và thiết lập API tĩnh.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

# Import engine từ cơ cấu Database
from database import get_db, engine
# Import các Models nhằm đăng ký ORM Declarative Entity metadata vào luồng kiểm tra
import models

# Cấu hình kiến trúc ghi log nhật ký hệ thống
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("api_main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Quản lý vòng đời hoạt động của ứng dụng FastAPI đồng bộ CSDL thời gian thực.
    Tiến hành cơ chế 'Create-All' ánh xạ SQLAlchemy Metadata sang Object-Relational Model.
    """
    logger.info("Chạy bộ nạp khởi động (Bootstrap). Cấu hình di trú Schema Model vào DB...")
    import asyncio
    max_retries = 5
    for attempt in range(max_retries):
        try:
            async with engine.begin() as conn:
                # Uỷ thác Asyncio thực phi thao tác synchronous (Tạo Schema)
                await conn.run_sync(models.Base.metadata.create_all)
            logger.info("Hoàn tất việc kết xuất các Table thực thể (Stations, RoadSegments) trên máy chủ DB.")
            break # Thoát vòng lặp nếu thành công
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Database đang khởi động... Thử lại sau 3 giây (Lần {attempt + 1}/{max_retries})")
                await asyncio.sleep(3)
            else:
                logger.error(f"Xảy ra lỗi can thiệp tạo kiến trúc CSDL: {str(e)}")
    
    yield  # Uỷ thác quyền điều khiển lại API Routing
    
    # Bước đóng hệ thống an toàn ở điểm Outbound
    logger.info("Máy chủ chuẩn bị dừng các dịch vụ. Dọn dẹp Database connections...")
    await engine.dispose()

# Giao diện lập trình gốc FastAPI ứng với Lifecycle Async Context Manager
app = FastAPI(
    title="24H-GNN Traffic Prediction API",
    description="Hệ thống API RESTful phục vụ kiến trúc dự báo giao thông đô thị dựa trên Graph Neural Networks (GNN).",
    version="1.0.0",
    lifespan=lifespan
)

# Đăng ký các phân hệ (Routers) vào ứng dụng API chính
from routers.traffic import router as traffic_router
app.include_router(traffic_router)

# Cấp quyền CORS để frontend localhost truy cập an toàn
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
    Endpoint thử nghiệm kết nối CSDL PostgreSQL tại lớp Async.
    """
    logger.info("Khởi chạy quy trình xác nhận DB bằng GET /test-db.")
    try:
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
        raise HTTPException(status_code=500, detail=f"Không thể thiết lập kết nối Database: {str(e)}")
