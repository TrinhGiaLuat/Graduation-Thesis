"""
Module database.py
------------------
Chịu trách nhiệm thiết lập và quản lý kết nối cơ sở dữ liệu (PostgreSQL) sử dụng SQLAlchemy với engine bất đồng bộ (asyncpg).
Cung cấp session kết nối thông qua hàm get_db() phục vụ cho cơ chế Dependency Injection trong hệ sinh thái FastAPI.
"""
import os
import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from dotenv import load_dotenv

# Thiết lập logging ghi lại tiến trình xử lý
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Tải biến môi trường từ mốc root cấu hình .env (nếu có sẵn ngoài Docker)
load_dotenv()

# Truy xuất thông tin cấu hình môi trường cho kiến trúc Database
POSTGRES_USER = os.getenv("POSTGRES_USER", "admin")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "adminsecret")
POSTGRES_DB = os.getenv("POSTGRES_DB", "traffic_db")
# Mặc định localhost để Windows/Local kết nối được port 5432. Docker sẽ Override lại bằng biến môi trường.
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

# Chuỗi kết nối Database theo chuẩn asyncpg
DATABASE_URL = f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

class Base(DeclarativeBase):
    """
    Lớp nền tảng (Base) theo chuẩn SQLAlchemy 2.0 — các Model ORM kế thừa class này
    để tự động đăng ký metadata vào hệ thống ánh xạ quan hệ.
    """
    pass

try:
    # Khởi tạo engine bất đồng bộ
    engine = create_async_engine(DATABASE_URL, echo=True)
    
    # Factory khởi tạo phiên kết nối AsyncSession
    AsyncSessionLocal = async_sessionmaker(
        bind=engine, 
        class_=AsyncSession, 
        expire_on_commit=False
    )
    logger.info("Hoàn tất thiết lập cơ chế kết nốt Database Engine (asyncpg).")
except Exception as e:
    logger.error(f"Thất bại trong việc kiến tạo CSDL Engine: {str(e)}")
    raise e

async def get_db():
    """
    Hàm cung cấp Dependency Injection cho các API Endpoint để truy cập Session CSDL.
    Cơ chế Try-Finally đảm bảo rằng connection luôn được đóng lại sau khi hoàn thành request, 
    tránh rò rỉ bộ nhớ hoặc kiệt quệ connection pool.
    
    Yields:
        AsyncSession: Phiên kết nối bất đồng bộ tới Cơ sở dữ liệu PostgreSQL.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Xảy ra ngoại lệ trong quá trình thao tác Database Session: {str(e)}")
            raise
        finally:
            await session.close()
            # Log DEBUG thay vì INFO để tránh spam console trên mỗi request
            logger.debug("Đã đóng kết nối Database Session một cách an toàn.")
