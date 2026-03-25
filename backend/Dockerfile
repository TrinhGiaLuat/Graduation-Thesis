# Sử dụng base image python:3.10-slim
FROM python:3.10-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy tệp requirements và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Mở port
EXPOSE 8000

# Lệnh chạy mặc định
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
