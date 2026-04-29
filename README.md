<div align="center">

#  Xây Dựng Ứng Dụng Web Dự Báo Lưu Lượng Giao Thông Đô Thị Sử Dụng Mạng Nơ-ron Đồ Thị (GNN)

**Ứng dụng Graph Neural Network (GNN) dự báo lưu lượng giao thông đô thị**

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://postgresql.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

<br/>

> 📍 **Đồ án Tốt nghiệp** — Trịnh Gia Luật  
> 🏫 Ngành Kỹ thuật Phần mềm  
> 🗓️ Năm học 2025 – 2026

</div>

---

## 📋 Mục Lục

1. [Giới thiệu dự án](#-giới-thiệu-dự-án)
2. [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
3. [Thuật toán AI — Graph WaveNet](#-thuật-toán-ai--graph-wavenet)
4. [Kết quả huấn luyện](#-kết-quả-huấn-luyện)
5. [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
6. [Hướng dẫn cài đặt & khởi chạy](#-hướng-dẫn-cài-đặt--khởi-chạy)
7. [Giao diện Dashboard](#-giao-diện-dashboard)
8. [Tổng quan API Backend](#-tổng-quan-api-backend)
9. [Công nghệ sử dụng](#-công-nghệ-sử-dụng)

---

## 🎯 Giới Thiệu Dự Án

**Ứng Dụng Web** là một hệ thống giám sát và dự báo lưu lượng giao thông đô thị thông minh, được phát triển như một sản phẩm hoàn chỉnh từ tầng AI cho đến giao diện người dùng.

### Bài toán đặt ra

Ùn tắc giao thông là vấn đề nghiêm trọng tại các đô thị lớn, gây thiệt hại kinh tế hàng nghìn tỷ đồng mỗi năm. Các hệ thống giám sát hiện tại thường chỉ phản ứng **sau khi** tắc đường xảy ra. Dự án này đặt mục tiêu chuyển đổi sang mô hình **dự báo chủ động**: phát hiện nguy cơ ùn tắc **trước 15 – 60 phút** để cơ quan quản lý có thời gian điều phối.

### Giải pháp đề xuất

Ứng dụng mô hình học sâu trên đồ thị (**Graph WaveNet**) để:
- Học đồng thời **mối quan hệ không gian** giữa các nút giao thông (ai ảnh hưởng đến ai)
- Nhận diện **quy luật thời gian** theo chu kỳ (giờ cao điểm sáng / chiều)
- Dự báo lưu lượng trong **3 tầm nhìn song song**: +15 phút, +30 phút, +60 phút

---

## 🏗️ Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────────────────────────┐
│                         NGƯỜI DÙNG (Browser)                         │
│                      React Dashboard (Port 3000)                     │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ HTTP / REST API
┌───────────────────────────────▼─────────────────────────────────────┐
│                         BACKEND LAYER                                │
│              FastAPI + SQLAlchemy Async (Port 8000)                  │
│                                                                      │
│   ┌──────────────┐   ┌──────────────┐   ┌────────────────────────┐  │
│   │ /api/traffic │   │/api/reports  │   │    predict_service.py  │  │
│   │  (snapshot)  │   │ (summary,    │   │  PredictService class  │  │
│   │  (history)   │   │  history)    │   │  → GraphWaveNet model  │  │
│   └──────────────┘   └──────────────┘   └────────────────────────┘  │
└───────────┬─────────────────────────────────────┬───────────────────┘
            │                                     │
┌───────────▼────────┐               ┌────────────▼───────────────────┐
│    PostgreSQL DB   │               │          AI CORE               │
│  (Port 5432)       │               │  Graph WaveNet (PyTorch)       │
│  - stations        │               │  307 nodes × 5 features        │
│  - traffic_records │               │  Input: 12 steps (60p quá khứ) │
│  - prediction_logs │               │  Output: 12 steps (60p tương   │
│                    │               │  lai) — đa tầm nhìn           │
└────────────────────┘               └────────────────────────────────┘
```

> **Docker Compose** điều phối toàn bộ 3 services (Frontend, Backend, Database) trong một lệnh duy nhất.

---

## 🧠 Thuật Toán AI — Graph WaveNet

### Tổng quan kiến trúc

Mô hình **Graph WaveNet** ([Wu et al., 2019](https://arxiv.org/abs/1906.00121)) kết hợp hai thành phần học sâu chính:

#### 1. Lớp Temporal — Dilated Causal TCN

```
Input: (Batch, Features, Nodes, Timesteps)
        ↓
  Gated Temporal Convolution (dilation = 1, 2, 4, 8...)
        ↓
  Hàm kích hoạt: tanh(W_f * x) ⊙ σ(W_g * x)  [Gated Activation]
```

Lớp TCN với dilation tăng dần giúp mô hình học được **phụ thuộc dài hạn** (quy luật giờ cao điểm) mà không làm tăng số tham số.

#### 2. Lớp Spatial — Adaptive Graph Convolution

```
Z = σ(A_hat · X · W_spatial)
    
Trong đó A_hat = SoftMax(ReLU(E_1 · E_2^T))
```

- **E_1, E_2** là các ma trận nhúng (Embedding) của từng nút — được học hoàn toàn tự động từ dữ liệu
- **Self-Adaptive Adjacency Matrix**: AI tự xây dựng ma trận quan hệ giữa các trạm mà không cần cung cấp sẵn đồ thị, giúp phát hiện các liên kết ẩn bản đồ không thể hiện

#### 3. Cơ chế Dự báo Đa tầm nhìn (Multi-Horizon)

```
Đầu vào: X_(t-11) → X_(t-10) → ... → X_t    [12 mốc quá khứ]
                          ↓
                   Graph WaveNet
                          ↓
Đầu ra:  Ŷ_(t+1) → Ŷ_(t+2) → ... → Ŷ_(t+12)  [12 mốc tương lai]
         (+5 phút)             (+60 phút)
```

Mô hình dự báo **đồng thời** 12 mốc tương lai trong một lần suy diễn (one-shot inference), tạo ra dự báo nhất quán cho tầm nhìn 15, 30 và 60 phút.

---

## 📊 Kết Quả Huấn Luyện

Mô hình được huấn luyện và đánh giá trên tập dữ liệu **PeMS04** (San Francisco Bay Area):

| Tầm nhìn | MAE ↓ | RMSE ↓ | MAPE ↓ | R² ↑ |
|:--------:|:-----:|:------:|:------:|:----:|
| **15 phút** | 17.67 | 28.84 | 12.03% | 0.9668 |
| **30 phút** | 18.52 | 30.33 | 12.47% | 0.9633 |
| **60 phút** | 19.94 | 32.54 | 13.48% | 0.9577 |
| **Trung bình** | **18.53** | **30.33** | **12.55%** | **0.9633** |

> **R² = 0.9633** — Mô hình giải thích được **96.3%** sự biến thiên của lưu lượng thực tế.

### Thông số tập dữ liệu PeMS04

| Thuộc tính | Giá trị |
|:-----------|:--------|
| Số trạm đo | 307 cảm biến |
| Chu kỳ đo | 5 phút / lần |
| Thời gian | 2/1/2018 – 28/2/2018 (59 ngày) |
| Đặc trưng | Flow (lưu lượng), Occupancy (mật độ), Speed (tốc độ), ToD, DoW |
| Phân chia | Train 70% / Val 10% / Test 20% |

---

## 📁 Cấu Trúc Thư Mục

```
Graduation-Thesis/
│
├── 📂 ai_core/                      # Toàn bộ mã nguồn AI
│   ├── 📂 models/
│   │   └── graph_wavenet.py         # Kiến trúc mạng GNN chính
│   ├── 📂 data/processed/pems04/    # Dữ liệu đã tiền xử lý (gitignored)
│   ├── 📂 checkpoints/              # Trọng số mô hình .pt (gitignored)
│   ├── preprocess_pems04_multimodal.py  # ETL pipeline 5 features
│   ├── train_pems04.py              # Script huấn luyện Graph WaveNet
│   ├── evaluate_pems04.py           # Script đánh giá MAE/RMSE/MAPE
│   ├── train_eval_lstm.py           # Baseline LSTM so sánh
│   └── train_eval_stgcn.py          # Baseline STGCN so sánh
│
├── 📂 backend/                      # API Server (FastAPI)
│   ├── 📂 app/services/
│   │   └── predict_service.py       # Core AI Inference Service
│   ├── 📂 routers/
│   │   ├── traffic.py               # /api/traffic/ — Dữ liệu & Snapshot
│   │   └── reports.py               # /api/reports/ — Báo cáo & Nhật ký
│   ├── main.py                      # Entry point FastAPI + Lifespan
│   ├── database.py                  # AsyncPG Engine + Session
│   ├── models.py                    # ORM Models (Station, PredictionLog...)
│   ├── seed_pems04.py               # Script nạp 307 trạm vào DB
│   └── requirements.txt
│
├── 📂 frontend/                     # Giao diện React
│   ├── 📂 src/
│   │   ├── 📂 components/
│   │   │   ├── TrafficMap.js        # Bản đồ Leaflet + Marker động
│   │   │   ├── 📂 layout/
│   │   │   │   ├── Header.jsx       # Thanh đầu trang + AI Model Info
│   │   │   │   ├── Sidebar.jsx      # Menu điều hướng + Tìm kiếm trạm
│   │   │   │   ├── RightPanel.jsx   # Top 10 trạm ùn tắc
│   │   │   │   ├── ReportPanel.jsx  # Trang Báo cáo KPI
│   │   │   │   └── HistoryPanel.jsx # Trang Nhật ký cảnh báo
│   │   │   └── 📂 auth/
│   │   │       └── LoginPage.jsx    # Trang đăng nhập
│   │   ├── 📂 context/
│   │   │   └── TrafficContext.js    # Global State + Polling logic
│   │   ├── App.js
│   │   └── index.js
│   ├── tailwind.config.js
│   └── package.json
│
├── docker-compose.yml               # Orchestration 3 services
├── .gitignore
└── README.md
```

---

## 🚀 Hướng Dẫn Cài Đặt & Khởi Chạy

### Yêu cầu tiên quyết

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) đã được cài đặt và đang chạy
- Git

### Bước 1 — Clone dự án

```bash
git clone https://github.com/TrinhGiaLuat/Graduation-Thesis.git
cd Graduation-Thesis
```

### Bước 2 — Chuẩn bị dữ liệu AI *(Bỏ qua nếu đã có)*

> Vì dữ liệu lớn (~300MB) không được đẩy lên Git, bạn cần tự đặt vào đúng vị trí:

```
ai_core/
├── data/processed/pems04/
│   ├── train.npz
│   ├── val.npz
│   ├── test.npz
│   ├── scaler.pkl
│   └── adj_mx.pkl
└── checkpoints/
    └── model_pems04_best.pt
```

### Bước 3 — Khởi chạy toàn hệ thống

```bash
docker-compose up -d --build
```

Lần đầu build sẽ mất khoảng **15–20 phút** để cài đặt các thư viện. Các lần tiếp theo chỉ mất **~30 giây**.

### Bước 4 — Nạp dữ liệu trạm vào Database *(Chỉ lần đầu)*

```bash
docker exec -it 24h_gnn_backend python seed_pems04.py
```

### Bước 5 — Truy cập hệ thống

| Dịch vụ | Địa chỉ |
|:--------|:--------|
| 🌐 **Dashboard Web** | http://localhost:3000 |
| ⚡ **API Backend** | http://localhost:8000 |
| 📚 **API Docs (Swagger)** | http://localhost:8000/docs |

**Tài khoản đăng nhập mặc định:**
```
Username: Trinhgialuat
Password: Trinhgialuat123@
```

### Dừng hệ thống

```bash
docker-compose down
```

---

## 🖥️ Giao Diện Dashboard

Hệ thống giao diện được thiết kế theo phong cách **Dark Mode SaaS**, bao gồm 3 trang chính:

### Tab 1 — Giám sát Trực tuyến
- **Bản đồ Leaflet** thể hiện 307 trạm với màu sắc động theo lưu lượng:
  - 🟢 Xanh: Thông thoáng (< 30% ngưỡng)
  - 🟡 Vàng: Đông xe (30–70% ngưỡng)
  - 🔴 Đỏ: Tắc nghẽn (> 70% ngưỡng)
- **Biểu đồ chi tiết** khi click vào từng trạm (lịch sử + dự báo GNN)
- **Top 10 trạm ùn tắc** được xếp hạng thời gian thực
- **Ô tìm kiếm trạm** theo tên/ID với gợi ý tự động và tính năng bay tới (FlyTo) trên bản đồ
- **Toggle cảnh báo Toast** khi phát hiện trạm vượt ngưỡng 500 xe/5 phút

### Tab 2 — Báo cáo Phân tích
- **KPI cards**: Tổng lượt quét hệ thống, Tổng phương tiện giám sát, Số điểm đen tắc nghẽn
- **Biểu đồ diện tích** (Area Chart) theo dõi xu hướng lưu lượng toàn mạng lưới
- **Bảng Top trạm** có lưu lượng trung bình cao nhất trong phiên giám sát

### Tab 3 — Nhật Ký Cảnh Báo
- Lịch sử chi tiết các sự kiện cảnh báo (trạm vượt ngưỡng 400 xe)
- Lọc theo tầm nhìn dự báo và trạng thái (Nguy cơ / Kẹt xe nặng)
- Hiển thị tối đa **50 bản ghi gần nhất** từ database để không làm nặng giao diện

### Modal Thông tin AI
- Nhấn icon **(i)** trên thanh header để xem bảng giải thích kiến trúc mô hình GNN
- Thông số hiệu suất thực nghiệm (MAE, RMSE, MAPE) trực tiếp từ tập Test

---

## 📡 Tổng Quan API Backend

### Traffic Endpoints

| Method | Endpoint | Mô tả |
|:------:|:---------|:------|
| `GET` | `/api/stations` | Lấy danh sách 307 trạm (tọa độ, tên) |
| `GET` | `/api/traffic/{station_id}` | Lịch sử + dự báo chi tiết 1 trạm |
| `GET` | `/api/traffic/snapshot` | Dự báo **toàn bộ 307 trạm** tại 1 horizon |

**Tham số Snapshot:**
```
/api/traffic/snapshot?horizon=3&timestep=10
                              ↑              ↑
                        3 = +15 phút    Bước thời gian
                        6 = +30 phút    (8:00 + 10×5p = 8:50)
                        12 = +60 phút
```

### Report Endpoints

| Method | Endpoint | Mô tả |
|:------:|:---------|:------|
| `GET` | `/api/reports/summary` | Thống kê tổng hợp KPI phiên giám sát |
| `GET` | `/api/reports/history` | Nhật ký 50 cảnh báo gần nhất |
| `DELETE` | `/api/reports/reset` | Xóa sạch dữ liệu log DB + CSV |

---

## 🛠️ Công Nghệ Sử Dụng

### Tầng AI & Khoa học Dữ liệu

| Thư viện | Phiên bản | Vai trò |
|:---------|:---------:|:--------|
| PyTorch | 2.0+ | Framework huấn luyện Deep Learning |
| NumPy | 1.24+ | Xử lý mảng tensor đa chiều |
| Scikit-learn | 1.3+ | Chuẩn hóa dữ liệu (Z-score Scaler) |
| Matplotlib | 3.7+ | Vẽ biểu đồ kết quả huấn luyện |

### Tầng Backend

| Công nghệ | Vai trò |
|:----------|:--------|
| **FastAPI** | Framework API bất đồng bộ hiệu năng cao |
| **SQLAlchemy 2.0** | ORM Async với PostgreSQL |
| **asyncpg** | Driver kết nối PostgreSQL bất đồng bộ |
| **Uvicorn** | ASGI Server production-ready |

### Tầng Frontend

| Công nghệ | Vai trò |
|:----------|:--------|
| **React 18** | UI Framework, Hooks API |
| **Tailwind CSS** | Utility-first CSS, Dark Mode |
| **React Leaflet** | Bản đồ tương tác, Marker, FlyTo |
| **Recharts** | Biểu đồ Line/Area Chart |
| **Axios** | HTTP Client gọi API |
| **react-hot-toast** | Hệ thống thông báo Toast |
| **lucide-react** | Bộ icon hiện đại SVG |

### Hạ tầng

| Công nghệ | Vai trò |
|:----------|:--------|
| **Docker & Docker Compose** | Containerization, Orchestration |
| **PostgreSQL 15** | Cơ sở dữ liệu quan hệ chính |
| **GitHub** | Quản lý mã nguồn và phiên bản |

---

<div align="center">

**Made with ❤️ for Graduation Thesis 2025–2026**

*Trịnh Gia Luật — Kỹ thuật Phần mềm*

</div>
