import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer } from 'recharts';
import { Activity, Car, AlertTriangle, TrendingUp, Trash2 } from 'lucide-react';

const ReportPanel = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchReport = async () => {
      try {
        const resSummary = await axios.get('/api/reports/summary');
        setData(resSummary.data);
      } catch (error) {
        console.error("Lỗi khi tải báo cáo:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchReport();
    // Refresh every 10 seconds to keep it pseudo-live
    const interval = setInterval(fetchReport, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleReset = async () => {
    if (window.confirm("Bạn có chắc chắn muốn xóa toàn bộ lịch sử dự báo? Hành động này không thể hoàn tác.")) {
      try {
        await axios.delete('/api/reports/reset');
        // Refresh data
        const resSummary = await axios.get('/api/reports/summary');
        setData(resSummary.data);
        alert("Đã xóa sạch dữ liệu thành công!");
      } catch (error) {
        console.error("Lỗi khi xóa dữ liệu:", error);
        alert("Có lỗi xảy ra khi xóa dữ liệu.");
      }
    }
  };

  if (loading || !data) {
    return (
      <div className="flex-1 flex items-center justify-center bg-slate-900 text-slate-400">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-4 border-slate-700 border-t-blue-500 rounded-full animate-spin"></div>
          <p>Đang trích xuất dữ liệu báo cáo...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col bg-slate-900 overflow-y-auto p-8 gap-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-slate-100 flex items-center gap-3">
            <Activity className="text-blue-500" /> Báo Cáo Phân Tích Tổng Quan
          </h2>
          <p className="text-slate-400 text-sm mt-1">Dữ liệu được tổng hợp từ lịch sử giả lập thời gian thực</p>
        </div>
        <div className="flex gap-3">
          <button 
            onClick={handleReset}
            className="px-4 py-2 bg-slate-800 hover:bg-rose-900/30 text-rose-400 border border-slate-700 hover:border-rose-500/50 rounded-lg font-semibold transition-all flex items-center gap-2"
          >
            <Trash2 size={18} /> Làm mới dữ liệu
          </button>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-3 gap-6">
        <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-5 flex items-center gap-4 group relative">
          <div className="w-12 h-12 rounded-full bg-blue-500/20 flex items-center justify-center text-blue-500 shrink-0">
            <TrendingUp size={24} />
          </div>
          <div>
            <p className="text-slate-400 text-sm font-medium">Số Lượt Quét Hệ Thống</p>
            <p className="text-2xl font-bold text-slate-100">{data.total_predictions.toLocaleString()}</p>
            <p className="text-[10px] text-slate-500 mt-1">Đánh giá mức độ hoạt động liên tục của AI.</p>
          </div>
        </div>
        <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-5 flex items-center gap-4">
          <div className="w-12 h-12 rounded-full bg-emerald-500/20 flex items-center justify-center text-emerald-500 shrink-0">
            <Car size={24} />
          </div>
          <div>
            <p className="text-slate-400 text-sm font-medium">Tổng Lượt Phương Tiện</p>
            <p className="text-2xl font-bold text-slate-100">{Math.round(data.total_flow).toLocaleString()} <span className="text-sm font-normal text-slate-500">xe</span></p>
            <p className="text-[10px] text-slate-500 mt-1">Áp lực giao thông đã ghi nhận trong phiên.</p>
          </div>
        </div>
        <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-5 flex items-center gap-4">
          <div className="w-12 h-12 rounded-full bg-rose-500/20 flex items-center justify-center text-rose-500 shrink-0">
            <AlertTriangle size={24} />
          </div>
          <div>
            <p className="text-slate-400 text-sm font-medium">Điểm Đen Tắc Nghẽn</p>
            <p className="text-2xl font-bold text-slate-100">{data.top_stations.length} <span className="text-sm font-normal text-slate-500">trạm</span></p>
            <p className="text-[10px] text-slate-500 mt-1">Số trạm lọt top báo động đỏ dài hạn.</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-6 flex-1 min-h-[400px]">
        {/* Trend Chart */}
        <div className="col-span-2 bg-slate-800/30 border border-slate-700/50 rounded-xl p-5 flex flex-col">
          <div className="mb-4">
            <h3 className="text-lg font-bold text-slate-200">Xu Hướng Lưu Lượng Toàn Mạng Lưới</h3>
            <p className="text-[11px] text-slate-400">Đánh giá phân tích vĩ mô: Nhìn vào đường biểu diễn để nhận biết thành phố đang bước vào giờ cao điểm hay bắt đầu vãn xe.</p>
          </div>
          <div className="flex-1 w-full min-h-0">
            {data.trend_data.length === 0 ? (
              <div className="h-full flex items-center justify-center text-slate-500">Chưa có đủ dữ liệu để vẽ biểu đồ</div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data.trend_data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorFlow" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                  <XAxis dataKey="time" stroke="#94a3b8" fontSize={12} tickMargin={10} />
                  <YAxis stroke="#94a3b8" fontSize={12} />
                  <RechartsTooltip 
                    contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', borderRadius: '8px' }}
                    itemStyle={{ color: '#3b82f6', fontWeight: 'bold' }}
                  />
                  <Area type="monotone" dataKey="flow" name="Tổng Lưu Lượng" stroke="#3b82f6" strokeWidth={3} fillOpacity={1} fill="url(#colorFlow)" />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        {/* Top Stations */}
        <div className="bg-slate-800/30 border border-slate-700/50 rounded-xl p-5 flex flex-col">
          <div className="mb-4">
            <h3 className="text-lg font-bold text-rose-400 flex items-center gap-2">
              <AlertTriangle size={18} /> Top Điểm Nóng Thường Xuyên
            </h3>
            <p className="text-[11px] text-slate-400 mt-1">Danh sách các trạm có lưu lượng trung bình cao nhất trong lịch sử phiên trực.</p>
          </div>
          <div className="flex-1 flex flex-col gap-3 overflow-y-auto pr-2">
            {data.top_stations.length === 0 ? (
              <div className="flex-1 flex items-center justify-center text-slate-500">Chưa có dữ liệu</div>
            ) : (
              data.top_stations.map((st, i) => {
                const isCritical = st.avg_flow > 500;
                return (
                  <div key={st.station_id} className={`bg-slate-800/80 border ${isCritical ? 'border-rose-500/50' : 'border-slate-700'} rounded-lg p-3 flex items-center gap-3`}>
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold text-white ${i === 0 ? 'bg-rose-500' : i === 1 ? 'bg-orange-500' : 'bg-amber-500'}`}>
                      #{i + 1}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-bold text-slate-200 truncate">{st.name}</p>
                      <div className="flex justify-between items-center mt-0.5">
                        <p className="text-xs text-slate-400">Trung bình: <span className="text-rose-400 font-semibold">{st.avg_flow} xe/5p</span></p>
                        {isCritical && <span className="text-[9px] px-1.5 py-0.5 bg-rose-500/20 text-rose-400 rounded border border-rose-500/30 uppercase font-bold tracking-wider">Cảnh báo</span>}
                      </div>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ReportPanel;
