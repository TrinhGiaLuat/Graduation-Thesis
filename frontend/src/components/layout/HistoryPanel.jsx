import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { History, Trash2 } from 'lucide-react';

const HistoryPanel = () => {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await axios.get('/api/reports/history');
        setHistory(res.data);
      } catch (error) {
        console.error("Lỗi khi tải nhật ký:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchHistory();
    // Refresh every 10 seconds to keep it pseudo-live
    const interval = setInterval(fetchHistory, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleReset = async () => {
    if (window.confirm("Bạn có chắc chắn muốn xóa toàn bộ lịch sử dự báo? Hành động này không thể hoàn tác.")) {
      try {
        await axios.delete('/api/reports/reset');
        // Refresh data
        const res = await axios.get('/api/reports/history');
        setHistory(res.data);
        alert("Đã xóa sạch dữ liệu thành công!");
      } catch (error) {
        console.error("Lỗi khi xóa dữ liệu:", error);
        alert("Có lỗi xảy ra khi xóa dữ liệu.");
      }
    }
  };

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center bg-slate-900 text-slate-400">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-4 border-slate-700 border-t-blue-500 rounded-full animate-spin"></div>
          <p>Đang tải nhật ký cảnh báo...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col bg-slate-900 overflow-y-auto p-8 gap-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-slate-100 flex items-center gap-3">
            <History className="text-amber-500" /> Nhật Ký Cảnh Báo & Giám Sát
          </h2>
          <p className="text-slate-400 text-sm mt-1">Lịch sử các trạm có lưu lượng dự báo cao (&gt; 400 xe) được hệ thống ghi nhận.</p>
        </div>
        <div className="flex gap-3 items-center">
          <span className="text-sm text-slate-500 bg-slate-800 px-3 py-1.5 rounded-lg border border-slate-700">
            Hiển thị tối đa <span className="text-amber-500 font-bold">50</span> bản ghi gần nhất
          </span>
          <button 
            onClick={handleReset}
            className="px-4 py-2 bg-slate-800 hover:bg-rose-900/30 text-rose-400 border border-slate-700 hover:border-rose-500/50 rounded-lg font-semibold transition-all flex items-center gap-2"
          >
            <Trash2 size={18} /> Làm mới dữ liệu
          </button>
        </div>
      </div>

      <div className="bg-slate-800/30 border border-slate-700/50 rounded-xl flex-1 flex flex-col min-h-0">
        <div className="overflow-y-auto flex-1">
          <table className="w-full text-left text-sm text-slate-400">
            <thead className="text-xs uppercase bg-slate-800/80 text-slate-300 border-b border-slate-700 sticky top-0 z-10">
              <tr>
                <th className="px-6 py-4">Thời Gian (Giờ Ảo)</th>
                <th className="px-6 py-4">Tầm Nhìn Dự Báo</th>
                <th className="px-6 py-4">Trạm Giao Thông</th>
                <th className="px-6 py-4">Mức Lưu Lượng (xe/5p)</th>
                <th className="px-6 py-4">Trạng Thái</th>
              </tr>
            </thead>
            <tbody>
              {history.length === 0 ? (
                <tr>
                  <td colSpan="5" className="px-6 py-12 text-center text-slate-500 text-base">Chưa có dữ liệu cảnh báo nào được ghi nhận.</td>
                </tr>
              ) : (
                history.map((item, idx) => (
                  <tr key={idx} className="border-b border-slate-700/50 hover:bg-slate-800/50 transition-colors">
                    <td className="px-6 py-4 font-medium text-slate-300">{item.time}</td>
                    <td className="px-6 py-4">+{item.horizon * 5} phút</td>
                    <td className="px-6 py-4">{item.station}</td>
                    <td className="px-6 py-4 font-bold text-slate-200">{item.flow}</td>
                    <td className="px-6 py-4">
                      {item.is_critical ? (
                        <span className="bg-rose-500/20 text-rose-400 text-[10px] font-bold px-3 py-1.5 rounded border border-rose-500/30 uppercase tracking-wider">
                          Kẹt Xe Nặng
                        </span>
                      ) : (
                        <span className="bg-orange-500/20 text-orange-400 text-[10px] font-bold px-3 py-1.5 rounded border border-orange-500/30 uppercase tracking-wider">
                          Nguy Cơ
                        </span>
                      )}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default HistoryPanel;
