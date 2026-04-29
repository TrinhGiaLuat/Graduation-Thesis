import React, { useState } from 'react';
import { Clock, User, LogOut, Info, Cpu, Network, Activity as ActivityIcon } from 'lucide-react';
import { useTraffic } from '../../context/TrafficContext';

const Header = ({ onLogout }) => {
  const { getVirtualTime } = useTraffic();
  const [showModelInfo, setShowModelInfo] = useState(false);

  return (
    <header className="h-14 bg-slate-900 border-b border-slate-800 flex items-center justify-between px-6 shrink-0">
      {/* Left */}
      <div className="flex flex-col flex-1">
        <div className="flex items-center gap-3">
          <h1 className="text-slate-100 font-bold text-lg leading-tight tracking-wide">
            Bảng Điều Khiển Dự Báo Giao Thông
          </h1>
          <button 
            onClick={() => setShowModelInfo(true)}
            className="p-1 rounded-full hover:bg-slate-800 text-slate-500 hover:text-blue-400 transition-colors"
            title="Thông tin Mô hình AI"
          >
            <Info size={16} />
          </button>
        </div>
        <span className="text-slate-400 text-xs">Hệ thống dự báo giao thông thông minh</span>
      </div>

      {/* Center - Indicator */}
      <div className="flex items-center gap-2 px-4 py-1.5 rounded-full border transition-colors bg-slate-800/80 border-emerald-500/30">
        <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.8)]"></div>
        <span className="text-xs font-bold tracking-widest uppercase text-emerald-400">
          ĐANG GIÁM SÁT TRỰC TUYẾN
        </span>
      </div>

      {/* Right */}
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2 text-slate-300">
          <Clock size={16} className="text-slate-500" />
          <span className="text-sm font-medium">{getVirtualTime()}</span>
        </div>
        <div className="flex items-center gap-4 text-slate-300 border-l border-slate-700 pl-6">
          <div className="flex items-center gap-2">
            <User size={16} className="text-slate-500" />
            <div className="flex flex-col">
              <span className="text-sm font-bold text-slate-200 leading-tight">Trịnh Gia Luật</span>
              <span className="text-[10px] text-slate-500 uppercase font-semibold tracking-wider">Quản trị viên</span>
            </div>
          </div>
          <button 
            onClick={onLogout}
            title="Đăng xuất"
            className="p-1.5 hover:bg-rose-500/10 text-slate-500 hover:text-rose-400 rounded-md transition-colors"
          >
            <LogOut size={16} />
          </button>
        </div>
      </div>

      {/* Model Info Modal */}
      {showModelInfo && (
        <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={() => setShowModelInfo(false)}>
          <div className="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl w-[600px] overflow-hidden" onClick={e => e.stopPropagation()}>
            <div className="flex justify-between items-center px-6 py-4 border-b border-slate-800 bg-slate-800/50">
              <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                <Cpu size={20} className="text-blue-500" />
                Kiến trúc Mô hình AI (Graph WaveNet)
              </h2>
              <button onClick={() => setShowModelInfo(false)} className="text-slate-500 hover:text-rose-500 font-bold text-xl">&times;</button>
            </div>
            
            <div className="p-6 flex flex-col gap-5 text-sm text-slate-300">
              <div className="flex gap-4 items-start">
                <div className="p-3 bg-blue-500/10 rounded-lg text-blue-400 shrink-0"><Network size={24} /></div>
                <div>
                  <h3 className="text-slate-200 font-bold mb-1 text-base">Đồ thị Không gian (Spatial Graph)</h3>
                  <p>Hệ thống mô hình hóa <b>307 trạm cảm biến</b> (Nodes) thành một đồ thị không gian. Khác với các mạng CNN truyền thống, Graph Neural Network (GNN) cho phép AI nhận biết được sự ảnh hưởng lẫn nhau giữa các ngã tư thông qua ma trận kề (Adjacency Matrix).</p>
                </div>
              </div>
              
              <div className="flex gap-4 items-start">
                <div className="p-3 bg-emerald-500/10 rounded-lg text-emerald-400 shrink-0"><ActivityIcon size={24} /></div>
                <div>
                  <h3 className="text-slate-200 font-bold mb-1 text-base">Chuỗi Thời gian (Temporal TCN)</h3>
                  <p>Mô hình sử dụng mạng tích chập mở rộng (Dilated Temporal Convolutional Network) để học các quy luật chu kỳ theo thời gian (giờ cao điểm sáng/chiều). Nó nhận vào <b>12 mốc quá khứ (60 phút)</b> để dự đoán đồng thời <b>12 mốc tương lai</b>.</p>
                </div>
              </div>

              <div className="bg-slate-800 rounded-lg p-4 mt-2 border border-slate-700/50">
                <h4 className="text-slate-300 font-bold mb-2 text-xs uppercase tracking-wider">Thông số hiệu suất (PeMS04 Dataset)</h4>
                <div className="grid grid-cols-3 gap-4">
                  <div className="text-center">
                    <p className="text-2xl font-bold text-emerald-400">18.53</p>
                    <p className="text-[10px] text-slate-500 uppercase mt-1">MAE (Sai số tuyệt đối)</p>
                  </div>
                  <div className="text-center border-l border-slate-700">
                    <p className="text-2xl font-bold text-blue-400">30.33</p>
                    <p className="text-[10px] text-slate-500 uppercase mt-1">RMSE (Độ lệch chuẩn)</p>
                  </div>
                  <div className="text-center border-l border-slate-700">
                    <p className="text-2xl font-bold text-amber-400">12.55%</p>
                    <p className="text-[10px] text-slate-500 uppercase mt-1">MAPE (Tỷ lệ sai số %)</p>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="px-6 py-4 bg-slate-800/30 border-t border-slate-800 flex justify-end">
              <button onClick={() => setShowModelInfo(false)} className="px-5 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors">
                Đã Hiểu
              </button>
            </div>
          </div>
        </div>
      )}
    </header>
  );
};

export default Header;
