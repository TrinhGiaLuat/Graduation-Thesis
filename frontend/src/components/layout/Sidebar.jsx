import React, { useState } from 'react';
import { Map, MapPin, Activity, History, Search } from 'lucide-react';
import { useTraffic } from '../../context/TrafficContext';

const Sidebar = () => {
  const { 
    activeHorizon, fetchSnapshot,
    currentStep, activeTab, setActiveTab,
    enableAlerts, setEnableAlerts,
    stations, setSelectedStationId
  } = useTraffic();

  const [searchQuery, setSearchQuery] = useState('');
  const [showSuggestions, setShowSuggestions] = useState(false);

  const filteredStations = stations.filter(st => 
    st.name.toLowerCase().includes(searchQuery.toLowerCase()) || 
    st.id.toString() === searchQuery
  ).slice(0, 5); // Hiển thị tối đa 5 gợi ý

  return (
    <aside className="w-72 bg-slate-900 border-r border-slate-800 flex flex-col shrink-0 overflow-y-auto">
      {/* Menu Tabs */}
      <div className="flex border-b border-slate-800 p-2 gap-1 flex-wrap">
        <button 
          onClick={() => setActiveTab('live')}
          className={`flex-1 min-w-[30%] py-2 px-2 rounded-md text-[13px] font-semibold flex items-center justify-center gap-1.5 transition-colors ${activeTab === 'live' ? 'bg-slate-800 text-emerald-400' : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-200'}`}
        >
          <Activity size={14} /> Trực Tiếp
        </button>
        <button 
          onClick={() => setActiveTab('report')}
          className={`flex-1 min-w-[30%] py-2 px-2 rounded-md text-[13px] font-semibold flex items-center justify-center gap-1.5 transition-colors ${activeTab === 'report' ? 'bg-slate-800 text-blue-400' : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-200'}`}
        >
          Báo Cáo
        </button>
        <button 
          onClick={() => setActiveTab('history')}
          className={`flex-1 min-w-[30%] py-2 px-2 rounded-md text-[13px] font-semibold flex items-center justify-center gap-1.5 transition-colors ${activeTab === 'history' ? 'bg-slate-800 text-amber-500' : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-200'}`}
        >
          <History size={14} /> Nhật Ký
        </button>
      </div>

      <div className="p-5 flex flex-col gap-8">
      
        {/* Tìm kiếm trạm */}
        <section className="relative">
          <h2 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-3">
            Tìm Kiếm Trạm
          </h2>
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <Search size={16} className="text-slate-500" />
            </div>
            <input
              type="text"
              placeholder="Nhập tên hoặc ID trạm..."
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setShowSuggestions(true);
              }}
              onFocus={() => setShowSuggestions(true)}
              onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
              className="w-full bg-slate-800/50 border border-slate-700 text-sm text-slate-200 rounded-lg pl-10 pr-3 py-2.5 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all placeholder:text-slate-600"
            />
          </div>
          
          {/* Gợi ý tìm kiếm */}
          {showSuggestions && searchQuery && filteredStations.length > 0 && (
            <div className="absolute z-50 w-full mt-1 bg-slate-800 border border-slate-700 rounded-lg shadow-xl overflow-hidden">
              {filteredStations.map(st => (
                <div 
                  key={st.id}
                  onClick={() => {
                    setSelectedStationId(st.id);
                    setSearchQuery(st.name);
                    setShowSuggestions(false);
                    // Nếu đang ở tab khác thì tự chuyển về tab Trực tiếp để xem bản đồ
                    if(activeTab !== 'live') setActiveTab('live');
                  }}
                  className="px-4 py-2 text-sm text-slate-300 hover:bg-blue-600 hover:text-white cursor-pointer transition-colors border-b border-slate-700/50 last:border-0 flex justify-between items-center"
                >
                  <span>{st.name}</span>
                  <span className="text-[10px] text-slate-500 bg-slate-900 px-1.5 py-0.5 rounded">ID: {st.id}</span>
                </div>
              ))}
            </div>
          )}
          {showSuggestions && searchQuery && filteredStations.length === 0 && (
            <div className="absolute z-50 w-full mt-1 bg-slate-800 border border-slate-700 rounded-lg shadow-xl px-4 py-3 text-sm text-slate-500 text-center">
              Không tìm thấy trạm nào
            </div>
          )}
        </section>
        <section>
          <h2 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4">
            Tầm Nhìn Dự Báo
          </h2>
          <div className="flex flex-col gap-2">
            {[
              { label: '15 phút', horizon: 3 },
              { label: '30 phút', horizon: 6 },
              { label: '60 phút', horizon: 12 }
            ].map(({ label, horizon }) => {
              const isActive = activeHorizon === horizon;
              return (
                <label 
                  key={horizon} 
                  onClick={() => fetchSnapshot(horizon, currentStep)}
                  className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${isActive ? 'bg-blue-500/10 border-blue-500/50' : 'bg-slate-800/30 border-slate-700 hover:bg-slate-800/50'}`}
                >
                  <input type="radio" name="horizon" className="accent-blue-500" checked={isActive} readOnly />
                  <span className={`text-sm font-medium ${isActive ? 'text-blue-400' : 'text-slate-300'}`}>{label}</span>
                </label>
              );
            })}
          </div>
        </section>

        {/* View Options */}
        <section>
          <h2 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4">
            Tuỳ Chọn Tuỳ Biến
          </h2>
          <div className="flex flex-col gap-3">
            {[
              { id: 'heatmap', label: 'Hiển thị Heatmap', icon: <Map size={16} />, active: true },
              { id: 'roads', label: 'Hiển thị Đường (Road)', icon: <MapPin size={16} />, active: false },
            ].map(opt => (
              <label key={opt.id} className="flex items-center justify-between cursor-pointer group">
                <div className="flex items-center gap-2 text-sm text-slate-300 group-hover:text-slate-100 transition-colors">
                  <span className="text-slate-500 group-hover:text-slate-400">{opt.icon}</span>
                  {opt.label}
                </div>
                <div className={`w-8 h-4 rounded-full relative transition-colors ${opt.active ? 'bg-emerald-500' : 'bg-slate-700'}`}>
                  <div className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-transform ${opt.active ? 'left-4' : 'left-0.5'}`}></div>
                </div>
              </label>
            ))}
            
            {/* Nút bật tắt Toast Cảnh báo */}
            <label className="flex items-center justify-between cursor-pointer group mt-2 pt-3 border-t border-slate-800">
              <div className="flex items-center gap-2 text-sm text-slate-300 group-hover:text-slate-100 transition-colors">
                <span className="text-slate-500 group-hover:text-slate-400"><Activity size={16} /></span>
                Bật Cảnh báo Kẹt xe (Toast)
              </div>
              <div 
                onClick={(e) => {
                  e.preventDefault();
                  setEnableAlerts(!enableAlerts);
                }}
                className={`w-8 h-4 rounded-full relative transition-colors ${enableAlerts ? 'bg-rose-500' : 'bg-slate-700'}`}
              >
                <div className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-transform ${enableAlerts ? 'left-4' : 'left-0.5'}`}></div>
              </div>
            </label>
          </div>
        </section>
      </div>
    </aside>
  );
};

export default Sidebar;
