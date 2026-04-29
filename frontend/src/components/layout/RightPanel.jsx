import React from 'react';
import { AlertTriangle, Info } from 'lucide-react';
import { useTraffic } from '../../context/TrafficContext';

const RightPanel = () => {
  const { topCongested, markerRefs } = useTraffic();

  const handleRankingClick = (station) => {
    const marker = markerRefs.current[station.id];
    if (marker) {
      const map = marker._map;
      if (map) {
        map.flyTo([station.lat, station.lng], 15, { duration: 1.5 });
        setTimeout(() => {
          marker.openPopup();
          // We can also trigger the chart fetch here, but let the popup's own onClick handle it if needed.
        }, 1500);
      }
    }
  };

  const getMarkerColor = (flow) => {
    const MAX_FLOW = 523;
    if (flow < 0.3 * MAX_FLOW) return '#22c55e';
    if (flow <= 0.7 * MAX_FLOW) return '#f59e0b';
    return '#ef4444';
  };

  return (
    <aside className="w-80 bg-slate-900 border-l border-slate-800 flex flex-col shrink-0">
      <div className="p-4 border-b border-slate-800 flex items-center justify-between">
        <h2 className="text-sm font-bold text-slate-100 flex items-center gap-2">
          <AlertTriangle size={16} className="text-rose-500" /> 
          Top 10 Điểm Nóng
        </h2>
      </div>
      
      <div className="flex-1 overflow-y-auto p-4">
        {topCongested.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-40 text-slate-500 text-center gap-2">
            <Info size={24} />
            <p className="text-xs">Chưa có dữ liệu dự báo.<br/>Hãy chọn mốc thời gian bên trái.</p>
          </div>
        ) : (
          <div className="flex flex-col gap-2">
            {topCongested.map((st, i) => {
              const flow = st.predicted_flow;
              const color = getMarkerColor(flow);
              return (
                <div 
                  key={st.id} 
                  onClick={() => handleRankingClick(st)}
                  className="flex items-center gap-3 p-3 bg-slate-800/30 border border-slate-700/50 rounded-lg hover:bg-slate-800/80 cursor-pointer transition-transform hover:-translate-x-1"
                >
                  <div className={`w-6 h-6 rounded flex items-center justify-center text-xs font-bold text-white ${i === 0 ? 'bg-rose-500' : i === 1 ? 'bg-orange-500' : i === 2 ? 'bg-amber-500' : 'bg-slate-600'}`}>
                    {i + 1}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className={`text-xs truncate ${i < 3 ? 'font-bold text-slate-100' : 'font-semibold text-slate-300'}`}>
                      {st.name}
                    </p>
                    <div className="flex items-center gap-1.5 mt-1">
                      <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: color }}></div>
                      <span className="text-[10px] text-slate-400">{Math.round(flow)} xe/5p</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </aside>
  );
};

export default RightPanel;
