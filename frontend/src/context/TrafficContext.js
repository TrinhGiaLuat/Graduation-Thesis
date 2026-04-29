import React, { createContext, useContext, useState, useRef, useEffect } from 'react';
import axios from 'axios';
import toast from 'react-hot-toast';

const TrafficContext = createContext();

export const useTraffic = () => useContext(TrafficContext);

export const TrafficProvider = ({ children }) => {
  const [stations, setStations] = useState([]);
  const [snapshotData, setSnapshotData] = useState({});
  const [loadingSnapshot, setLoadingSnapshot] = useState(false);
  const [activeHorizon, setActiveHorizon] = useState(null);
  const [currentStep, setCurrentStep] = useState(0); // For Live Simulation
  const [activeTab, setActiveTab] = useState('live'); // 'live' hoặc 'report'
  const [enableAlerts, setEnableAlerts] = useState(false); // Mặc định tắt để tránh phiền
  const [selectedStationId, setSelectedStationId] = useState(null); // Trạm được chọn từ tìm kiếm
  
  // Lưu trữ danh sách các trạm đã cảnh báo để tránh spam toast
  const alertedStationsRef = useRef(new Set());
  
  // Refs
  const markerRefs = useRef({});

  // --------------------------------------------------------
  // Virtual Clock Logic (Bắt đầu từ 08:00 AM, mỗi step +5 phút)
  // --------------------------------------------------------
  const getVirtualTime = () => {
    const startMinutes = 8 * 60; // 08:00 AM
    const totalMinutes = startMinutes + (currentStep * 5);
    const h = Math.floor(totalMinutes / 60) % 24;
    const m = totalMinutes % 60;
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}`;
  };

  // --------------------------------------------------------
  // Hệ thống Giám sát Tự động (Continuous Monitoring)
  // --------------------------------------------------------
  useEffect(() => {
    const DEMO_INTERVAL_MS = 10000; // Đổi thành 300000 khi chạy thực tế
    
    const interval = setInterval(() => {
      setCurrentStep(prev => prev + 1);
      // Reset danh sách cảnh báo khi chuyển sang mốc thời gian mới
      alertedStationsRef.current.clear();
    }, DEMO_INTERVAL_MS);
    
    return () => clearInterval(interval);
  }, []);

  // Khi currentStep thay đổi, tự động lấy dữ liệu mới (nếu đang bật horizon)
  useEffect(() => {
    if (activeHorizon) {
      // Dùng timestep = currentStep để lấy dữ liệu đúng thời điểm
      fetchSnapshot(activeHorizon, currentStep);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentStep]);

  // Fetch stations on mount
  useEffect(() => {
    const fetchStations = async () => {
      try {
        const res = await axios.get('/api/stations');
        setStations(res.data);
      } catch (error) {
        console.error('Lỗi khi tải danh sách trạm:', error);
      }
    };
    fetchStations();
  }, []);

  const fetchSnapshot = async (horizon, timestep = null) => {
    setLoadingSnapshot(true);
    setActiveHorizon(horizon);
    try {
      const url = timestep !== null 
        ? `/api/traffic/snapshot?horizon=${horizon}&timestep=${timestep}`
        : `/api/traffic/snapshot?horizon=${horizon}`;
      const res = await axios.get(url);
      
      const flowMap = {};
      res.data.forEach(item => { 
        flowMap[item.station_id] = item.predicted_flow; 
        
        // LOGIC CẢNH BÁO KẸT XE ĐỘT BIẾN (SPIKE DETECTION)
        const THRESHOLD = 500; // Ngưỡng kẹt xe
        if (enableAlerts && item.predicted_flow > THRESHOLD && !alertedStationsRef.current.has(item.station_id)) {
          const stationName = stations.find(s => s.id === item.station_id)?.name || `Trạm ${item.station_id}`;
          toast.error(`Phát hiện kẹt xe tại ${stationName}! Lưu lượng: ${Math.round(item.predicted_flow)} xe/5p`, {
            duration: 5000,
            icon: '🚨',
            style: {
              background: '#334155',
              color: '#fff',
              border: '1px solid #ef4444'
            },
          });
          alertedStationsRef.current.add(item.station_id);
        }
      });
      setSnapshotData(flowMap);
    } catch (error) {
      console.error('Lỗi khi gọi Snapshot API:', error);
    } finally {
      setLoadingSnapshot(false);
    }
  };

  const topCongested = Object.keys(snapshotData).length > 0 
    ? [...stations]
        .map(st => ({ ...st, predicted_flow: snapshotData[st.id] || 0 }))
        .sort((a, b) => b.predicted_flow - a.predicted_flow)
        .slice(0, 10)
    : [];

  return (
    <TrafficContext.Provider value={{
      stations,
      snapshotData,
      loadingSnapshot,
      activeHorizon,
      currentStep,
      activeTab,
      setActiveTab,
      getVirtualTime,
      fetchSnapshot,
      topCongested,
      markerRefs,
      enableAlerts,
      setEnableAlerts,
      selectedStationId,
      setSelectedStationId
    }}>
      {children}
    </TrafficContext.Provider>
  );
};
