import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline } from 'react-leaflet';
import axios from 'axios';
import L from 'leaflet';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer } from 'recharts';

// Xử lý lỗi nạp icon mặc định của cấu trúc Webpack/Leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
  iconUrl: require('leaflet/dist/images/marker-icon.png'),
  shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

const TrafficMap = () => {
  const [stations, setStations] = useState([]);
  const [edges, setEdges] = useState([]);
  const [trafficData, setTrafficData] = useState({});
  const [loadingTraffic, setLoadingTraffic] = useState(false);
  
  // Trọng tâm khu vực nội thành Hà Nội mặc định
  const hanoiCenter = [21.0285, 105.8048];

  useEffect(() => {
    // Tải song song danh sách trạm (Node) và mạng lưới đường liên kết (Edge)
    const fetchData = async () => {
      try {
        const [stationsRes, edgesRes] = await Promise.all([
          axios.get('/api/stations'),
          axios.get('/api/graph')
        ]);
        setStations(stationsRes.data);
        setEdges(edgesRes.data);
      } catch (error) {
        console.error("Lỗi khi tải dữ liệu bộ xương đồ thị bản đồ:", error);
      }
    };
    fetchData();
  }, []);

  // Hàm xử lý Click Marker: Kéo dữ liệu API Time-series theo Station ID
  const handleMarkerClick = async (stationId) => {
    // Tối ưu hoá: Thu hồi lưu lượng Cache nếu đã call rồi
    if (trafficData[stationId]) return; 
    
    setLoadingTraffic(true);
    try {
      const response = await axios.get(`/api/traffic/${stationId}`);
      // Chuẩn hoá Timestamp (chỉ trích xuất số nguyên giờ) để hiển thị mượt trên LineChart
      const formattedData = response.data.map(record => {
        const date = new Date(record.timestamp);
        return {
          ...record,
          timeLabel: `${date.getHours()}h`
        };
      });
      
      setTrafficData(prev => ({
        ...prev,
        [stationId]: formattedData
      }));
    } catch (error) {
      console.error(`Lỗi tải dữ liệu cho trạm ${stationId}:`, error);
    } finally {
      setLoadingTraffic(false);
    }
  };

  // Kiến tạo mảng các đoạn thẳng (Polyline) nối hai trạm thông qua List <Edges>
  const renderPolylines = () => {
    if (!stations.length || !edges.length) return null;
    
    // Xây dựng hệ toạ độ Map cấp tốc (Hashmap) Station ID -> [Lat, Lng]
    const stationCoords = {};
    stations.forEach(st => {
      stationCoords[st.id] = [st.lat, st.lng];
    });

    return edges.map(edge => {
      const source = stationCoords[edge.source_station_id];
      const target = stationCoords[edge.target_station_id];
      
      if (source && target) {
        return (
          <Polyline 
            key={edge.id} 
            positions={[source, target]} 
            pathOptions={{ color: '#ff7800', weight: 4, opacity: 0.7 }} 
          />
        );
      }
      return null;
    });
  };

  return (
    <MapContainer 
      center={hanoiCenter} 
      zoom={14} 
      style={{ height: '100%', width: '100%' }}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      
      {/* 1. Lớp Layer Vẽ mạng lưới giao thông (Graph Edges) */}
      {renderPolylines()}
      
      {/* 2. Lớp Layer Vẽ Nút Giao / Trạm Giao Thông */}
      {stations.map(station => (
        <Marker 
          key={station.id} 
          position={[station.lat, station.lng]}
          eventHandlers={{
            click: () => handleMarkerClick(station.id),
          }}
        >
          <Popup maxWidth={450} minWidth={350}>
             <div style={{ padding: '8px', fontFamily: 'sans-serif' }}>
              <h3 style={{ margin: '0 0 10px 0', fontSize: '15px', color: '#1f2937' }}>
                📍 {station.name} ({station.station_id_str})
              </h3>
              
              {loadingTraffic && !trafficData[station.id] ? (
                <div style={{ textAlign: 'center', padding: '20px 0', color: '#6b7280' }}>
                   Đang tải tiến trình lưu lượng xe...
                </div>
              ) : trafficData[station.id] ? (
                <div style={{ width: '100%', height: 220, marginTop: '10px' }}>
                  <p style={{ margin: '0 0 8px 0', fontSize: '12px', fontWeight: 'bold', color: '#4b5563' }}>
                    📈 Lưu lượng 24h qua (xe/giờ)
                  </p>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={trafficData[station.id]}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb" />
                      <XAxis dataKey="timeLabel" fontSize={10} tickMargin={5} stroke="#6b7280" />
                      <YAxis fontSize={10} width={35} stroke="#6b7280" />
                      <RechartsTooltip 
                        contentStyle={{ fontSize: '11px', borderRadius: '4px', border: '1px solid #e5e7eb' }}
                        labelStyle={{ fontWeight: 'bold', color: '#374151' }}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="volume" 
                        stroke="#f97316" 
                        strokeWidth={3}
                        dot={{ r: 2, fill: '#f97316', strokeWidth: 0 }} 
                        activeDot={{ r: 5, strokeWidth: 0 }} 
                        name="Lưu thông (Xe)"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <p style={{ fontSize: '12px', color: '#9ca3af' }}>Hệ thống không thu thập được dữ liệu 24h.</p>
              )}
            </div>
          </Popup>
        </Marker>
      ))}
    </MapContainer>
  );
};

export default TrafficMap;
