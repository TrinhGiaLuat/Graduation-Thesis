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
  
  // Trọng tâm quy chiếu khu vực Los Angeles (Mô hình METR-LA)
  const laCenter = [34.0522, -118.2437];

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
    if (trafficData[stationId]) return;
    setLoadingTraffic(true);
    try {
      const response = await axios.get(`/api/traffic/${stationId}`);
      const rawData = response.data; // 24 phần tử: 12 lịch sử + 12 dự báo

      /**
       * Xây dựng mảng 25 điểm liên tục ghép 2 đường liền mạch:
       * - Điểm 0 -> 11: Lịch sử (-55p đến -5p dùng field `history`)
       * - Điểm 12 (HIỆN TẠI): Cả `history` và `forecast` đều có giá trị → 2 đường nối nhau
       * - Điểm 13 -> 24: Dự báo (+5p đến +60p dùng field `forecast`)
       */
      const historyPoints = rawData.filter(r => !r.is_prediction);  // 12 điểm
      const forecastPoints = rawData.filter(r => r.is_prediction);   // 12 điểm

      // Giá trị tại điểm giao (HIỆN TẠI): lấy avg_speed của điểm cuối lịch sử
      const bridgeValue = historyPoints.length > 0
        ? historyPoints[historyPoints.length - 1].avg_speed
        : null;

      const chartData = [
        // 11 điểm lịch sử đầu tiên (-55p đến -5p)
        ...historyPoints.slice(0, -1).map((r, i) => ({
          label: `-${(12 - 1 - i) * 5}p`,
          history: r.avg_speed,
          forecast: null,
          is_prediction: false,
        })),
        // Điểm giao HIỆN TẠI: cả 2 giá trị đều có → 2 đường nối liền mạch
        {
          label: 'HIỆN TẠI',
          history: bridgeValue,
          forecast: bridgeValue,
          is_prediction: false,
        },
        // 12 điểm dự báo (+5p đến +60p)
        ...forecastPoints.map((r, i) => ({
          label: `+${(i + 1) * 5}p`,
          history: null,
          forecast: r.avg_speed,
          is_prediction: true,
        })),
      ];

      setTrafficData(prev => ({ ...prev, [stationId]: chartData }));
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
      center={laCenter} 
      zoom={11} 
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
                    📈 Tốc độ giao thông 24h (mph)
                    <span style={{ marginLeft: '8px', fontSize: '10px', fontWeight: 'normal', color: '#9ca3af' }}>
                      (12h lịch sử + 12h dự báo GNN)
                    </span>
                  </p>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={trafficData[station.id]}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb" />

                      {/* Trục X: Nhãn thời gian tương đối -55p → HIỆN TẠI → +60p */}
                      <XAxis
                        dataKey="label"
                        fontSize={9}
                        tickMargin={4}
                        stroke="#6b7280"
                        interval={0}
                        tick={({ x, y, payload }) => {
                          const isCurrent = payload.value === 'HIỆN TẠI';
                          return (
                            <text
                              key={`tick-${payload.value}-${x}`}
                              x={x} y={y + 10}
                              textAnchor="middle"
                              fontSize={isCurrent ? 9 : 8}
                              fontWeight={isCurrent ? 'bold' : 'normal'}
                              fill={isCurrent ? '#ef4444' : '#6b7280'}
                            >
                              {isCurrent ? '▼' : payload.value}
                            </text>
                          );
                        }}
                      />

                      <YAxis
                        fontSize={10}
                        width={38}
                        stroke="#6b7280"
                        domain={[0, 80]}
                        label={{ value: 'mph', angle: -90, position: 'insideLeft', offset: 10, style: { fontSize: 9, fill: '#9ca3af' } }}
                      />

                      {/* Custom Tooltip phân biệt Lịch sử / Dự báo */}
                      <RechartsTooltip
                        contentStyle={{ fontSize: '11px', borderRadius: '6px', border: '1px solid #e5e7eb', background: '#fff' }}
                        labelStyle={{ fontWeight: 'bold', color: '#374151', marginBottom: '4px' }}
                        formatter={(value, name, props) => {
                          if (value === null || value === undefined) return null;
                          const label = props.payload?.label || '';
                          const isPred = props.payload?.is_prediction;
                          const prefix = isPred ? `Dự báo GNN (${label})` : `Lịch sử (${label})`;
                          return [`${Number(value).toFixed(1)} mph`, prefix];
                        }}
                      />

                      {/* Đường Lịch sử - xanh dương, nét liền */}
                      <Line
                        type="monotone"
                        dataKey="history"
                        stroke="#3b82f6"
                        strokeWidth={2.5}
                        dot={(props) => {
                          if (props.payload?.label === 'HIỆN TẠI') return null;
                          return <circle key={`dot-hist-${props.index}`} cx={props.cx} cy={props.cy} r={2} fill="#3b82f6" strokeWidth={0} />;
                        }}
                        activeDot={{ r: 5, strokeWidth: 0 }}
                        name="Lịch sử"
                        connectNulls={false}
                      />

                      {/* Đường Dự báo GNN - cam, nét đứt */}
                      <Line
                        type="monotone"
                        dataKey="forecast"
                        stroke="#f97316"
                        strokeWidth={2.5}
                        strokeDasharray="5 3"
                        dot={(props) => {
                          if (props.payload?.label === 'HIỆN TẠI') return null;
                          return <circle key={`dot-pred-${props.index}`} cx={props.cx} cy={props.cy} r={2} fill="#f97316" strokeWidth={0} />;
                        }}
                        activeDot={{ r: 5, strokeWidth: 0 }}
                        name="Dự báo GNN"
                        connectNulls={false}
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
