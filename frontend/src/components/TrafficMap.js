import React, { useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import axios from 'axios';
import L from 'leaflet';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip as RechartsTooltip, ResponsiveContainer
} from 'recharts';
import { useTraffic } from '../context/TrafficContext';

// Xử lý lỗi nạp icon mặc định của cấu trúc Webpack/Leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
  iconUrl:       require('leaflet/dist/images/marker-icon.png'),
  shadowUrl:     require('leaflet/dist/images/marker-shadow.png'),
});

const MAX_FLOW = 523;

const createColoredIcon = (color, flow = null) => {
  const label = flow !== null ? `<span style="
    position:absolute; top:-18px; left:50%; transform:translateX(-50%);
    background:rgba(0,0,0,0.65); color:#fff; border-radius:4px;
    padding:1px 4px; font-size:9px; white-space:nowrap; pointer-events:none;
  ">${Math.round(flow)}</span>` : '';

  return L.divIcon({
    className: '',
    html: `<div style="position:relative;">
      ${label}
      <div style="
        width:14px; height:14px; border-radius:50%;
        background:${color}; border:2px solid rgba(255,255,255,0.9);
        box-shadow:0 0 6px rgba(0,0,0,0.4);
      "></div>
    </div>`,
    iconSize:   [14, 14],
    iconAnchor: [7, 7],
  });
};

const getMarkerColor = (flow) => {
  if (flow < 0.3 * MAX_FLOW) return '#22c55e';
  if (flow <= 0.7 * MAX_FLOW) return '#f59e0b';
  return '#ef4444';
};

const horizonToLabel = (horizon) => {
  if (!horizon) return '';
  return `t+${horizon * 5} phút`;
};

const MapController = ({ selectedStationId, stations }) => {
  const map = useMap();
  React.useEffect(() => {
    if (selectedStationId) {
      const st = stations.find(s => s.id === selectedStationId);
      if (st) {
        map.flyTo([st.lat, st.lng], 16, { duration: 1.5 });
      }
    }
  }, [selectedStationId, stations, map]);
  return null;
};

const TrafficMap = () => {
  const { 
    stations, snapshotData, activeHorizon, 
    currentStep, markerRefs, selectedStationId
  } = useTraffic();

  const [trafficData, setTrafficData] = useState({});
  const [loadingTraffic, setLoadingTraffic] = useState(false);

  const sfCenter = [37.7749, -122.4194];

  // Lắng nghe sự thay đổi của snapshotData từ Context để cập nhật icon Marker
  React.useEffect(() => {
    Object.keys(markerRefs.current).forEach(stationIdStr => {
      const stId   = parseInt(stationIdStr);
      const marker = markerRefs.current[stationIdStr];
      const flow   = snapshotData[stId];
      if (marker && flow !== undefined) {
        marker.setIcon(createColoredIcon(getMarkerColor(flow), flow));
      }
    });
  }, [snapshotData, markerRefs]);

  // ---------------------------------------------------------------------------
  // Click vào Marker: Hiện biểu đồ lịch sử + dự báo chi tiết của trạm đó
  // ---------------------------------------------------------------------------
  const handleMarkerClick = async (stationId) => {
    setLoadingTraffic(true);
    try {
      const response = await axios.get(`/api/traffic/${stationId}?timestep=${currentStep}`);
      const rawData = response.data;

      const historyPoints = rawData.filter(r => !r.is_prediction);
      const forecastPoints = rawData.filter(r => r.is_prediction);
      const bridgeValue = historyPoints.length > 0
        ? historyPoints[historyPoints.length - 1].volume : null;

      const chartData = [
        ...historyPoints.slice(0, -1).map((r, i) => ({
          label: `-${(12 - 1 - i) * 5}p`, history: r.volume, forecast: null, is_prediction: false,
        })),
        { label: 'HIỆN TẠI', history: bridgeValue, forecast: bridgeValue, is_prediction: false },
        ...forecastPoints.map((r, i) => ({
          label: `+${(i + 1) * 5}p`, history: null, forecast: r.volume, is_prediction: true,
        })),
      ];
      setTrafficData(prev => ({ ...prev, [stationId]: chartData }));
    } catch (error) {
      console.error(`Lỗi tải dữ liệu cho trạm ${stationId}:`, error);
    } finally {
      setLoadingTraffic(false);
    }
  };

  return (
    <div className="h-full w-full z-0">
      <MapContainer
        center={sfCenter}
        zoom={12}
        className="h-full w-full"
        zoomControl={false}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <MapController selectedStationId={selectedStationId} stations={stations} />

        {/* 1. Lớp Layer Vẽ mạng lưới giao thông (Graph Edges) - Tạm ẩn để giao diện thoáng hơn */}
        {/* {renderPolylines()} */}

        {/* 2. Lớp Layer Vẽ Marker cho từng trạm với màu sắc động */}
        {stations.map(station => {
          // Xác định màu icon: Nếu đã có snapshot thì dùng màu động, chưa có dùng xanh mặc định
          const flow      = snapshotData[station.id];
          const iconColor = flow !== undefined ? getMarkerColor(flow) : '#3b82f6';
          const icon      = createColoredIcon(iconColor, flow !== undefined ? flow : null);

          return (
            <Marker
              key={station.id}
              position={[station.lat, station.lng]}
              icon={icon}
              ref={el => { if (el) markerRefs.current[station.id] = el; }}
              eventHandlers={{ click: () => handleMarkerClick(station.id) }}
            >
              <Popup maxWidth={450} minWidth={350}>
                <div style={{ padding: '8px', fontFamily: 'sans-serif' }}>
                  <h3 style={{ margin: '0 0 10px 0', fontSize: '15px', color: '#1f2937' }}>
                    📍 {station.name} (Sensor ID: {station.station_id_str})
                  </h3>

                  {/* Hiển thị lưu lượng dự báo snapshot nếu có */}
                  {flow !== undefined && (
                    <div style={{
                      display: 'inline-flex', alignItems: 'center', gap: '6px',
                      background: getMarkerColor(flow) + '22',
                      border: `1px solid ${getMarkerColor(flow)}`,
                      borderRadius: '6px', padding: '4px 10px', marginBottom: '10px',
                    }}>
                      <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: getMarkerColor(flow) }} />
                      <span style={{ fontSize: '12px', fontWeight: '600', color: '#1f2937' }}>
                        Dự báo tại {horizonToLabel(activeHorizon)}: {Math.round(flow)} xe/5 phút
                      </span>
                    </div>
                  )}

                  {/* Biểu đồ lịch sử + dự báo chi tiết */}
                  {loadingTraffic && !trafficData[station.id] ? (
                    <div style={{ textAlign: 'center', padding: '20px 0', color: '#6b7280' }}>
                      ⏳ Đang tính toán lưu lượng AI...
                    </div>
                  ) : trafficData[station.id] ? (
                    <div style={{ width: '100%', height: 220, marginTop: '10px' }}>
                      <p style={{ margin: '0 0 8px 0', fontSize: '12px', fontWeight: 'bold', color: '#4b5563' }}>
                        📈 Lưu lượng giao thông 1h tới (Số xe)
                        <span style={{ marginLeft: '8px', fontSize: '10px', fontWeight: 'normal', color: '#9ca3af' }}>
                          (Dự báo đa tầm nhìn GNN)
                        </span>
                      </p>
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={trafficData[station.id]}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb" />
                          <XAxis
                            dataKey="label" fontSize={9} tickMargin={4} stroke="#6b7280" interval={0}
                            tick={({ x, y, payload }) => {
                              const isCurrent = payload.value === 'HIỆN TẠI';
                              return (
                                <text key={`tick-${payload.value}-${x}`}
                                  x={x} y={y + 10} textAnchor="middle"
                                  fontSize={isCurrent ? 9 : 8}
                                  fontWeight={isCurrent ? 'bold' : 'normal'}
                                  fill={isCurrent ? '#ef4444' : '#6b7280'}
                                >
                                  {isCurrent ? '▼' : payload.value}
                                </text>
                              );
                            }}
                          />
                          <YAxis fontSize={10} width={38} stroke="#6b7280"
                            label={{ value: 'Số xe', angle: -90, position: 'insideLeft', offset: 10, style: { fontSize: 9, fill: '#9ca3af' } }}
                          />
                          <RechartsTooltip
                            contentStyle={{ fontSize: '11px', borderRadius: '6px', border: '1px solid #e5e7eb', background: '#fff' }}
                            labelStyle={{ fontWeight: 'bold', color: '#374151', marginBottom: '4px' }}
                            formatter={(value, name, props) => {
                              if (value === null || value === undefined) return null;
                              const label  = props.payload?.label || '';
                              const isPred = props.payload?.is_prediction;
                              const prefix = isPred ? `Dự báo (${label})` : `Thực tế (${label})`;
                              return [`${Number(value).toFixed(0)} xe`, prefix];
                            }}
                          />
                          <Line type="monotone" dataKey="history" stroke="#3b82f6" strokeWidth={2.5}
                            dot={(props) => {
                              if (props.payload?.label === 'HIỆN TẠI') return null;
                              return <circle key={`dot-hist-${props.index}`} cx={props.cx} cy={props.cy} r={2} fill="#3b82f6" strokeWidth={0} />;
                            }}
                            activeDot={{ r: 5, strokeWidth: 0 }} name="Lịch sử" connectNulls={false}
                          />
                          <Line type="monotone" dataKey="forecast" stroke="#f97316" strokeWidth={2.5}
                            strokeDasharray="5 3"
                            dot={(props) => {
                              if (props.payload?.label === 'HIỆN TẠI') return null;
                              return <circle key={`dot-pred-${props.index}`} cx={props.cx} cy={props.cy} r={2} fill="#f97316" strokeWidth={0} />;
                            }}
                            activeDot={{ r: 5, strokeWidth: 0 }} name="Dự báo GNN" connectNulls={false}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  ) : (
                    <p style={{ fontSize: '12px', color: '#9ca3af' }}>
                      Click để xem biểu đồ dự báo chi tiết.
                    </p>
                  )}
                </div>
              </Popup>
            </Marker>
          );
        })}
      </MapContainer>
    </div>
  );
};

export default TrafficMap;
