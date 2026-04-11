import React from 'react';
import TrafficMap from './components/TrafficMap';

function App() {
  return (
    // Sử dụng chiều cao 100vh để chiếm bao phủ màn hình trình duyệt
    <div className="App" style={{ height: '100vh', width: '100vw', margin: 0, padding: 0 }}>
      <TrafficMap />
    </div>
  );
}

export default App;
