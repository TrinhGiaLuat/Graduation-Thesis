import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
// Bắt buộc để Leaflet render đúng CSS map tiles
import 'leaflet/dist/leaflet.css';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
