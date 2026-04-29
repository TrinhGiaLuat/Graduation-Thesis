import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';
// Bắt buộc để Leaflet render đúng CSS map tiles
import 'leaflet/dist/leaflet.css';
import { Toaster } from 'react-hot-toast';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
    <Toaster position="top-right" />
  </React.StrictMode>
);
