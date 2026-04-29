import React from 'react';
import Header from './components/layout/Header';
import Sidebar from './components/layout/Sidebar';
import RightPanel from './components/layout/RightPanel';
import ReportPanel from './components/layout/ReportPanel';
import HistoryPanel from './components/layout/HistoryPanel';
import TrafficMap from './components/TrafficMap';
import LoginPage from './components/auth/LoginPage';
import { TrafficProvider, useTraffic } from './context/TrafficContext';

const MainLayout = ({ onLogout }) => {
  const { activeTab } = useTraffic();

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-slate-900 text-slate-50 font-sans">
      <Header onLogout={onLogout} />
      
      <div className="flex flex-1 overflow-hidden">
        <Sidebar />
        
        {activeTab === 'live' ? (
          <>
            <main className="flex-1 relative">
              <TrafficMap />
            </main>
            <RightPanel />
          </>
        ) : activeTab === 'report' ? (
          <ReportPanel />
        ) : (
          <HistoryPanel />
        )}
      </div>
    </div>
  );
};

function App() {
  const [isAuthenticated, setIsAuthenticated] = React.useState(false);

  const handleLogin = () => {
    setIsAuthenticated(true);
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
  };

  if (!isAuthenticated) {
    return <LoginPage onLogin={handleLogin} />;
  }

  return (
    <TrafficProvider>
      <MainLayout onLogout={handleLogout} />
    </TrafficProvider>
  );
}

export default App;
