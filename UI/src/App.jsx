// src/App.jsx
import React, { useState } from 'react';
import { AppProvider } from './store/AppContext.jsx';
import StatusBar       from './components/shared/StatusBar.jsx';
import NavRail         from './components/shared/NavRail.jsx';
import MapView         from './components/map/MapView.jsx';
import MapSidebar      from './components/map/MapSidebar.jsx';
import SuspectsPanel   from './components/panels/SuspectsPanel.jsx';
import OccurrencesPanel from './components/panels/OccurrencesPanel.jsx';
import CamerasPanel    from './components/feeds/CamerasPanel.jsx';
import IntelligencePanel from './components/panels/IntelligencePanel.jsx';

function Content({ activeTab }) {
  return (
    <div className="content-area">
      {activeTab === 'map' && (
        <div className="map-layout">
          <MapView />
          <MapSidebar />
        </div>
      )}
      {activeTab === 'suspects'     && <SuspectsPanel />}
      {activeTab === 'occurrences'  && <OccurrencesPanel />}
      {activeTab === 'cameras'      && <CamerasPanel />}
      {activeTab === 'intelligence' && <IntelligencePanel />}
    </div>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState('map');

  return (
    <AppProvider>
      <div className="app-shell">
        <StatusBar />
        <div className="app-body">
          <NavRail activeTab={activeTab} onTabChange={setActiveTab} />
          <Content activeTab={activeTab} />
        </div>
      </div>
    </AppProvider>
  );
}
