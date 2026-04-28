// src/components/panels/IntelligencePanel.jsx
// Placeholder — will be rebuilt in Phase 3 with Python prediction engine data
// (uncertainty radius, camera graph predictions, deployment orders).
import React from 'react';

export default function IntelligencePanel() {
  return (
    <div className="panel">
      <div className="panel-toolbar">
        <h2 className="panel-title">INTELLIGENCE</h2>
      </div>
      <div className="intel-body">
        <div className="empty-state-large">
          <span className="es-icon-large">⬘</span>
          <span>
            Intelligence engine not yet connected.<br />
            Prediction and uncertainty features will be available after Phase 3 integration.
          </span>
        </div>
      </div>
    </div>
  );
}
