// src/components/map/MapSidebar.jsx
import React from 'react';
import { useApp } from '../../store/AppContext.jsx';
import { shortId, formatAge, suspectColor } from '../../utils/format.js';

export default function MapSidebar() {
  const { state, actions } = useApp();
  const { suspects, selectedSuspectId } = state;

  return (
    <div className="map-sidebar">
      <div className="sidebar-tabs">
        <button className="stab stab-active">SUSPECTS</button>
      </div>

      <div className="sidebar-body">
        <div className="sidebar-section">
          <div className="sidebar-count">{suspects.length} SUSPECT(S)</div>
          {suspects.map((s, idx) => {
            const color = suspectColor(idx);
            const isSelected = s.suspect_id === selectedSuspectId;
            return (
              <div
                key={s.suspect_id}
                className={`suspect-card ${isSelected ? 'suspect-card-selected' : ''}`}
                style={{ borderLeftColor: color }}
                onClick={() => actions.selectSuspect(isSelected ? null : s.suspect_id)}
              >
                <div className="sc-header">
                  <span className="sc-num" style={{ background: color }}>{idx + 1}</span>
                  <span className="sc-id">{shortId(s.suspect_id)}</span>
                  <span className="sc-age">{formatAge(s.timestamp)}</span>
                </div>
                <div className="sc-row">
                  <span className="sc-lbl">CAMERA</span>
                  <span>{s.camera_id ?? '—'}</span>
                </div>
                <div className="sc-row">
                  <span className="sc-lbl">SOURCE</span>
                  <span>{s.source ?? '—'}</span>
                </div>
                {s.description && (
                  <div className="sc-row">
                    <span className="sc-lbl">DESC</span>
                    <span>{s.description}</span>
                  </div>
                )}
              </div>
            );
          })}
          {suspects.length === 0 && <div className="empty-note">No active suspects</div>}
        </div>
      </div>
    </div>
  );
}
