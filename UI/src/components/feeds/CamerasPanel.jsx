// src/components/feeds/CamerasPanel.jsx
import React, { useState } from 'react';
import { useApp } from '../../store/AppContext.jsx';
import { CrimeDetectorAPI } from '../../api/crimedetector.js';

function LiveFeed({ cam, health, onClick }) {
  const streamUrl = CrimeDetectorAPI.streamUrl(cam.id);
  const liveStatus = health?.cameras_live?.[cam.id];
  const queueDepth = health?.queues?.[cam.id] ?? 0;
  const isLive = liveStatus !== false;

  return (
    <div className={`feed-card ${isLive ? 'feed-live' : 'feed-dead'}`} onClick={onClick}>
      <div className="feed-header">
        <span className="feed-id">{cam.id}</span>
        <span className={`feed-dot ${isLive ? 'dot-green' : 'dot-red'}`} />
        <span className="feed-status">{isLive ? 'LIVE' : 'OFFLINE'}</span>
      </div>
      <div className="feed-viewport">
        {isLive ? (
          <img
            src={streamUrl}
            alt={`Camera ${cam.id}`}
            className="feed-img"
            onError={e => { e.target.style.display = 'none'; }}
          />
        ) : (
          <div className="feed-offline">
            <span className="offline-icon">📷</span>
            <span>No signal</span>
          </div>
        )}
      </div>
      <div className="feed-footer">
        <span className="feed-location">{cam.metadata?.location_name}</span>
        <div className="queue-bar-wrap">
          <div className="queue-bar-fill" style={{ width: `${(queueDepth / 30) * 100}%` }} />
        </div>
        <span className="queue-depth">{queueDepth}/30</span>
      </div>
    </div>
  );
}

export default function CamerasPanel() {
  const { state, actions } = useApp();
  const { cameras, cdHealth } = state;
  const [fullscreen, setFullscreen] = useState(null); // camera id
  const [cols, setCols] = useState(2);               // grid columns

  const fullCam = cameras.find(c => c.id === fullscreen);

  async function handleClearSuspects(camId) {
    try {
      const res = await CrimeDetectorAPI.removeCameraSuspects(camId);
      actions.pushAlert(`Removed ${res.suspects_removed} suspect(s) from ${camId}`, 'success');
    } catch (e) {
      actions.pushAlert(e.message, 'error');
    }
  }

  return (
    <div className="panel">
      <div className="panel-toolbar">
        <h2 className="panel-title">CAMERA NETWORK</h2>
        <div className="panel-tools">
          <span className="tool-label">GRID</span>
          {[1, 2, 3, 4].map(n => (
            <button
              key={n}
              className={`grid-btn ${cols === n ? 'grid-btn-active' : ''}`}
              onClick={() => setCols(n)}
            >{n}×</button>
          ))}
        </div>
      </div>

      {/* Feed grid */}
      <div className="feed-grid" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
        {cameras.map(cam => (
          <LiveFeed
            key={cam.id}
            cam={cam}
            health={cdHealth}
            onClick={() => setFullscreen(cam.id)}
          />
        ))}
        {cameras.length === 0 && (
          <div className="empty-note" style={{ gridColumn: '1 / -1' }}>
            No cameras registered. Waiting for CrimeDetector connection…
          </div>
        )}
      </div>

      {/* Fullscreen preview modal */}
      {fullCam && (
        <div className="modal-overlay" onClick={() => setFullscreen(null)}>
          <div className="feed-modal" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <div>
                <span className="feed-id">{fullCam.id}</span>
                <span className="feed-modal-loc">{fullCam.metadata?.location_name}</span>
                <span className="feed-modal-coord">
                  {fullCam.metadata?.latitude?.toFixed(5)}, {fullCam.metadata?.longitude?.toFixed(5)}
                </span>
              </div>
              <div className="feed-modal-actions">
                <button
                  className="btn btn-danger"
                  onClick={() => handleClearSuspects(fullCam.id)}
                >
                  ✕ CLEAR SUSPECTS
                </button>
                <button className="close-btn" onClick={() => setFullscreen(null)}>✕</button>
              </div>
            </div>
            <div className="feed-modal-body">
              <img
                src={CrimeDetectorAPI.streamUrl(fullCam.id)}
                alt={fullCam.id}
                className="feed-modal-img"
              />
            </div>
            <div className="feed-modal-footer">
              <div className="fmf-row">
                <span className="fmf-lbl">URL</span>
                <span className="fmf-val">{fullCam.url}</span>
              </div>
              <div className="fmf-row">
                <span className="fmf-lbl">QUEUE</span>
                <span className="fmf-val">{cdHealth?.queues?.[fullCam.id] ?? '—'} / 30 frames</span>
              </div>
              <div className="fmf-row">
                <span className="fmf-lbl">STATUS</span>
                <span className={`fmf-val ${cdHealth?.cameras_live?.[fullCam.id] ? 'col-green' : 'col-red'}`}>
                  {cdHealth?.cameras_live?.[fullCam.id] ? '● LIVE' : '○ OFFLINE'}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
