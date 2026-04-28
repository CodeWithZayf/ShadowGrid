// src/components/shared/StatusBar.jsx
import React, { useState, useEffect } from 'react';
import { useApp } from '../../store/AppContext.jsx';

function Clock() {
  const [t, setT] = useState('');
  useEffect(() => {
    const tick = () =>
      setT(new Date().toLocaleTimeString('en-IN', { hour12: false }));
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);
  return <span className="clock">{t}</span>;
}

export default function StatusBar() {
  const { state, actions } = useApp();
  const { cdOnline, cdHealth, suspects, occurrences, alertQueue } = state;

  return (
    <header className="statusbar">
      <div className="statusbar-left">
        <span className="brand">
          <span className="brand-accent">Shadow</span>Grid
        </span>
        <span className="brand-sub">INTELLIGENCE PLATFORM</span>
      </div>

      <div className="statusbar-center">
        <div className="server-pill" title="CrimeDetector (port 8000)">
          <span className={`dot ${cdOnline ? 'dot-green' : 'dot-red'}`} />
          <span className="pill-label">SERVER</span>
          {cdOnline && cdHealth && (
            <span className="pill-stat">{cdHealth.cameras_registered ?? 0} CAM</span>
          )}
        </div>
      </div>

      <div className="statusbar-right">
        <div className="stat-badge stat-suspect">
          <span className="stat-n">{suspects.length}</span>
          <span className="stat-l">SUSPECTS</span>
        </div>
        <div className="stat-badge stat-occur">
          <span className="stat-n">{occurrences.length}</span>
          <span className="stat-l">OCCURRENCES</span>
        </div>
        <Clock />
      </div>

      {/* Alert toast strip */}
      {alertQueue.length > 0 && (
        <div className="alert-strip">
          {alertQueue.slice(0, 3).map(a => (
            <div key={a.id} className={`alert-toast alert-${a.level}`}>
              <span>{a.msg}</span>
              <button className="alert-close" onClick={() => actions.dismissAlert(a.id)}>✕</button>
            </div>
          ))}
        </div>
      )}
    </header>
  );
}
