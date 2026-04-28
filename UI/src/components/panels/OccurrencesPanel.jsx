// src/components/panels/OccurrencesPanel.jsx
import React, { useState } from 'react';
import { useApp } from '../../store/AppContext.jsx';
import { shortId, formatTs, formatAge } from '../../utils/format.js';

export default function OccurrencesPanel() {
  const { state, actions } = useApp();
  const { occurrences, suspects } = state;
  const [filter, setFilter]   = useState('');
  const [filterType, setFilterType] = useState('all'); // 'all' | 'suspect' | 'camera'

  const filtered = occurrences.filter(o => {
    if (!filter) return true;
    if (filterType === 'suspect') return o.suspect_id.includes(filter);
    if (filterType === 'camera')  return o.camera_id.includes(filter);
    return o.suspect_id.includes(filter) || o.camera_id.includes(filter) || o.occurrence_id.includes(filter);
  });

  function simColor(sim) {
    if (sim >= 0.9) return '#4ade80';
    if (sim >= 0.8) return '#facc15';
    return '#fb923c';
  }

  return (
    <div className="panel">
      <div className="panel-toolbar">
        <h2 className="panel-title">OCCURRENCE LOG</h2>
        <div className="panel-tools">
          <select className="form-select small" value={filterType} onChange={e => setFilterType(e.target.value)}>
            <option value="all">All fields</option>
            <option value="suspect">Suspect ID</option>
            <option value="camera">Camera ID</option>
          </select>
          <input
            className="search-input"
            placeholder="Filter…"
            value={filter}
            onChange={e => setFilter(e.target.value)}
          />
        </div>
      </div>

      <div className="timeline-summary">
        <div className="ts-chip">
          <span className="ts-n">{occurrences.length}</span>
          <span className="ts-l">TOTAL</span>
        </div>
        <div className="ts-chip">
          <span className="ts-n">
            {occurrences.filter(o => Date.now() / 1000 - o.timestamp < 300).length}
          </span>
          <span className="ts-l">LAST 5 MIN</span>
        </div>
        <div className="ts-chip">
          <span className="ts-n">
            {new Set(occurrences.map(o => o.suspect_id)).size}
          </span>
          <span className="ts-l">SUSPECTS</span>
        </div>
        <div className="ts-chip">
          <span className="ts-n">
            {new Set(occurrences.map(o => o.camera_id)).size}
          </span>
          <span className="ts-l">CAMERAS</span>
        </div>
      </div>

      <div className="table-wrap">
        <div className="table-stat">Showing {filtered.length} of {occurrences.length}</div>
        <table className="data-table">
          <thead>
            <tr>
              <th>OCCURRENCE ID</th>
              <th>SUSPECT ID</th>
              <th>CAMERA</th>
              <th>CAR ID</th>
              <th>SIMILARITY</th>
              <th>TIMESTAMP</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map(o => {
              const sim = o.similarity ?? 0;
              const sc = simColor(sim);
              return (
                <tr key={o.occurrence_id}>
                  <td><span className="id-cell" title={o.occurrence_id}>{shortId(o.occurrence_id)}</span></td>
                  <td>
                    <span
                      className="id-cell link-cell"
                      title={o.suspect_id}
                      onClick={() => actions.selectSuspect(o.suspect_id)}
                    >
                      {shortId(o.suspect_id)}
                    </span>
                  </td>
                  <td className="mono-cell">{o.camera_id}</td>
                  <td className="mono-cell">{o.car_id}</td>
                  <td>
                    <div className="sim-wrap">
                      <div className="sim-bar-bg">
                        <div className="sim-bar-fill" style={{ width: `${(sim * 100).toFixed(0)}%`, background: sc }} />
                      </div>
                      <span className="sim-val" style={{ color: sc }}>{(sim * 100).toFixed(1)}%</span>
                    </div>
                  </td>
                  <td className="mono-cell ts-cell">
                    <div>{formatTs(o.timestamp)}</div>
                    <div className="age-hint">{formatAge(o.timestamp)}</div>
                  </td>
                </tr>
              );
            })}
            {filtered.length === 0 && (
              <tr><td colSpan={6} className="empty-row">No occurrences recorded yet</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
