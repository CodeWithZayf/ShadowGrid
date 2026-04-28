// src/components/panels/SuspectsPanel.jsx
import React, { useState, useRef } from 'react';
import { useApp } from '../../store/AppContext.jsx';
import { shortId, formatTs, formatAge, sourceBadge } from '../../utils/format.js';
import { CrimeDetectorAPI } from '../../api/crimedetector.js';
import EmbeddingChart from './EmbeddingChart.jsx';

export default function SuspectsPanel() {
  const { state, actions } = useApp();
  const { suspects, cameras } = state;

  const [showAdd, setShowAdd]       = useState(false);
  const [addMode, setAddMode]       = useState('image'); // 'image' | 'description'
  const [preview, setPreview]       = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [viewEmbed, setViewEmbed]   = useState(null);   // suspect to show embedding for
  const [filter, setFilter]         = useState('');

  const fileRef      = useRef();
  const descRef      = useRef();
  const camRef       = useRef();
  const imgDescRef   = useRef();

  const filtered = filter
    ? suspects.filter(s =>
        s.suspect_id.includes(filter) ||
        s.camera_id.includes(filter) ||
        (s.description || '').toLowerCase().includes(filter.toLowerCase())
      )
    : suspects;

  async function handleSubmit() {
    setSubmitting(true);
    try {
      if (addMode === 'image') {
        const file = fileRef.current?.files?.[0];
        if (!file) { actions.pushAlert('Select an image', 'error'); return; }
        await CrimeDetectorAPI.addSuspectImage(file, camRef.current?.value, imgDescRef.current?.value);
      } else {
        const desc = descRef.current?.value?.trim();
        if (!desc) { actions.pushAlert('Enter a description', 'error'); return; }
        await CrimeDetectorAPI.addSuspectDescription(desc, camRef.current?.value);
      }
      setShowAdd(false);
      setPreview(null);
      actions.pushAlert('Suspect added', 'success');
    } catch (e) {
      actions.pushAlert(e.message, 'error');
    } finally {
      setSubmitting(false);
    }
  }

  function handleFileChange(e) {
    const f = e.target.files?.[0];
    if (f) setPreview(URL.createObjectURL(f));
  }

  // Match lookup — will be populated in Phase 3 by the Python prediction engine
  const matchMap = {};

  return (
    <div className="panel">
      <div className="panel-toolbar">
        <h2 className="panel-title">SUSPECTS REGISTRY</h2>
        <div className="panel-tools">
          <input
            className="search-input"
            placeholder="Filter by ID / camera / description…"
            value={filter}
            onChange={e => setFilter(e.target.value)}
          />
          <button className="btn btn-accent" onClick={() => setShowAdd(v => !v)}>
            {showAdd ? '✕ CANCEL' : '+ ADD SUSPECT'}
          </button>
        </div>
      </div>

      {/* Add suspect form */}
      {showAdd && (
        <div className="add-form">
          <div className="add-form-tabs">
            <button className={`atab ${addMode === 'image' ? 'atab-active' : ''}`} onClick={() => setAddMode('image')}>FROM IMAGE</button>
            <button className={`atab ${addMode === 'description' ? 'atab-active' : ''}`} onClick={() => setAddMode('description')}>FROM DESCRIPTION</button>
          </div>
          <div className="add-form-body">
            {addMode === 'image' && (
              <>
                <div className="upload-zone" onClick={() => fileRef.current?.click()}>
                  {preview
                    ? <img src={preview} className="upload-preview" alt="preview" />
                    : <div className="upload-placeholder"><span>⊕</span><small>Click to select image</small></div>
                  }
                  <input ref={fileRef} type="file" accept="image/*" style={{ display: 'none' }} onChange={handleFileChange} />
                </div>
                <input ref={imgDescRef} className="form-input" placeholder="Description (optional)" />
              </>
            )}
            {addMode === 'description' && (
              <textarea ref={descRef} className="form-textarea" rows={3} placeholder="Describe the suspect…" />
            )}
            <select ref={camRef} className="form-select">
              <option value="">Camera (optional)</option>
              {cameras.map(c => <option key={c.id} value={c.id}>{c.id} — {c.metadata?.location_name}</option>)}
            </select>
            <button className="btn btn-accent" disabled={submitting} onClick={handleSubmit}>
              {submitting ? 'SUBMITTING…' : 'SUBMIT'}
            </button>
          </div>
        </div>
      )}

      {/* Embedding modal */}
      {viewEmbed && (
        <div className="modal-overlay" onClick={() => setViewEmbed(null)}>
          <div className="modal-box" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <span>EMBEDDING — {shortId(viewEmbed.suspect_id)}</span>
              <button className="close-btn" onClick={() => setViewEmbed(null)}>✕</button>
            </div>
            <EmbeddingChart embedding={viewEmbed.embedding} />
          </div>
        </div>
      )}

      {/* Table */}
      <div className="table-wrap">
        <div className="table-stat">
          Showing {filtered.length} of {suspects.length} suspects
        </div>
        <table className="data-table">
          <thead>
            <tr>
              <th>SUSPECT ID</th>
              <th>CAMERA</th>
              <th>SOURCE</th>
              <th>TIMESTAMP</th>
              <th>DESCRIPTION</th>
              <th>MATCHES</th>
              <th>EMBEDDING</th>
              <th>ACTIONS</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map(s => {
              const badge = sourceBadge(s.source);
              const matches = matchMap[s.suspect_id] || [];
              return (
                <tr key={s.suspect_id} className={matches.length ? 'row-match' : ''}>
                  <td><span className="id-cell" title={s.suspect_id}>{shortId(s.suspect_id)}</span></td>
                  <td className="mono-cell">{s.camera_id}</td>
                  <td>
                    <span className="badge" style={{ color: badge.color, borderColor: badge.color + '55' }}>
                      {badge.label}
                    </span>
                  </td>
                  <td className="mono-cell ts-cell">
                    <div>{formatTs(s.timestamp)}</div>
                    <div className="age-hint">{formatAge(s.timestamp)}</div>
                  </td>
                  <td className="desc-cell">{s.description || <span className="null-val">—</span>}</td>
                  <td>
                    {matches.length > 0
                      ? <span className="match-pill">⚠ {matches.length}</span>
                      : <span className="null-val">—</span>
                    }
                  </td>
                  <td>
                    {s.embedding?.length > 0
                      ? <button className="icon-btn" onClick={() => setViewEmbed(s)}>VIEW</button>
                      : <span className="null-val">zero</span>
                    }
                  </td>
                  <td>
                    <button className="icon-btn danger-btn" onClick={() => actions.deleteSuspect(s.suspect_id)}>✕</button>
                  </td>
                </tr>
              );
            })}
            {filtered.length === 0 && (
              <tr><td colSpan={8} className="empty-row">No suspects match the current filter</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
