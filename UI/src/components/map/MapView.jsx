// src/components/map/MapView.jsx
import React, { useEffect, useRef, useCallback } from 'react';
import L from 'leaflet';
import { useApp } from '../../store/AppContext.jsx';
import { suspectColor, shortId } from '../../utils/format.js';

// Fix Leaflet default icon paths broken by Vite
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl:       'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl:     'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

// ── Color palette ─────────────────────────────────────────────────────────────
const COL = {
  cameraLive:      '#34d399',    // emerald
  cameraDead:      '#6b7280',
  suspect:         '#ef4444',
};

// ── Custom DivIcon factories ──────────────────────────────────────────────────
function makeIcon(color, symbol, title) {
  return L.divIcon({
    className: '',
    iconSize: [28, 28],
    iconAnchor: [14, 14],
    html: `<div title="${title}" style="
      width:28px;height:28px;border-radius:50%;
      background:${color};border:2px solid rgba(255,255,255,0.7);
      display:flex;align-items:center;justify-content:center;
      font-size:13px;color:#fff;font-weight:700;
      box-shadow:0 0 8px ${color}88;
    ">${symbol}</div>`,
  });
}

function makeSuspectIcon(color, idx) {
  return L.divIcon({
    className: '',
    iconSize: [32, 32],
    iconAnchor: [16, 16],
    html: `<div style="
      width:32px;height:32px;border-radius:50%;
      background:${color};border:2.5px solid #fff;
      display:flex;align-items:center;justify-content:center;
      font-size:11px;color:#fff;font-weight:800;
      box-shadow:0 0 12px ${color};
      animation:pulse-marker 2s ease-in-out infinite;
    ">${idx + 1}</div>`,
  });
}

// ── MapView ───────────────────────────────────────────────────────────────────
export default function MapView() {
  const mapRef      = useRef(null);
  const mapInstance = useRef(null);
  const layersRef   = useRef({});  // named layer groups we rebuild each render

  const { state, actions } = useApp();
  const {
    cameras, suspects, occurrences,
    selectedSuspectId, cdHealth,
  } = state;

  // ── Initialise map once ────────────────────────────────────────────────────
  useEffect(() => {
    if (mapInstance.current) return;

    const map = L.map(mapRef.current, {
      center: [22.57, 88.36],
      zoom: 13,
      zoomControl: false,
      preferCanvas: true,
    });

    L.control.zoom({ position: 'bottomright' }).addTo(map);

    // OSM tiles — max zoom 19 for full road detail
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap contributors',
      maxZoom: 19,
      maxNativeZoom: 19,
    }).addTo(map);

    // Named layer groups
    const groups = ['cameraMarkers', 'suspectMarkers'];
    groups.forEach(name => {
      layersRef.current[name] = L.layerGroup().addTo(map);
    });

    mapInstance.current = map;
    return () => {
      map.remove();
      mapInstance.current = null;
    };
  }, []);

  // ── Rebuild overlays whenever data changes ─────────────────────────────────
  const rebuildOverlays = useCallback(() => {
    const map = mapInstance.current;
    if (!map) return;
    const lg = layersRef.current;

    // Clear all
    Object.values(lg).forEach(g => g.clearLayers());

    const liveCams = cdHealth?.cameras_live || {};

    // ── Cameras ────────────────────────────────────────────────────────────
    cameras.forEach(cam => {
      const lat = cam.metadata?.latitude;
      const lon = cam.metadata?.longitude;
      if (!lat || !lon) return;
      const live = liveCams[cam.id] !== false;
      const marker = L.marker([lat, lon], {
        icon: makeIcon(live ? COL.cameraLive : COL.cameraDead, '📷', cam.metadata.location_name),
        zIndexOffset: 100,
      });
      marker.bindPopup(`
        <div class="map-popup">
          <strong>${cam.id}</strong><br/>
          ${cam.metadata.location_name}<br/>
          <span style="color:${live ? '#34d399' : '#ef4444'}">${live ? '● LIVE' : '○ OFFLINE'}</span>
        </div>
      `);
      lg.cameraMarkers.addLayer(marker);
    });

    // ── Suspect markers (from recent occurrences grouped by camera) ────────
    // TODO (Phase 3): Replace with GlobalTrack positions + uncertainty radius
    // For now, mark cameras that have had recent suspect activity
    const recentSuspectCams = new Set(suspects.map(s => s.camera_id));
    cameras.forEach(cam => {
      if (!recentSuspectCams.has(cam.id)) return;
      const lat = cam.metadata?.latitude;
      const lon = cam.metadata?.longitude;
      if (!lat || !lon) return;

      const camSuspects = suspects.filter(s => s.camera_id === cam.id);
      camSuspects.forEach((s, idx) => {
        const color = suspectColor(idx);
        const marker = L.marker([lat, lon], {
          icon: makeSuspectIcon(color, idx),
          zIndexOffset: 500,
        });
        marker.on('click', () => actions.selectSuspect(s.suspect_id));
        marker.bindPopup(`
          <div class="map-popup">
            <strong>Suspect ${idx + 1}</strong><br/>
            ID: ${shortId(s.suspect_id)}<br/>
            Camera: ${cam.id}
          </div>
        `);
        lg.suspectMarkers.addLayer(marker);
      });
    });
  }, [cameras, suspects, selectedSuspectId, cdHealth, actions]);

  useEffect(() => {
    rebuildOverlays();
  }, [rebuildOverlays]);

  return (
    <div className="mapview-wrap">
      <div ref={mapRef} className="leaflet-map" />
      <MapLegend />
    </div>
  );
}

function MapLegend() {
  const items = [
    { color: COL.cameraLive,  label: 'Live camera' },
    { color: COL.cameraDead,  label: 'Offline camera' },
    { color: COL.suspect,     label: 'Suspect activity' },
  ];
  return (
    <div className="map-legend">
      <div className="legend-title">LEGEND</div>
      {items.map(it => (
        <div key={it.label} className="legend-row">
          <span className="legend-dot" style={{ background: it.color }} />
          <span>{it.label}</span>
        </div>
      ))}
    </div>
  );
}
