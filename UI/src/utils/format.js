// src/utils/format.js

export function shortId(id) {
  if (!id) return '—';
  return id.slice(0, 8) + '…';
}

export function formatTs(ts) {
  if (!ts) return '—';
  return new Date(ts * 1000).toLocaleString('en-IN', {
    day: '2-digit', month: '2-digit', year: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
    hour12: false,
  });
}

export function formatAge(ts) {
  if (!ts) return '—';
  const sec = Math.floor(Date.now() / 1000 - ts);
  if (sec < 60)   return `${sec}s ago`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`;
  return `${Math.floor(sec / 3600)}h ago`;
}

export function formatMeters(m) {
  if (m >= 1000) return `${(m / 1000).toFixed(2)} km`;
  return `${Math.round(m)} m`;
}

export function sourceBadge(source) {
  if (source === 'crime_detection')   return { label: 'CRIME',  color: '#ef4444' };
  if (source === 'manual_upload')     return { label: 'IMAGE',  color: '#f59e0b' };
  if (source === 'manual_description') return { label: 'TEXT',  color: '#6b7280' };
  return { label: source?.toUpperCase() ?? '?', color: '#6b7280' };
}

export function targetIcon(type) {
  if (type === 'airport')   return '✈';
  if (type === 'railway')   return '🚆';
  if (type === 'crossover') return '🚧';
  return '📍';
}

export function suspectColor(index) {
  const palette = [
    '#f97316', '#a855f7', '#ec4899', '#14b8a6',
    '#eab308', '#06b6d4', '#84cc16', '#f43f5e',
  ];
  return palette[index % palette.length];
}
