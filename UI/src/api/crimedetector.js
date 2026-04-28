// src/api/crimedetector.js
// All calls to the Python FastAPI CrimeDetector server (port 8000).

const BASE = 'http://localhost:8000';

async function get(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`CrimeDetector ${path} → ${res.status}`);
  return res.json();
}

async function del(path) {
  const res = await fetch(`${BASE}${path}`, { method: 'DELETE' });
  if (!res.ok) throw new Error(`CrimeDetector DELETE ${path} → ${res.status}`);
  return res.json();
}

async function postForm(path, formData) {
  const res = await fetch(`${BASE}${path}`, { method: 'POST', body: formData });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `CrimeDetector POST ${path} → ${res.status}`);
  }
  return res.json();
}

export const CrimeDetectorAPI = {
  // Health
  health: () => get('/health'),

  // Cameras
  getCameras: () => get('/cameras'),
  streamUrl: (camId) => `${BASE}/cameras/${encodeURIComponent(camId)}/stream`,
  frameUrl:  (camId) => `${BASE}/cameras/${encodeURIComponent(camId)}/frame`,
  removeCameraSuspects: (camId) => del(`/cameras/${encodeURIComponent(camId)}/suspect`),

  // Suspects
  getSuspects: (sinceTs = null, limit = 200) => {
    let q = `/suspects?limit=${limit}`;
    if (sinceTs) q += `&since_timestamp=${sinceTs}`;
    return get(q);
  },
  deleteSuspect: (id) => del(`/suspects/${encodeURIComponent(id)}`),
  addSuspectImage: (file, cameraId, description) => {
    const fd = new FormData();
    fd.append('file', file);
    if (cameraId)   fd.append('camera_id', cameraId);
    if (description) fd.append('description', description);
    return postForm('/suspects/upload', fd);
  },
  addSuspectDescription: (description, cameraId) => {
    const fd = new FormData();
    fd.append('description', description);
    if (cameraId) fd.append('camera_id', cameraId);
    return postForm('/suspects/description', fd);
  },

  // Occurrences
  getOccurrences: (sinceTs = null, limit = 500) => {
    let q = `/occurrences?limit=${limit}`;
    if (sinceTs) q += `&since_timestamp=${sinceTs}`;
    return get(q);
  },
  getOccurrencesBySuspect: (id) => get(`/occurrences/suspect/${encodeURIComponent(id)}`),
  getOccurrencesByCamera:  (id) => get(`/occurrences/camera/${encodeURIComponent(id)}`),
};
