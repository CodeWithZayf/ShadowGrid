/**
 * SENTINEL — AI Surveillance Console
 * app.js — Complete frontend logic
 *
 * Communicates with the FastAPI server at the configured base URL.
 * All API calls are centralised in the `api` object.
 * UI is split into modules matching the server's domain objects.
 */

'use strict';

// ═══════════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════════

const State = {
  baseUrl: 'http://localhost:8000',
  connected: false,
  cameras: [],
  suspects: [],
  occurrences: [],
  health: null,
  pollInterval: null,
  activeCameraId: null,     // camera currently shown in detail panel
  streamRefreshInterval: null,
};

// ═══════════════════════════════════════════════════════════════════
// DOM REFS — resolve once
// ═══════════════════════════════════════════════════════════════════

const $ = id => document.getElementById(id);
const $$ = sel => document.querySelectorAll(sel);

const DOM = {
  serverUrl:          $('serverUrl'),
  connectBtn:         $('connectBtn'),
  statusDot:          $('statusDot'),
  statusLabel:        $('statusLabel'),
  hCams:              $('hCams'),
  hSuspects:          $('hSuspects'),
  hOccurrences:       $('hOccurrences'),
  clock:              $('clock'),
  toastContainer:     $('toastContainer'),

  // Cameras tab
  cameraGrid:         $('cameraGrid'),
  cameraEmpty:        $('cameraEmpty'),
  refreshCamerasBtn:  $('refreshCamerasBtn'),
  camDetailOverlay:   $('camDetailOverlay'),
  detailCamId:        $('detailCamId'),
  closeCamDetail:     $('closeCamDetail'),
  liveStreamImg:      $('liveStreamImg'),
  camMetaGrid:        $('camMetaGrid'),
  removeCamSuspectsBtn: $('removeCamSuspectsBtn'),

  // Suspects tab
  refreshSuspectsBtn: $('refreshSuspectsBtn'),
  openAddSuspectBtn:  $('openAddSuspectBtn'),
  addSuspectPanel:    $('addSuspectPanel'),
  closeAddSuspectBtn: $('closeAddSuspectBtn'),
  suspectsBody:       $('suspectsBody'),
  suspectsEmpty:      $('suspectsEmpty'),
  suspectImageFile:   $('suspectImageFile'),
  uploadZone:         $('uploadZone'),
  uploadPreview:      $('uploadPreview'),
  imgCameraId:        $('imgCameraId'),
  imgDescription:     $('imgDescription'),
  submitImageSuspect: $('submitImageSuspect'),
  descText:           $('descText'),
  descCameraId:       $('descCameraId'),
  submitDescSuspect:  $('submitDescSuspect'),

  // Occurrences tab
  refreshOccurrencesBtn: $('refreshOccurrencesBtn'),
  occFilterType:      $('occFilterType'),
  occFilterValue:     $('occFilterValue'),
  applyOccFilter:     $('applyOccFilter'),
  clearOccFilter:     $('clearOccFilter'),
  occurrencesBody:    $('occurrencesBody'),
  occurrencesEmpty:   $('occurrencesEmpty'),

  // Health tab
  refreshHealthBtn:   $('refreshHealthBtn'),
  hsStatus:           $('hs-status'),
  hsSuspects:         $('hs-suspects'),
  hsCams:             $('hs-cams'),
  hsOccurrences:      $('hs-occurrences'),
  queueBarsWrap:      $('queueBarsWrap'),
  healthRawJson:      $('healthRawJson'),
  etMethod:           $('etMethod'),
  etPath:             $('etPath'),
  etBody:             $('etBody'),
  etSendBtn:          $('etSendBtn'),
  etStatusBadge:      $('etStatusBadge'),
  etResponseBody:     $('etResponseBody'),

  // Embedding modal
  embeddingModal:     $('embeddingModal'),
  closeEmbeddingModal:$('closeEmbeddingModal'),
  embedMeta:          $('embedMeta'),
  embedChart:         $('embedChart'),
  embedStats:         $('embedStats'),
  embedRaw:           $('embedRaw'),
};

// ═══════════════════════════════════════════════════════════════════
// API LAYER
// ═══════════════════════════════════════════════════════════════════

const api = {
  async request(method, path, body = null, formData = null) {
    const url = State.baseUrl.replace(/\/$/, '') + path;
    const opts = { method };
    if (formData) {
      opts.body = formData;
    } else if (body !== null) {
      opts.headers = { 'Content-Type': 'application/json' };
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(url, opts);
    const contentType = res.headers.get('content-type') || '';
    let data = null;
    if (contentType.includes('application/json')) {
      data = await res.json();
    } else if (res.status !== 204) {
      data = await res.text();
    }
    return { status: res.status, ok: res.ok, data };
  },

  get:    (path)        => api.request('GET', path),
  delete: (path)        => api.request('DELETE', path),
  post:   (path, body)  => api.request('POST', path, body),
  postForm: (path, fd)  => api.request('POST', path, null, fd),

  // Specific endpoints
  health:            () => api.get('/health'),
  cameras:           () => api.get('/cameras'),
  suspects:          () => api.get('/suspects'),
  occurrences:       () => api.get('/occurrences'),
  occByCamera:       (id) => api.get(`/occurrences/camera/${encodeURIComponent(id)}`),
  occBySuspect:      (id) => api.get(`/occurrences/suspect/${encodeURIComponent(id)}`),
  deleteSuspect:     (id) => api.delete(`/suspects/${encodeURIComponent(id)}`),
  removeCamSuspects: (id) => api.delete(`/cameras/${encodeURIComponent(id)}/suspect`),
  addSuspectImage:   (fd)  => api.postForm('/suspects/upload', fd),
  addSuspectDesc:    (fd)  => api.postForm('/suspects/description', fd),

  streamUrl: (cameraId) =>
    `${State.baseUrl.replace(/\/$/, '')}/cameras/${encodeURIComponent(cameraId)}/stream`,
  frameUrl:  (cameraId) =>
    `${State.baseUrl.replace(/\/$/, '')}/cameras/${encodeURIComponent(cameraId)}/frame`,
};

// ═══════════════════════════════════════════════════════════════════
// TOAST
// ═══════════════════════════════════════════════════════════════════

function toast(msg, type = 'success', duration = 3000) {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  DOM.toastContainer.appendChild(el);
  setTimeout(() => {
    el.classList.add('toast-out');
    el.addEventListener('animationend', () => el.remove(), { once: true });
  }, duration);
}

// ═══════════════════════════════════════════════════════════════════
// STATUS
// ═══════════════════════════════════════════════════════════════════

function setStatus(online, label = null) {
  State.connected = online;
  DOM.statusDot.className = 'status-dot' + (online ? ' online' : ' error');
  DOM.statusLabel.textContent = label || (online ? 'ONLINE' : 'OFFLINE');
}

function updateHeaderStats() {
  DOM.hCams.textContent        = State.cameras.length;
  DOM.hSuspects.textContent    = State.suspects.length;
  DOM.hOccurrences.textContent = State.occurrences.length;
}

// ═══════════════════════════════════════════════════════════════════
// CONNECT
// ═══════════════════════════════════════════════════════════════════

async function connect() {
  State.baseUrl = DOM.serverUrl.value.trim().replace(/\/$/, '');
  DOM.connectBtn.disabled = true;
  DOM.connectBtn.textContent = '…';

  try {
    const r = await api.health();
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    setStatus(true, 'ONLINE');
    State.health = r.data;
    toast('Connected to server', 'success');
    await refreshAll();
    startPolling();
  } catch (e) {
    setStatus(false, 'ERROR');
    toast(`Connection failed: ${e.message}`, 'error', 5000);
  } finally {
    DOM.connectBtn.disabled = false;
    DOM.connectBtn.textContent = 'CONNECT';
  }
}

function startPolling() {
  if (State.pollInterval) clearInterval(State.pollInterval);
  State.pollInterval = setInterval(async () => {
    if (!State.connected) return;
    try {
      await refreshAll();
    } catch (_) { /* silent */ }
  }, 5000);
}

async function refreshAll() {
  await Promise.allSettled([
    refreshCameras(),
    refreshSuspects(),
    refreshOccurrences(),
    refreshHealth(),
  ]);
  updateHeaderStats();
}

// ═══════════════════════════════════════════════════════════════════
// CAMERAS
// ═══════════════════════════════════════════════════════════════════

async function refreshCameras() {
  const r = await api.cameras();
  if (!r.ok) return;
  State.cameras = r.data || [];
  renderCameraGrid();
}

function renderCameraGrid() {
  const grid = DOM.cameraGrid;
  const cams = State.cameras;

  if (cams.length === 0) {
    DOM.cameraEmpty.style.display = '';
    // Remove any existing cards
    grid.querySelectorAll('.camera-card').forEach(c => c.remove());
    return;
  }
  DOM.cameraEmpty.style.display = 'none';

  // Keyed update: add new, remove stale
  const existingIds = new Set(
    [...grid.querySelectorAll('.camera-card')].map(el => el.dataset.camId)
  );
  const currentIds = new Set(cams.map(c => c.id));

  // Remove stale
  grid.querySelectorAll('.camera-card').forEach(el => {
    if (!currentIds.has(el.dataset.camId)) el.remove();
  });

  // Add new cards
  cams.forEach(cam => {
    if (existingIds.has(cam.id)) return; // already rendered
    const card = buildCameraCard(cam);
    grid.appendChild(card);
  });

  // Refresh thumbnails on existing cards
  grid.querySelectorAll('.camera-card img[data-cam-thumb]').forEach(img => {
    const camId = img.closest('.camera-card').dataset.camId;
    refreshThumb(img, camId);
  });
}

function buildCameraCard(cam) {
  const card = document.createElement('div');
  card.className = 'camera-card';
  card.dataset.camId = cam.id;

  const meta = cam.metadata || {};
  const lat  = meta.latitude  != null ? meta.latitude.toFixed(5)  : '—';
  const lon  = meta.longitude != null ? meta.longitude.toFixed(5) : '—';

  card.innerHTML = `
    <div class="cam-card-thumb">
      <img data-cam-thumb src="" alt="Camera ${cam.id}" loading="lazy" />
      <div class="cam-card-overlay">${cam.id}</div>
    </div>
    <div class="cam-card-info">
      <div class="cam-card-id">${cam.id}</div>
      <div class="cam-card-meta">
        <div class="cam-meta-row">
          <span class="cam-meta-lbl">LOCATION</span>
          <span>${meta.location_name || '—'}</span>
        </div>
        <div class="cam-meta-row">
          <span class="cam-meta-lbl">LAT</span>
          <span class="cell-mono">${lat}</span>
        </div>
        <div class="cam-meta-row">
          <span class="cam-meta-lbl">LON</span>
          <span class="cell-mono">${lon}</span>
        </div>
        <div class="cam-meta-row">
          <span class="cam-meta-lbl">URL</span>
          <span style="word-break:break-all;font-size:10px;color:var(--text-dim)">${cam.url || '—'}</span>
        </div>
      </div>
    </div>
  `;

  const img = card.querySelector('img');
  refreshThumb(img, cam.id);
  card.addEventListener('click', () => openCameraDetail(cam));
  return card;
}

function refreshThumb(img, camId) {
  // Fetch latest frame as JPEG; use cache-busting timestamp
  const url = api.frameUrl(camId) + `?t=${Date.now()}`;
  const tmp = new Image();
  tmp.onload  = () => { img.src = tmp.src; img.style.opacity = '1'; };
  tmp.onerror = () => { img.src = ''; };
  tmp.src = url;
}

// ── Camera detail panel ─────────────────────────────────────────────

function openCameraDetail(cam) {
  State.activeCameraId = cam.id;
  const meta = cam.metadata || {};

  DOM.detailCamId.textContent = cam.id;

  // Build meta grid
  DOM.camMetaGrid.innerHTML = [
    ['LOCATION', meta.location_name || '—'],
    ['LATITUDE',  meta.latitude  != null ? meta.latitude.toFixed(6)  : '—'],
    ['LONGITUDE', meta.longitude != null ? meta.longitude.toFixed(6) : '—'],
    ['STREAM URL', cam.url || '—'],
  ].map(([lbl, val]) => `
    <div class="cam-meta-cell">
      <div class="cmc-label">${lbl}</div>
      <div class="cmc-value" style="word-break:break-all">${val}</div>
    </div>
  `).join('');

  // Live stream
  DOM.liveStreamImg.src = api.streamUrl(cam.id);

  DOM.camDetailOverlay.classList.remove('hidden');
}

function closeCameraDetail() {
  DOM.camDetailOverlay.classList.add('hidden');
  DOM.liveStreamImg.src = ''; // stop the stream
  State.activeCameraId = null;
}

async function removeCameraFromSuspects() {
  const camId = State.activeCameraId;
  if (!camId) return;
  try {
    const r = await api.removeCamSuspects(camId);
    if (!r.ok) throw new Error(r.data?.detail || `HTTP ${r.status}`);
    toast(`Removed ${r.data.suspects_removed} suspect(s) from ${camId}`, 'success');
    await refreshSuspects();
    updateHeaderStats();
  } catch (e) {
    toast(`Error: ${e.message}`, 'error');
  }
}

// ═══════════════════════════════════════════════════════════════════
// SUSPECTS
// ═══════════════════════════════════════════════════════════════════

async function refreshSuspects() {
  const r = await api.suspects();
  if (!r.ok) return;
  State.suspects = r.data || [];
  renderSuspectsTable(State.suspects);
}

function renderSuspectsTable(suspects) {
  const empty = DOM.suspectsEmpty;
  const tbody = DOM.suspectsBody;

  if (suspects.length === 0) {
    empty.classList.remove('hidden');
    tbody.innerHTML = '';
    return;
  }
  empty.classList.add('hidden');

  tbody.innerHTML = suspects.map(s => {
    const ts   = formatTimestamp(s.timestamp);
    const src  = sourceBadge(s.source);
    const desc = s.description ? escHtml(s.description.slice(0, 50)) + (s.description.length > 50 ? '…' : '') : '<span style="color:var(--text-dim)">—</span>';
    const emb  = s.embedding ? `<button class="embed-btn" data-embed='${safeEmbedAttr(s.embedding)}' data-meta="Suspect: ${s.suspect_id}">VIEW</button>` : '—';
    return `
      <tr>
        <td><div class="cell-id" title="${s.suspect_id}">${shortId(s.suspect_id)}</div></td>
        <td class="cell-mono">${escHtml(s.camera_id || '—')}</td>
        <td class="cell-mono">${s.car_id === -1 ? '—' : s.car_id}</td>
        <td>${src}</td>
        <td class="cell-mono" style="white-space:nowrap">${ts}</td>
        <td style="max-width:180px">${desc}</td>
        <td>${emb}</td>
        <td>
          <button class="del-btn" data-suspect-id="${s.suspect_id}">✕ DEL</button>
        </td>
      </tr>
    `;
  }).join('');

  // Bind embed buttons
  tbody.querySelectorAll('.embed-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const embedding = JSON.parse(btn.getAttribute('data-embed'));
      const meta = btn.getAttribute('data-meta');
      openEmbeddingModal(embedding, meta);
    });
  });

  // Bind delete buttons
  tbody.querySelectorAll('.del-btn').forEach(btn => {
    btn.addEventListener('click', () => deleteSuspect(btn.dataset.suspectId));
  });
}

async function deleteSuspect(id) {
  if (!confirm(`Remove suspect ${shortId(id)}?`)) return;
  try {
    const r = await api.deleteSuspect(id);
    if (!r.ok) throw new Error(r.data?.detail || `HTTP ${r.status}`);
    toast('Suspect removed', 'success');
    await refreshSuspects();
    updateHeaderStats();
  } catch (e) {
    toast(`Error: ${e.message}`, 'error');
  }
}

// ── Add suspect from image ──────────────────────────────────────────

function setupImageUpload() {
  const zone  = DOM.uploadZone;
  const input = DOM.suspectImageFile;
  const prev  = DOM.uploadPreview;

  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) previewFile(file);
  });

  input.addEventListener('change', () => {
    if (input.files[0]) previewFile(input.files[0]);
  });

  function previewFile(file) {
    const reader = new FileReader();
    reader.onload = e => {
      prev.src = e.target.result;
      prev.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
    // Store on input if dropped
    if (input.files.length === 0) {
      const dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
    }
  }
}

async function submitImageSuspect() {
  const file = DOM.suspectImageFile.files[0];
  if (!file) { toast('Select an image first', 'info'); return; }

  DOM.submitImageSuspect.disabled = true;
  DOM.submitImageSuspect.textContent = 'EMBEDDING…';

  try {
    const fd = new FormData();
    fd.append('file', file);
    const camId = DOM.imgCameraId.value.trim();
    const desc  = DOM.imgDescription.value.trim();
    if (camId) fd.append('camera_id', camId);
    if (desc)  fd.append('description', desc);

    const r = await api.addSuspectImage(fd);
    if (!r.ok) throw new Error(r.data?.detail || `HTTP ${r.status}`);
    toast('Suspect added from image', 'success');

    // Reset form
    DOM.suspectImageFile.value = '';
    DOM.uploadPreview.src = '';
    DOM.uploadPreview.classList.add('hidden');
    DOM.imgCameraId.value = '';
    DOM.imgDescription.value = '';

    await refreshSuspects();
    updateHeaderStats();
  } catch (e) {
    toast(`Error: ${e.message}`, 'error');
  } finally {
    DOM.submitImageSuspect.disabled = false;
    DOM.submitImageSuspect.textContent = 'SUBMIT & EMBED';
  }
}

async function submitDescSuspect() {
  const desc = DOM.descText.value.trim();
  if (!desc) { toast('Enter a description', 'info'); return; }

  DOM.submitDescSuspect.disabled = true;

  try {
    const fd = new FormData();
    fd.append('description', desc);
    const camId = DOM.descCameraId.value.trim();
    if (camId) fd.append('camera_id', camId);

    const r = await api.addSuspectDesc(fd);
    if (!r.ok) throw new Error(r.data?.detail || `HTTP ${r.status}`);
    toast('Suspect added from description', 'success');

    DOM.descText.value = '';
    DOM.descCameraId.value = '';

    await refreshSuspects();
    updateHeaderStats();
  } catch (e) {
    toast(`Error: ${e.message}`, 'error');
  } finally {
    DOM.submitDescSuspect.disabled = false;
  }
}

// ═══════════════════════════════════════════════════════════════════
// OCCURRENCES
// ═══════════════════════════════════════════════════════════════════

async function refreshOccurrences() {
  const r = await api.occurrences();
  if (!r.ok) return;
  State.occurrences = r.data || [];
  renderOccurrencesTable(State.occurrences);
}

function renderOccurrencesTable(occs) {
  const empty = DOM.occurrencesEmpty;
  const tbody = DOM.occurrencesBody;

  if (occs.length === 0) {
    empty.classList.remove('hidden');
    tbody.innerHTML = '';
    return;
  }
  empty.classList.add('hidden');

  tbody.innerHTML = occs.map(o => {
    const ts   = formatTimestamp(o.timestamp);
    const sim  = typeof o.similarity === 'number' ? o.similarity : 0;
    const simCls = sim >= 0.85 ? 'sim-high' : sim >= 0.70 ? 'sim-med' : 'sim-low';
    const simPct = Math.round(sim * 100);
    const emb  = o.embedding ? `<button class="embed-btn" data-embed='${safeEmbedAttr(o.embedding)}' data-meta="Occurrence: ${o.occurrence_id}">VIEW</button>` : '—';
    return `
      <tr>
        <td><div class="cell-id" title="${o.occurrence_id}">${shortId(o.occurrence_id)}</div></td>
        <td><div class="cell-id" title="${o.suspect_id}">${shortId(o.suspect_id)}</div></td>
        <td class="cell-mono">${escHtml(o.camera_id || '—')}</td>
        <td class="cell-mono">${o.car_id}</td>
        <td>
          <div class="sim-bar-wrap">
            <div class="sim-bar-bg">
              <div class="sim-bar-fill ${simCls}" style="width:${simPct}%"></div>
            </div>
            <span class="sim-val">${simPct}%</span>
          </div>
        </td>
        <td class="cell-mono" style="white-space:nowrap">${ts}</td>
        <td>${emb}</td>
      </tr>
    `;
  }).join('');

  // Bind embed buttons
  tbody.querySelectorAll('.embed-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const embedding = JSON.parse(btn.getAttribute('data-embed'));
      const meta = btn.getAttribute('data-meta');
      openEmbeddingModal(embedding, meta);
    });
  });
}

async function applyOccurrenceFilter() {
  const filterType = DOM.occFilterType.value;
  const filterVal  = DOM.occFilterValue.value.trim();

  if (filterType === 'all' || !filterVal) {
    renderOccurrencesTable(State.occurrences);
    return;
  }

  try {
    let r;
    if (filterType === 'suspect') {
      r = await api.occBySuspect(filterVal);
    } else {
      r = await api.occByCamera(filterVal);
    }
    if (!r.ok) throw new Error(r.data?.detail || `HTTP ${r.status}`);
    renderOccurrencesTable(r.data || []);
    toast(`Filtered: ${(r.data || []).length} result(s)`, 'info');
  } catch (e) {
    toast(`Filter error: ${e.message}`, 'error');
  }
}

// ═══════════════════════════════════════════════════════════════════
// HEALTH
// ═══════════════════════════════════════════════════════════════════

async function refreshHealth() {
  const r = await api.health();
  if (!r.ok) { setStatus(false, 'ERROR'); return; }
  State.health = r.data;
  renderHealth(r.data);
  setStatus(true, 'ONLINE');
}

function renderHealth(data) {
  DOM.hsStatus.textContent      = data.status ? data.status.toUpperCase() : '—';
  DOM.hsCams.textContent        = data.cameras_registered ?? '—';
  DOM.hsSuspects.textContent    = data.suspects_count ?? '—';
  DOM.hsOccurrences.textContent = data.occurrences_count ?? '—';
  DOM.healthRawJson.textContent = JSON.stringify(data, null, 2);

  // Queue bars
  const queues = data.queues || {};
  const keys   = Object.keys(queues);
  const maxQ   = Math.max(1, ...Object.values(queues));

  if (keys.length === 0) {
    DOM.queueBarsWrap.innerHTML = '<div class="empty-mini">No queues.</div>';
    return;
  }

  DOM.queueBarsWrap.innerHTML = keys.map(k => {
    const val = queues[k];
    const pct = Math.round((val / maxQ) * 100);
    return `
      <div class="queue-bar-row">
        <span class="qb-label" title="${k}">${k}</span>
        <div class="qb-track">
          <div class="qb-fill" style="width:${pct}%"></div>
        </div>
        <span class="qb-val">${val}</span>
      </div>
    `;
  }).join('');
}

// ─── Endpoint tester ────────────────────────────────────────────────

async function sendEndpointTest() {
  const method = DOM.etMethod.value;
  const path   = DOM.etPath.value.trim() || '/health';
  const rawBody = DOM.etBody.value.trim();

  DOM.etSendBtn.disabled = true;
  DOM.etStatusBadge.className = 'et-status-badge';
  DOM.etStatusBadge.textContent = '…';
  DOM.etResponseBody.textContent = 'Loading…';

  let body = null;
  if (rawBody) {
    try { body = JSON.parse(rawBody); }
    catch (_) { toast('Invalid JSON body', 'error'); DOM.etSendBtn.disabled = false; return; }
  }

  try {
    const r = await api.request(method, path, body);
    DOM.etStatusBadge.textContent = r.status;
    DOM.etStatusBadge.classList.add(r.ok ? 'ok' : 'err');
    DOM.etResponseBody.textContent = typeof r.data === 'object'
      ? JSON.stringify(r.data, null, 2)
      : (r.data || '(empty)');
  } catch (e) {
    DOM.etStatusBadge.textContent = 'ERR';
    DOM.etStatusBadge.classList.add('err');
    DOM.etResponseBody.textContent = e.message;
  } finally {
    DOM.etSendBtn.disabled = false;
  }
}

// ═══════════════════════════════════════════════════════════════════
// EMBEDDING MODAL
// ═══════════════════════════════════════════════════════════════════

function openEmbeddingModal(embedding, metaLabel) {
  const arr = Array.isArray(embedding) ? embedding : [];

  // Meta info
  DOM.embedMeta.innerHTML = `
    <div><span style="color:var(--text-dim)">SOURCE:</span> ${escHtml(metaLabel)}</div>
    <div><span style="color:var(--text-dim)">DIMENSIONS:</span> ${arr.length}</div>
  `;

  // Stats
  const min   = Math.min(...arr);
  const max   = Math.max(...arr);
  const mean  = arr.reduce((a, b) => a + b, 0) / (arr.length || 1);
  const norm  = Math.sqrt(arr.reduce((a, b) => a + b * b, 0));
  const nonzero = arr.filter(v => Math.abs(v) > 1e-6).length;

  DOM.embedStats.innerHTML = [
    ['MIN',  min.toFixed(4)],
    ['MAX',  max.toFixed(4)],
    ['MEAN', mean.toFixed(4)],
    ['L2',   norm.toFixed(4)],
    ['NON-ZERO', `${nonzero}/${arr.length}`],
  ].map(([lbl, val]) => `
    <div class="embed-stat">
      <span class="es-val">${val}</span>
      <span class="es-lbl">${lbl}</span>
    </div>
  `).join('');

  // Raw
  DOM.embedRaw.textContent = arr.map((v, i) =>
    `[${String(i).padStart(3, '0')}] ${v >= 0 ? ' ' : ''}${v.toFixed(6)}`
  ).join('\n');

  // Chart
  drawEmbeddingChart(arr);

  DOM.embeddingModal.classList.remove('hidden');
}

function drawEmbeddingChart(arr) {
  const canvas = DOM.embedChart;
  const ctx    = canvas.getContext('2d');
  const dpr    = window.devicePixelRatio || 1;

  // Resize canvas properly
  const rect = canvas.parentElement.getBoundingClientRect();
  const w    = Math.floor(rect.width || 600);
  const h    = 100;
  canvas.width  = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width  = w + 'px';
  canvas.style.height = h + 'px';
  ctx.scale(dpr, dpr);

  ctx.clearRect(0, 0, w, h);

  if (arr.length === 0) return;

  const min = Math.min(...arr);
  const max = Math.max(...arr);
  const range = max - min || 1;

  const step   = w / arr.length;
  const padT   = 8;
  const padB   = 8;
  const drawH  = h - padT - padB;
  const zeroY  = padT + drawH * (1 - (0 - min) / range);

  // Background grid
  ctx.strokeStyle = '#1e2e22';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padT + (drawH / 4) * i;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }

  // Zero line
  ctx.strokeStyle = '#2a4030';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(0, zeroY); ctx.lineTo(w, zeroY); ctx.stroke();
  ctx.setLineDash([]);

  // Fill under curve
  const gradient = ctx.createLinearGradient(0, padT, 0, h);
  gradient.addColorStop(0,   'rgba(61,220,110,0.3)');
  gradient.addColorStop(0.5, 'rgba(61,220,110,0.08)');
  gradient.addColorStop(1,   'rgba(61,220,110,0)');

  ctx.beginPath();
  arr.forEach((v, i) => {
    const x = i * step + step / 2;
    const y = padT + drawH * (1 - (v - min) / range);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.lineTo((arr.length - 1) * step + step / 2, h);
  ctx.lineTo(step / 2, h);
  ctx.closePath();
  ctx.fillStyle = gradient;
  ctx.fill();

  // Line
  ctx.beginPath();
  ctx.strokeStyle = '#3ddc6e';
  ctx.lineWidth = 1.5;
  arr.forEach((v, i) => {
    const x = i * step + step / 2;
    const y = padT + drawH * (1 - (v - min) / range);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

// ═══════════════════════════════════════════════════════════════════
// TAB NAVIGATION
// ═══════════════════════════════════════════════════════════════════

function setupTabs() {
  $$('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      $$('.tab-btn').forEach(b => b.classList.remove('active'));
      $$('.tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      $(`tab-${btn.dataset.tab}`).classList.add('active');

      // Refresh data when switching to tab
      if (!State.connected) return;
      const tab = btn.dataset.tab;
      if (tab === 'cameras')     refreshCameras();
      if (tab === 'suspects')    refreshSuspects();
      if (tab === 'occurrences') refreshOccurrences();
      if (tab === 'health')      refreshHealth();
    });
  });
}

// ═══════════════════════════════════════════════════════════════════
// ADD SUSPECT PANEL TABS
// ═══════════════════════════════════════════════════════════════════

function setupAspTabs() {
  $$('.asp-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      $$('.asp-tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const mode = btn.dataset.aspmode;
      $$('.asp-mode').forEach(m => m.classList.add('hidden'));
      $(`aspMode-${mode}`).classList.remove('hidden');
    });
  });
}

// ═══════════════════════════════════════════════════════════════════
// CLOCK
// ═══════════════════════════════════════════════════════════════════

function startClock() {
  setInterval(() => {
    const now = new Date();
    DOM.clock.textContent = now.toTimeString().slice(0, 8);
  }, 1000);
}

// ═══════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function shortId(id) {
  if (!id) return '—';
  if (id.length <= 13) return id;
  return id.slice(0, 8) + '…' + id.slice(-4);
}

function formatTimestamp(ts) {
  if (!ts) return '—';
  const d = new Date(ts * 1000);
  return d.toLocaleString('en-GB', {
    year: '2-digit', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
    hour12: false,
  }).replace(',', '');
}

function sourceBadge(src) {
  if (!src) return '—';
  if (src === 'crime_detection')   return `<span class="cell-badge badge-crime">CRIME</span>`;
  if (src === 'manual_upload')     return `<span class="cell-badge badge-upload">IMAGE</span>`;
  if (src === 'manual_description') return `<span class="cell-badge badge-desc">TEXT</span>`;
  return `<span class="cell-badge badge-desc">${escHtml(src)}</span>`;
}

function safeEmbedAttr(embedding) {
  // Store as compact JSON; we'll parse it back in the click handler
  // Limit to first 512 values to keep DOM attribute size sane
  const arr = Array.isArray(embedding) ? embedding.slice(0, 512) : [];
  return JSON.stringify(arr).replace(/'/g, '&#39;');
}

// ═══════════════════════════════════════════════════════════════════
// EVENT WIRING
// ═══════════════════════════════════════════════════════════════════

function wireEvents() {
  // Connect
  DOM.connectBtn.addEventListener('click', connect);
  DOM.serverUrl.addEventListener('keydown', e => { if (e.key === 'Enter') connect(); });

  // Cameras
  DOM.refreshCamerasBtn.addEventListener('click', refreshCameras);
  DOM.closeCamDetail.addEventListener('click', closeCameraDetail);
  DOM.camDetailOverlay.addEventListener('click', e => {
    if (e.target === DOM.camDetailOverlay) closeCameraDetail();
  });
  DOM.removeCamSuspectsBtn.addEventListener('click', removeCameraFromSuspects);

  // Suspects
  DOM.refreshSuspectsBtn.addEventListener('click', refreshSuspects);
  DOM.openAddSuspectBtn.addEventListener('click', () => DOM.addSuspectPanel.classList.remove('hidden'));
  DOM.closeAddSuspectBtn.addEventListener('click', () => DOM.addSuspectPanel.classList.add('hidden'));
  DOM.submitImageSuspect.addEventListener('click', submitImageSuspect);
  DOM.submitDescSuspect.addEventListener('click', submitDescSuspect);

  // Occurrences
  DOM.refreshOccurrencesBtn.addEventListener('click', refreshOccurrences);
  DOM.applyOccFilter.addEventListener('click', applyOccurrenceFilter);
  DOM.clearOccFilter.addEventListener('click', () => {
    DOM.occFilterValue.value = '';
    DOM.occFilterType.value  = 'all';
    renderOccurrencesTable(State.occurrences);
  });
  DOM.occFilterValue.addEventListener('keydown', e => { if (e.key === 'Enter') applyOccurrenceFilter(); });

  // Health
  DOM.refreshHealthBtn.addEventListener('click', refreshHealth);
  DOM.etSendBtn.addEventListener('click', sendEndpointTest);
  DOM.etPath.addEventListener('keydown', e => { if (e.key === 'Enter') sendEndpointTest(); });

  // Embedding modal
  DOM.closeEmbeddingModal.addEventListener('click', () => DOM.embeddingModal.classList.add('hidden'));
  DOM.embeddingModal.addEventListener('click', e => {
    if (e.target === DOM.embeddingModal) DOM.embeddingModal.classList.add('hidden');
  });

  // Keyboard: Escape closes any open overlay
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
      DOM.embeddingModal.classList.add('hidden');
      closeCameraDetail();
    }
  });
}

// ═══════════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════════

function init() {
  setupTabs();
  setupAspTabs();
  setupImageUpload();
  wireEvents();
  startClock();
  setStatus(false);
}

document.addEventListener('DOMContentLoaded', init);
