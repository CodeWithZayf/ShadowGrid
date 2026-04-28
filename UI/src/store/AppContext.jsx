// src/store/AppContext.jsx
// Central data store. All polling lives here; components read state via useApp().

import React, { createContext, useContext, useReducer, useEffect, useRef, useCallback } from 'react';
import { CrimeDetectorAPI } from '../api/crimedetector.js';

// ── Poll intervals (ms) ───────────────────────────────────────────────────────
const POLL_HEALTH_MS      = 3_000;
const POLL_CAMERAS_MS     = 15_000;
const POLL_SUSPECTS_MS    = 5_000;
const POLL_OCCURRENCES_MS = 4_000;

// ── Initial state ─────────────────────────────────────────────────────────────
const initialState = {
  // Connectivity
  cdOnline: false,
  cdHealth: null,

  // CrimeDetector data
  cameras: [],
  suspects: [],
  occurrences: [],

  // UI state
  selectedSuspectId: null,
  alertQueue: [],         // [{id, msg, level, ts}]
};

// ── Reducer ───────────────────────────────────────────────────────────────────
function reducer(state, action) {
  switch (action.type) {
    case 'SET_CD_ONLINE':   return { ...state, cdOnline: action.payload };
    case 'SET_CD_HEALTH':   return { ...state, cdHealth: action.payload };
    case 'SET_CAMERAS':     return { ...state, cameras: action.payload };

    case 'MERGE_SUSPECTS': {
      // Merge new suspects in; deduplicate by suspect_id; keep newest-first
      const existing = new Map(state.suspects.map(s => [s.suspect_id, s]));
      for (const s of action.payload) existing.set(s.suspect_id, s);
      const merged = [...existing.values()].sort((a, b) => b.timestamp - a.timestamp);
      return { ...state, suspects: merged };
    }

    case 'MERGE_OCCURRENCES': {
      const existing = new Map(state.occurrences.map(o => [o.occurrence_id, o]));
      for (const o of action.payload) existing.set(o.occurrence_id, o);
      const merged = [...existing.values()].sort((a, b) => b.timestamp - a.timestamp);
      return { ...state, occurrences: merged };
    }

    case 'DELETE_SUSPECT':
      return { ...state, suspects: state.suspects.filter(s => s.suspect_id !== action.payload) };

    case 'SELECT_SUSPECT':
      return { ...state, selectedSuspectId: action.payload };

    case 'PUSH_ALERT': {
      const alert = { id: Date.now() + Math.random(), ts: Date.now(), ...action.payload };
      return { ...state, alertQueue: [alert, ...state.alertQueue].slice(0, 20) };
    }

    case 'DISMISS_ALERT':
      return { ...state, alertQueue: state.alertQueue.filter(a => a.id !== action.payload) };

    default:
      return state;
  }
}

// ── Context ───────────────────────────────────────────────────────────────────
const AppContext = createContext(null);

export function AppProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState);

  // Delta-poll timestamps
  const lastSuspectTs     = useRef(0);
  const lastOccurrenceTs  = useRef(0);

  const pushAlert = useCallback((msg, level = 'info') => {
    dispatch({ type: 'PUSH_ALERT', payload: { msg, level } });
  }, []);

  // ── Poll: CrimeDetector health ──────────────────────────────────────────────
  useEffect(() => {
    async function poll() {
      try {
        const h = await CrimeDetectorAPI.health();
        dispatch({ type: 'SET_CD_HEALTH', payload: h });
        dispatch({ type: 'SET_CD_ONLINE', payload: true });
      } catch {
        dispatch({ type: 'SET_CD_ONLINE', payload: false });
      }
    }
    poll();
    const id = setInterval(poll, POLL_HEALTH_MS);
    return () => clearInterval(id);
  }, []);

  // ── Poll: cameras ───────────────────────────────────────────────────────────
  useEffect(() => {
    async function poll() {
      try {
        const cams = await CrimeDetectorAPI.getCameras();
        dispatch({ type: 'SET_CAMERAS', payload: cams });
      } catch { /* silently skip */ }
    }
    poll();
    const id = setInterval(poll, POLL_CAMERAS_MS);
    return () => clearInterval(id);
  }, []);

  // ── Poll: suspects (delta) ──────────────────────────────────────────────────
  useEffect(() => {
    async function poll() {
      try {
        const since = lastSuspectTs.current || null;
        const items = await CrimeDetectorAPI.getSuspects(since, 200);
        if (items.length > 0) {
          dispatch({ type: 'MERGE_SUSPECTS', payload: items });
          const maxTs = Math.max(...items.map(s => s.timestamp));
          if (maxTs > (lastSuspectTs.current || 0)) {
            lastSuspectTs.current = maxTs;
            pushAlert(`${items.length} new suspect(s) detected`, 'warning');
          }
        }
      } catch { /* skip */ }
    }
    poll();
    const id = setInterval(poll, POLL_SUSPECTS_MS);
    return () => clearInterval(id);
  }, [pushAlert]);

  // ── Poll: occurrences (delta) ───────────────────────────────────────────────
  useEffect(() => {
    async function poll() {
      try {
        const since = lastOccurrenceTs.current || null;
        const items = await CrimeDetectorAPI.getOccurrences(since, 500);
        if (items.length > 0) {
          dispatch({ type: 'MERGE_OCCURRENCES', payload: items });
          const maxTs = Math.max(...items.map(o => o.timestamp));
          if (maxTs > (lastOccurrenceTs.current || 0)) {
            lastOccurrenceTs.current = maxTs;
          }
        }
      } catch { /* skip */ }
    }
    poll();
    const id = setInterval(poll, POLL_OCCURRENCES_MS);
    return () => clearInterval(id);
  }, []);

  const actions = {
    selectSuspect: (id) => dispatch({ type: 'SELECT_SUSPECT', payload: id }),
    deleteSuspect: async (id) => {
      try {
        await CrimeDetectorAPI.deleteSuspect(id);
        dispatch({ type: 'DELETE_SUSPECT', payload: id });
        pushAlert('Suspect removed', 'success');
      } catch (e) {
        pushAlert(`Failed to remove suspect: ${e.message}`, 'error');
      }
    },
    dismissAlert: (id) => dispatch({ type: 'DISMISS_ALERT', payload: id }),
    pushAlert,
    dispatch,
  };

  return (
    <AppContext.Provider value={{ state, actions }}>
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useApp must be used inside AppProvider');
  return ctx;
}
