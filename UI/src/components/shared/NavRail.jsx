// src/components/shared/NavRail.jsx
import React from 'react';

const TABS = [
  { id: 'map',          icon: '⊕', label: 'MAP' },
  { id: 'suspects',     icon: '◈', label: 'SUSPECTS' },
  { id: 'occurrences',  icon: '◉', label: 'OCCURRENCES' },
  { id: 'cameras',      icon: '⬡', label: 'CAMERAS' },
  { id: 'intelligence', icon: '⬘', label: 'INTEL' },
];

export default function NavRail({ activeTab, onTabChange }) {
  return (
    <nav className="navrail">
      {TABS.map(t => (
        <button
          key={t.id}
          className={`nav-btn ${activeTab === t.id ? 'nav-active' : ''}`}
          onClick={() => onTabChange(t.id)}
          title={t.label}
        >
          <span className="nav-icon">{t.icon}</span>
          <span className="nav-label">{t.label}</span>
        </button>
      ))}
    </nav>
  );
}
