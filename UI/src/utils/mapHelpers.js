// src/utils/mapHelpers.js
// Helpers for drawing on Leaflet Canvas overlays.
// All drawing functions accept a Leaflet map instance and raw lat/lon data.

/**
 * Convert [lat, lon] to Leaflet container pixel point.
 * @param {L.Map} map
 * @param {number} lat
 * @param {number} lon
 * @returns {{ x: number, y: number }}
 */
export function latLonToPixel(map, lat, lon) {
  const pt = map.latLngToContainerPoint([lat, lon]);
  return { x: pt.x, y: pt.y };
}

/**
 * Project meters to pixels at the current zoom.
 * Uses Leaflet's internal CRS distance → pixel ratio.
 */
export function metersToPixels(map, lat, meters) {
  const zoom = map.getZoom();
  // Leaflet: 1 pixel ≈ resolution at given lat/zoom
  // resolution = 40075016.686 * Math.cos(lat * π/180) / (256 * 2^zoom)
  const res =
    (40075016.686 * Math.cos((lat * Math.PI) / 180)) /
    (256 * Math.pow(2, zoom));
  return meters / res;
}

/**
 * Return a CSS hex color with given opacity (0–1) blended on black.
 * @param {string} hex  '#rrggbb'
 * @param {number} a    0–1
 */
export function hexAlpha(hex, a) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${a})`;
}
