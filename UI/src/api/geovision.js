// src/api/geovision.js
// REMOVED — GeoVision C++ engine has been replaced by the Python prediction engine.
// This file is kept as a stub to prevent import errors during migration.
// TODO (Phase 3): Replace with calls to the Python prediction/uncertainty endpoints.

export const GeoVisionAPI = {
  health:              () => Promise.reject(new Error('GeoVision removed')),
  snapshot:            () => Promise.reject(new Error('GeoVision removed')),
  positions:           () => Promise.reject(new Error('GeoVision removed')),
  escapeProbabilities: () => Promise.reject(new Error('GeoVision removed')),
  deploymentOrders:    () => Promise.reject(new Error('GeoVision removed')),
  embeddingMatches:    () => Promise.reject(new Error('GeoVision removed')),
};
