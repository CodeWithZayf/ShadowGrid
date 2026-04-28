// src/components/panels/EmbeddingChart.jsx
import React, { useEffect, useRef } from 'react';

export default function EmbeddingChart({ embedding }) {
  const canvasRef = useRef();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !embedding?.length) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const arr = embedding.slice(0, 512);
    const min = Math.min(...arr);
    const max = Math.max(...arr);
    const range = max - min || 1;
    const step = W / arr.length;
    const padT = 10, padB = 10;
    const drawH = H - padT - padB;
    const zeroY = padT + drawH * (1 - (0 - min) / range);

    // Grid
    ctx.strokeStyle = '#1e2d1e';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = padT + (drawH / 4) * i;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }

    // Zero line
    ctx.strokeStyle = '#2d4a2d';
    ctx.setLineDash([5, 5]);
    ctx.beginPath(); ctx.moveTo(0, zeroY); ctx.lineTo(W, zeroY); ctx.stroke();
    ctx.setLineDash([]);

    // Fill
    const grad = ctx.createLinearGradient(0, padT, 0, H);
    grad.addColorStop(0, 'rgba(34,211,238,0.4)');
    grad.addColorStop(1, 'rgba(34,211,238,0)');
    ctx.beginPath();
    arr.forEach((v, i) => {
      const x = i * step + step / 2;
      const y = padT + drawH * (1 - (v - min) / range);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.lineTo((arr.length - 1) * step + step / 2, H);
    ctx.lineTo(step / 2, H);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Line
    ctx.strokeStyle = '#22d3ee';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    arr.forEach((v, i) => {
      const x = i * step + step / 2;
      const y = padT + drawH * (1 - (v - min) / range);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  }, [embedding]);

  if (!embedding?.length) return <div className="empty-note">No embedding</div>;

  const arr = embedding;
  const min = Math.min(...arr).toFixed(4);
  const max = Math.max(...arr).toFixed(4);
  const mean = (arr.reduce((a, b) => a + b, 0) / arr.length).toFixed(4);
  const norm = Math.sqrt(arr.reduce((a, b) => a + b * b, 0)).toFixed(4);

  return (
    <div className="embed-wrap">
      <canvas ref={canvasRef} width={560} height={100} className="embed-canvas" />
      <div className="embed-stats">
        <span>DIM {arr.length}</span>
        <span>MIN {min}</span>
        <span>MAX {max}</span>
        <span>MEAN {mean}</span>
        <span>L2 {norm}</span>
      </div>
    </div>
  );
}
