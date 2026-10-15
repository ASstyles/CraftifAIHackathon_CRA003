import React from 'react';
import { PerformanceMetrics } from '../types';
import { Zap, Clock, HardDrive, AlertOctagon, Thermometer } from 'lucide-react';

interface PerformanceGaugesProps {
  metrics: PerformanceMetrics | null;
  totalIncidents: number;
  activeIntrusions: number;
}

export const PerformanceGauges: React.FC<PerformanceGaugesProps> = ({
  metrics,
  totalIncidents,
  activeIntrusions
}) => {
  const fps = metrics?.fps ?? null;
  const inferenceLatency = metrics?.inference_latency_ms ?? null;
  const ingestLatency = metrics?.ingest_latency_ms ?? null;
  const vramUsed = metrics?.gpu_memory_used_mb ?? null;
  const vramTotal = metrics?.gpu_memory_total_mb ?? 4096.0;
  const gpuTemp = metrics?.gpu_temperature_c ?? null;
  const gpuUtil = metrics?.gpu_utilization_pct ?? null;
  const isStale = metrics?.is_stale ?? false;
  const isDemo = metrics?.mode === 'demo';
  const backend = metrics?.inference_backend ?? (isDemo ? 'DEMO_SYNTHETIC' : 'UNINITIALIZED');

  const vramPercent = vramUsed !== null ? Math.min(100, Math.round((vramUsed / vramTotal) * 100)) : 0;

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
      gap: '14px',
      marginBottom: '20px'
    }}>

      {/* 1. FPS Throughput Gauge */}
      <div className="glass-panel" style={{ padding: '16px', position: 'relative' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
          <span style={{ fontSize: '0.8rem', color: '#94a3b8', fontWeight: 500 }}>Throughput (Rolling)</span>
          <Zap size={16} color="#06b6d4" />
        </div>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '6px' }}>
          <span className="font-mono" style={{ fontSize: '1.8rem', fontWeight: 700, color: fps !== null ? '#38bdf8' : '#64748b' }}>
            {fps !== null ? fps.toFixed(1) : 'N/A'}
          </span>
          <span style={{ fontSize: '0.85rem', color: '#64748b' }}>FPS</span>
        </div>
        <div style={{ marginTop: '8px', fontSize: '0.72rem', display: 'flex', alignItems: 'center', gap: '4px' }}>
          {isDemo ? (
            <span style={{ color: '#f59e0b' }}>● Synthetic Demo Fixture</span>
          ) : isStale ? (
            <span style={{ color: '#ef4444' }}>● Stale / Idle (&gt;5s)</span>
          ) : fps !== null ? (
            <span style={{ color: '#10b981' }}>● Measured Rolling ({metrics?.metrics_source || 'live'})</span>
          ) : (
            <span style={{ color: '#64748b' }}>● Awaiting frame ingestion</span>
          )}
        </div>
      </div>

      {/* 2. Measured Inference Latency Gauge */}
      <div className="glass-panel" style={{ padding: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
          <span style={{ fontSize: '0.8rem', color: '#94a3b8', fontWeight: 500 }}>Inference Latency</span>
          <Clock size={16} color="#a855f7" />
        </div>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '6px' }}>
          <span className="font-mono" style={{ fontSize: '1.8rem', fontWeight: 700, color: inferenceLatency !== null ? '#c084fc' : '#64748b' }}>
            {inferenceLatency !== null ? inferenceLatency.toFixed(1) : 'N/A'}
          </span>
          <span style={{ fontSize: '0.85rem', color: '#64748b' }}>ms</span>
        </div>
        <div style={{ marginTop: '8px', fontSize: '0.72rem', color: '#94a3b8' }}>
          {ingestLatency !== null ? (
            <span>Ingest: <strong className="font-mono">{ingestLatency.toFixed(2)} ms</strong> | Target: 14.2 ms</span>
          ) : (
            <span>Target: <strong className="font-mono">14.2 ms</strong> (RTX 2050 TRT)</span>
          )}
        </div>
      </div>

      {/* 3. GPU VRAM Consumption */}
      <div className="glass-panel" style={{ padding: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
          <span style={{ fontSize: '0.8rem', color: '#94a3b8', fontWeight: 500 }}>RTX 2050 VRAM</span>
          <HardDrive size={16} color="#10b981" />
        </div>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '6px' }}>
          <span className="font-mono" style={{ fontSize: '1.8rem', fontWeight: 700, color: vramUsed !== null ? '#34d399' : '#64748b' }}>
            {vramUsed !== null ? Math.round(vramUsed) : 'N/A'}
          </span>
          <span style={{ fontSize: '0.85rem', color: '#64748b' }}>/ {Math.round(vramTotal)} MB</span>
        </div>
        {/* Progress Bar */}
        <div style={{ width: '100%', height: '5px', backgroundColor: 'rgba(255, 255, 255, 0.08)', borderRadius: '3px', marginTop: '10px', overflow: 'hidden' }}>
          <div style={{
            width: `${vramPercent}%`,
            height: '100%',
            backgroundColor: vramPercent > 85 ? '#ef4444' : '#10b981',
            borderRadius: '3px'
          }} />
        </div>
      </div>

      {/* 4. Active Intrusions Violations Counter */}
      <div className={`glass-panel ${activeIntrusions > 0 ? 'hazard-pulse' : ''}`} style={{
        padding: '16px',
        border: activeIntrusions > 0 ? '1px solid #ef4444' : '1px solid var(--border-subtle)',
        background: activeIntrusions > 0 ? 'rgba(239, 68, 68, 0.12)' : 'var(--bg-glass)'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
          <span style={{ fontSize: '0.8rem', color: activeIntrusions > 0 ? '#f87171' : '#94a3b8', fontWeight: 600 }}>
            Active Intrusions
          </span>
          <AlertOctagon size={16} color={activeIntrusions > 0 ? '#ef4444' : '#64748b'} />
        </div>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '6px' }}>
          <span className="font-mono" style={{
            fontSize: '1.8rem',
            fontWeight: 800,
            color: activeIntrusions > 0 ? '#f87171' : '#f1f5f9'
          }}>
            {activeIntrusions}
          </span>
          <span style={{ fontSize: '0.82rem', color: '#64748b' }}>violations</span>
        </div>
        <div style={{ marginTop: '8px', fontSize: '0.72rem', color: '#94a3b8' }}>
          Total Logged: <strong className="font-mono">{totalIncidents}</strong> incidents
        </div>
      </div>

      {/* 5. GPU Load & Temperature */}
      <div className="glass-panel" style={{ padding: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
          <span style={{ fontSize: '0.8rem', color: '#94a3b8', fontWeight: 500 }}>GPU Core Temp</span>
          <Thermometer size={16} color="#f59e0b" />
        </div>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '6px' }}>
          <span className="font-mono" style={{ fontSize: '1.8rem', fontWeight: 700, color: gpuTemp !== null ? '#fbbf24' : '#64748b' }}>
            {gpuTemp !== null ? gpuTemp.toFixed(0) : 'N/A'}
          </span>
          <span style={{ fontSize: '0.85rem', color: '#64748b' }}>{gpuTemp !== null ? '°C' : ''}</span>
        </div>
        <div style={{ marginTop: '8px', fontSize: '0.72rem', color: '#94a3b8' }}>
          {gpuUtil !== null ? (
            <span>Load: <strong className="font-mono">{gpuUtil.toFixed(0)}%</strong> | {backend}</span>
          ) : (
            <span>Backend: <strong className="font-mono">{backend}</strong></span>
          )}
        </div>
      </div>

    </div>
  );
};
