import React from 'react';
import { PipelineStatus } from '../types';
import { ShieldAlert, Cpu, Activity, Video, Award, RefreshCw } from 'lucide-react';

interface HeaderProps {
  status: PipelineStatus | null;
  connected: boolean;
  onToggleMode: () => void;
  onOpenEvidence: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  status,
  connected,
  onToggleMode,
  onOpenEvidence
}) => {
  const isLive = status?.mode === 'live';

  return (
    <header className="glass-panel" style={{ padding: '16px 24px', marginBottom: '20px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
        
        {/* Brand & Title */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{
            width: '44px',
            height: '44px',
            borderRadius: '10px',
            background: 'linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 20px rgba(6, 182, 212, 0.4)'
          }}>
            <ShieldAlert size={26} color="#ffffff" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <h1 style={{ fontSize: '1.45rem', fontWeight: '700', letterSpacing: '-0.02em', color: '#f8fafc' }}>
                SentinelVision
              </h1>
              <span className="badge badge-cyan" style={{ fontSize: '0.7rem' }}>
                Track A: PipeGen
              </span>
            </div>
            <p style={{ fontSize: '0.82rem', color: '#94a3b8', marginTop: '2px' }}>
              AI-Powered Industrial Safety & Restricted Zone Intrusion Monitoring
            </p>
          </div>
        </div>

        {/* Telemetry Pills & Actions */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
          
          {/* Connection Status */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '6px 12px',
            background: 'rgba(255, 255, 255, 0.04)',
            borderRadius: '8px',
            border: '1px solid var(--border-subtle)',
            fontSize: '0.8rem'
          }}>
            <span style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: connected ? '#10b981' : '#ef4444'
            }} className={connected ? 'live-indicator' : ''} />
            <span style={{ color: connected ? '#cbd5e1' : '#f87171', fontWeight: 500 }}>
              {connected ? 'Backend Online' : 'Backend Disconnected'}
            </span>
          </div>

          {/* GPU Telemetry Pill */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '6px 12px',
            background: 'rgba(6, 182, 212, 0.08)',
            borderRadius: '8px',
            border: '1px solid rgba(6, 182, 212, 0.25)',
            fontSize: '0.8rem'
          }}>
            <Cpu size={15} color="#06b6d4" />
            <span style={{ color: '#e2e8f0', fontWeight: 500 }}>
              RTX 2050 (4GB)
            </span>
            <span className="font-mono" style={{ color: '#38bdf8', fontSize: '0.75rem' }}>
              {status?.gpu?.temperature_c ? `${status.gpu.temperature_c}°C` : '54°C'}
            </span>
          </div>

          {/* PipeGen Core Indicator */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '6px 12px',
            background: 'rgba(59, 130, 246, 0.08)',
            borderRadius: '8px',
            border: '1px solid rgba(59, 130, 246, 0.25)',
            fontSize: '0.8rem'
          }}>
            <Activity size={15} color="#60a5fa" />
            <span style={{ color: '#cbd5e1' }}>PipeGen Core:</span>
            <span className="font-mono" style={{ color: '#93c5fd', fontWeight: 600 }}>v0.5.1</span>
          </div>

          {/* Mode Switcher */}
          <button
            id="btn-toggle-mode"
            onClick={onToggleMode}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '7px 14px',
              borderRadius: '8px',
              border: isLive ? '1px solid #10b981' : '1px solid #f59e0b',
              background: isLive ? 'rgba(16, 185, 129, 0.15)' : 'rgba(245, 158, 11, 0.15)',
              color: isLive ? '#34d399' : '#fbbf24',
              fontWeight: 600,
              fontSize: '0.8rem'
            }}
          >
            <Video size={15} />
            <span>{isLive ? 'LIVE GPU PIPELINE' : 'SAMPLE DEMO FIXTURE'}</span>
            <RefreshCw size={12} style={{ opacity: 0.7 }} />
          </button>

          {/* Track A Submission Evidence Button */}
          <button
            id="btn-open-evidence"
            onClick={onOpenEvidence}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '7px 16px',
              borderRadius: '8px',
              border: '1px solid rgba(6, 182, 212, 0.4)',
              background: 'linear-gradient(135deg, rgba(6, 182, 212, 0.2) 0%, rgba(59, 130, 246, 0.2) 100%)',
              color: '#38bdf8',
              fontWeight: 600,
              fontSize: '0.8rem',
              boxShadow: '0 0 15px rgba(6, 182, 212, 0.2)'
            }}
          >
            <Award size={16} />
            <span>Track A Evidence</span>
          </button>

        </div>
      </div>
    </header>
  );
};
