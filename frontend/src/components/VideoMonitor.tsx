import React, { useState } from 'react';
import { RestrictedZone } from '../types';
import { AlertTriangle, Eye, EyeOff, Shield, Radio } from 'lucide-react';

interface VideoMonitorProps {
  mode: 'live' | 'demo';
  activeIntrusions: number;
  zones: RestrictedZone[];
  latestZoneName?: string;
}

export const VideoMonitor: React.FC<VideoMonitorProps> = ({
  mode,
  activeIntrusions,
  zones,
  latestZoneName
}) => {
  const [showZones, setShowZones] = useState(true);

  const hasIntrusion = activeIntrusions > 0;

  return (
    <div className="glass-panel" style={{ padding: '16px', position: 'relative', overflow: 'hidden' }}>
      
      {/* Top Banner: Camera Header & Controls */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '12px',
        paddingBottom: '10px',
        borderBottom: '1px solid var(--border-subtle)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            width: '10px',
            height: '10px',
            borderRadius: '50%',
            backgroundColor: '#10b981'
          }} className="live-indicator" />
          <span style={{ fontWeight: 600, fontSize: '0.95rem', color: '#f1f5f9' }}>
            Primary Safety Optical Stream
          </span>
          <span className="font-mono" style={{ fontSize: '0.75rem', color: '#64748b' }}>
            [CAM-01 / 1280x720 @ 30 FPS]
          </span>
        </div>

        {/* Stream Actions */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <button
            id="btn-toggle-zones"
            onClick={() => setShowZones(!showZones)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '5px 10px',
              borderRadius: '6px',
              background: showZones ? 'rgba(6, 182, 212, 0.15)' : 'rgba(255, 255, 255, 0.05)',
              border: '1px solid ' + (showZones ? 'rgba(6, 182, 212, 0.3)' : 'var(--border-subtle)'),
              color: showZones ? '#38bdf8' : '#94a3b8',
              fontSize: '0.75rem',
              fontWeight: 500
            }}
          >
            {showZones ? <Eye size={14} /> : <EyeOff size={14} />}
            <span>Zones ({zones.length})</span>
          </button>
        </div>
      </div>

      {/* Video Container */}
      <div style={{
        position: 'relative',
        width: '100%',
        aspectRatio: '16 / 9',
        backgroundColor: '#05070e',
        borderRadius: '8px',
        overflow: 'hidden',
        border: hasIntrusion ? '2px solid #ef4444' : '1px solid var(--border-subtle)',
        boxShadow: hasIntrusion ? '0 0 25px rgba(239, 68, 68, 0.35)' : 'none',
        transition: 'all 0.3s ease'
      }}>

        {/* Video Feed Image from FastAPI */}
        <img
          id="video-feed"
          src="/api/stream/feed"
          alt="SentinelVision Live Safety Video Feed"
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            display: 'block'
          }}
          onError={(e) => {
            // Graceful fallback if backend stream isn't initialized yet
            const target = e.target as HTMLImageElement;
            target.style.display = 'none';
          }}
        />

        {/* Active Intrusion Critical Warning Header */}
        {hasIntrusion && (
          <div className="hazard-pulse" style={{
            position: 'absolute',
            top: '12px',
            left: '12px',
            right: '12px',
            padding: '10px 16px',
            background: 'linear-gradient(90deg, rgba(239, 68, 68, 0.92) 0%, rgba(185, 28, 28, 0.95) 100%)',
            borderRadius: '8px',
            color: '#ffffff',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            zIndex: 20
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <AlertTriangle size={20} />
              <div>
                <div style={{ fontWeight: '800', fontSize: '0.85rem', letterSpacing: '0.05em' }}>
                  SAFETY VIOLATION DETECTED
                </div>
                <div style={{ fontSize: '0.75rem', opacity: 0.9 }}>
                  Unauthorized personnel inside: {latestZoneName || 'Robotic Arm Perimeter'}
                </div>
              </div>
            </div>
            <span className="badge" style={{ background: '#ffffff', color: '#dc2626', fontWeight: '800' }}>
              CRITICAL HAZARD
            </span>
          </div>
        )}

        {/* Mandatory Hackathon Mode Watermark */}
        <div style={{
          position: 'absolute',
          bottom: '12px',
          left: '12px',
          zIndex: 10,
          display: 'flex',
          gap: '8px',
          alignItems: 'center'
        }}>
          <span className={`badge ${mode === 'live' ? 'badge-live' : 'badge-demo'}`}>
            <Radio size={12} />
            {mode === 'live' ? 'LIVE INFERENCE: TENSORRT ENGINE' : 'SAMPLE/DEMO FIXTURE: VERIFIED REPRODUCIBLE RUN'}
          </span>
        </div>

        {/* Model & PipeGen Overlay Tag */}
        <div style={{
          position: 'absolute',
          bottom: '12px',
          right: '12px',
          zIndex: 10,
          background: 'rgba(7, 10, 19, 0.85)',
          backdropFilter: 'blur(8px)',
          padding: '4px 10px',
          borderRadius: '6px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          fontSize: '0.72rem',
          color: '#94a3b8',
          display: 'flex',
          gap: '6px',
          alignItems: 'center'
        }}>
          <Shield size={12} color="#06b6d4" />
          <span>CraftifAI PipeGen IR Engine</span>
        </div>

      </div>

      {/* Video Details Footnote */}
      <div style={{
        marginTop: '10px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        fontSize: '0.75rem',
        color: '#64748b'
      }}>
        <span>Floor Plane Projection: Bottom-center foot contact point evaluated via Ray-Casting</span>
        <span>Suppression: Duplicate events throttled during dwell period</span>
      </div>

    </div>
  );
};
