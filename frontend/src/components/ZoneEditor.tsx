import React from 'react';
import { RestrictedZone } from '../types';
import { MapPin, Plus, Trash2, CheckCircle2 } from 'lucide-react';

interface ZoneEditorProps {
  zones: RestrictedZone[];
  onAddPresetZone: (preset: 'loading_dock' | 'conveyor') => void;
  onDeleteZone: (zoneId: string) => void;
}

export const ZoneEditor: React.FC<ZoneEditorProps> = ({
  zones,
  onAddPresetZone,
  onDeleteZone
}) => {
  return (
    <div className="glass-panel" style={{ padding: '16px', height: '100%' }}>
      
      {/* Header */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '14px',
        paddingBottom: '10px',
        borderBottom: '1px solid var(--border-subtle)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <MapPin size={18} color="#06b6d4" />
          <h2 style={{ fontSize: '1rem', fontWeight: 600, color: '#f8fafc' }}>
            Restricted Hazard Zones
          </h2>
        </div>

        {/* Quick Add Presets */}
        <div style={{ display: 'flex', gap: '6px' }}>
          <button
            id="btn-add-loading-dock"
            onClick={() => onAddPresetZone('loading_dock')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
              padding: '4px 8px',
              borderRadius: '6px',
              background: 'rgba(6, 182, 212, 0.1)',
              border: '1px solid rgba(6, 182, 212, 0.3)',
              color: '#38bdf8',
              fontSize: '0.72rem',
              fontWeight: 500
            }}
          >
            <Plus size={12} />
            <span>+ Loading Dock</span>
          </button>
        </div>
      </div>

      {/* Zone Cards */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {zones.map((zone) => (
          <div
            key={zone.zone_id}
            style={{
              padding: '12px',
              background: 'rgba(15, 23, 42, 0.6)',
              borderRadius: '8px',
              border: `1px solid ${zone.risk_level === 'CRITICAL' ? 'rgba(239, 68, 68, 0.3)' : 'rgba(245, 158, 11, 0.3)'}`,
              display: 'flex',
              flexDirection: 'column',
              gap: '6px'
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{
                  width: '10px',
                  height: '10px',
                  borderRadius: '2px',
                  backgroundColor: zone.color
                }} />
                <span style={{ fontWeight: 600, fontSize: '0.85rem', color: '#f1f5f9' }}>
                  {zone.name}
                </span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span className={`badge ${zone.risk_level === 'CRITICAL' ? 'badge-danger' : 'badge-demo'}`} style={{ fontSize: '0.65rem' }}>
                  {zone.risk_level}
                </span>
                {zones.length > 1 && (
                  <button
                    onClick={() => onDeleteZone(zone.zone_id)}
                    style={{ background: 'transparent', border: 'none', color: '#64748b', padding: '2px' }}
                    title="Remove Zone"
                  >
                    <Trash2 size={13} />
                  </button>
                )}
              </div>
            </div>

            {/* Coordinates info */}
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: '#94a3b8' }}>
              <span>Vertices: <strong className="font-mono" style={{ color: '#cbd5e1' }}>{zone.polygon.length} points</strong></span>
              <span>Cooldown: <strong className="font-mono" style={{ color: '#cbd5e1' }}>{zone.cooldown_seconds}s</strong></span>
              <span style={{ color: zone.is_active ? '#10b981' : '#64748b', display: 'flex', alignItems: 'center', gap: '3px' }}>
                <CheckCircle2 size={11} /> Active
              </span>
            </div>
          </div>
        ))}
      </div>

    </div>
  );
};
