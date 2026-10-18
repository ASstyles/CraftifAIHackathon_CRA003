import React, { useState } from 'react';
import { SafetyIncident } from '../types';
import { ShieldAlert, Download, Code, ChevronDown, ChevronUp } from 'lucide-react';

interface IncidentLogProps {
  incidents: SafetyIncident[];
}

export const IncidentLog: React.FC<IncidentLogProps> = ({ incidents }) => {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const handleExportJSON = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(incidents, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `sentinelvision_incidents_${Date.now()}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  return (
    <div className="glass-panel" style={{ padding: '16px', display: 'flex', flexDirection: 'column', height: '100%' }}>
      
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
          <ShieldAlert size={18} color="#ef4444" />
          <h2 style={{ fontSize: '1rem', fontWeight: 600, color: '#f8fafc' }}>
            Incident Audit Log
          </h2>
          <span className="badge badge-danger" style={{ fontSize: '0.7rem' }}>
            {incidents.length} Events
          </span>
        </div>

        <button
          id="btn-export-incidents"
          onClick={handleExportJSON}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            padding: '5px 10px',
            borderRadius: '6px',
            background: 'rgba(255, 255, 255, 0.05)',
            border: '1px solid var(--border-subtle)',
            color: '#cbd5e1',
            fontSize: '0.75rem',
            fontWeight: 500
          }}
        >
          <Download size={13} />
          <span>Export JSON</span>
        </button>
      </div>

      {/* Incident List */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        display: 'flex',
        flexDirection: 'column',
        gap: '10px',
        maxHeight: '480px',
        paddingRight: '4px'
      }}>
        {incidents.length === 0 ? (
          <div style={{
            textAlign: 'center',
            padding: '40px 20px',
            color: '#64748b',
            fontSize: '0.85rem'
          }}>
            <ShieldAlert size={32} style={{ margin: '0 auto 12px', opacity: 0.3 }} />
            No restricted zone violations recorded.
            <br />
            Personnel are operating in designated safe corridors.
          </div>
        ) : (
          incidents.map((inc, index) => {
            const itemKey = inc.incident_id || `${inc.zone_id}_${inc.track_id}_${index}`;
            const isExpanded = expandedId === itemKey;
            const timeFormatted = new Date(inc.timestamp).toLocaleTimeString();

            return (
              <div
                key={itemKey}
                style={{
                  background: 'rgba(15, 23, 42, 0.65)',
                  border: inc.is_update ? '1px solid rgba(245, 158, 11, 0.35)' : '1px solid rgba(239, 68, 68, 0.25)',
                  borderRadius: '8px',
                  padding: '12px',
                  transition: 'all 0.2s ease'
                }}
              >
                {/* Primary Row */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '10px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{
                      width: '8px',
                      height: '8px',
                      borderRadius: '50%',
                      backgroundColor: inc.is_update ? '#f59e0b' : '#ef4444'
                    }} />
                    <span style={{ fontWeight: 600, fontSize: '0.85rem', color: inc.is_update ? '#fbbf24' : '#f87171' }}>
                      {inc.zone_name || inc.zone_id}
                    </span>
                    <span className="badge badge-danger" style={{ fontSize: '0.65rem', padding: '2px 6px' }}>
                      {inc.risk_level}
                    </span>
                    {inc.is_update && (
                      <span style={{
                        fontSize: '0.65rem',
                        padding: '1px 6px',
                        borderRadius: '4px',
                        background: 'rgba(245, 158, 11, 0.15)',
                        color: '#fbbf24',
                        border: '1px solid rgba(245, 158, 11, 0.3)'
                      }}>
                        Update #{inc.update_seq || 1}
                      </span>
                    )}
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span className="font-mono" style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                      {timeFormatted}
                    </span>
                    <button
                      onClick={() => setExpandedId(isExpanded ? null : itemKey)}
                      style={{
                        background: 'transparent',
                        border: 'none',
                        color: '#64748b',
                        padding: '2px',
                        display: 'flex'
                      }}
                      title="Inspect Structured JSON"
                    >
                      {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                    </button>
                  </div>
                </div>

                {/* Sub-info Row */}
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginTop: '8px',
                  fontSize: '0.75rem',
                  color: '#94a3b8'
                }}>
                  <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                    <span>Target: <strong style={{ color: '#cbd5e1' }}>Person #{inc.track_id || 1}</strong></span>
                    <span>Confidence: <strong className="font-mono" style={{ color: '#38bdf8' }}>{(inc.confidence * 100).toFixed(1)}%</strong></span>
                  </div>
                  <div>
                    Dwell: <strong className="font-mono" style={{ color: '#fbbf24' }}>{inc.dwell_time_seconds.toFixed(1)}s</strong>
                  </div>
                </div>

                {/* Structured JSON Drawer (Track A Schema) */}
                {isExpanded && (
                  <div style={{
                    marginTop: '10px',
                    padding: '10px',
                    backgroundColor: '#05070e',
                    borderRadius: '6px',
                    border: '1px solid rgba(255, 255, 255, 0.08)',
                    fontSize: '0.72rem'
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px', color: '#64748b' }}>
                      <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <Code size={12} /> PipeGen Actionable JSON Payload
                      </span>
                    </div>
                    <pre className="font-mono" style={{ color: '#38bdf8', overflowX: 'auto', margin: 0 }}>
                      {JSON.stringify(inc, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

    </div>
  );
};
