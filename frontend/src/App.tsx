import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { VideoMonitor } from './components/VideoMonitor';
import { PerformanceGauges } from './components/PerformanceGauges';
import { IncidentLog } from './components/IncidentLog';
import { ZoneEditor } from './components/ZoneEditor';
import { TrackAEvidenceModal } from './components/TrackAEvidenceModal';
import { PipelineStatus, PerformanceMetrics, SafetyIncident, RestrictedZone } from './types';

export const App: React.FC = () => {
  const [status, setStatus] = useState<PipelineStatus | null>(null);
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [incidents, setIncidents] = useState<SafetyIncident[]>([]);
  const [zones, setZones] = useState<RestrictedZone[]>([]);
  const [connected, setConnected] = useState<boolean>(false);
  const [isEvidenceOpen, setIsEvidenceOpen] = useState<boolean>(false);

  // Poll system status and metrics
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [statusRes, metricsRes, incidentsRes, zonesRes] = await Promise.all([
          fetch('/api/status'),
          fetch('/api/metrics'),
          fetch('/api/incidents?limit=50'),
          fetch('/api/zones')
        ]);

        if (statusRes.ok && metricsRes.ok) {
          const statusData = await statusRes.json();
          const metricsData = await metricsRes.json();
          setStatus(statusData);
          setMetrics(metricsData);
          setConnected(true);
        }

        if (incidentsRes.ok) {
          const incData = await incidentsRes.json();
          setIncidents(incData);
        }

        if (zonesRes.ok) {
          const zoneData = await zonesRes.json();
          setZones(zoneData);
        }
      } catch (err) {
        setConnected(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 1200);
    return () => clearInterval(interval);
  }, []);

  // Server-Sent Events (SSE) for instant incident push
  useEffect(() => {
    let evtSource: EventSource | null = null;
    try {
      evtSource = new EventSource('/api/stream/events');
      evtSource.addEventListener('incident', (event: MessageEvent) => {
        try {
          const newInc: SafetyIncident = JSON.parse(event.data);
          setIncidents((prev) => [newInc, ...prev.slice(0, 99)]);
        } catch (e) {
          console.error('Failed to parse incident SSE', e);
        }
      });
    } catch (e) {
      console.warn('SSE not supported or connection failed', e);
    }

    return () => {
      if (evtSource) {
        evtSource.close();
      }
    };
  }, []);

  // Handle Mode Toggle
  const handleToggleMode = async () => {
    const nextMode = status?.mode === 'live' ? 'demo' : 'live';
    try {
      const res = await fetch('/api/mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: nextMode })
      });
      if (res.ok) {
        const updated = await fetch('/api/status');
        if (updated.ok) {
          setStatus(await updated.json());
        }
      }
    } catch (e) {
      console.error('Failed to toggle pipeline mode', e);
    }
  };

  // Add preset zone
  const handleAddPresetZone = async (preset: 'loading_dock' | 'conveyor') => {
    const newZone = preset === 'loading_dock' ? {
      zone_id: `zone_loading_${Date.now() % 1000}`,
      name: 'Loading Bay Forklift Zone',
      polygon: [[950.0, 200.0], [1250.0, 200.0], [1250.0, 600.0], [950.0, 600.0]],
      color: '#F59E0B',
      risk_level: 'HIGH',
      cooldown_seconds: 3.5
    } : {
      zone_id: `zone_conveyor_${Date.now() % 1000}`,
      name: 'High-Speed Conveyor Belt',
      polygon: [[300.0, 500.0], [700.0, 500.0], [700.0, 680.0], [300.0, 680.0]],
      color: '#EF4444',
      risk_level: 'CRITICAL',
      cooldown_seconds: 4.0
    };

    try {
      const res = await fetch('/api/zones', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newZone)
      });
      if (res.ok) {
        const updated = await fetch('/api/zones');
        if (updated.ok) setZones(await updated.json());
      }
    } catch (e) {
      console.error('Failed to add zone', e);
    }
  };

  // Delete zone
  const handleDeleteZone = async (zoneId: string) => {
    try {
      const res = await fetch(`/api/zones/${zoneId}`, { method: 'DELETE' });
      if (res.ok) {
        setZones(zones.filter((z) => z.zone_id !== zoneId));
      }
    } catch (e) {
      console.error('Failed to delete zone', e);
    }
  };

  const activeIntrusions = status?.active_intrusions ?? 0;
  const totalIncidents = status?.total_incidents ?? incidents.length;

  return (
    <div style={{ maxWidth: '1600px', margin: '0 auto', padding: '24px 20px' }}>
      
      {/* Top Header */}
      <Header
        status={status}
        connected={connected}
        onToggleMode={handleToggleMode}
        onOpenEvidence={() => setIsEvidenceOpen(true)}
      />

      {/* Real-time Performance Gauges */}
      <PerformanceGauges
        metrics={metrics}
        totalIncidents={totalIncidents}
        activeIntrusions={activeIntrusions}
      />

      {/* Main Grid: Video Stream + Zone Management + Incident Feed */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'minmax(0, 1.8fr) minmax(360px, 1fr)',
        gap: '20px',
        alignItems: 'start'
      }}>
        
        {/* Left Column: Video Monitor + Zone Configuration */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <VideoMonitor
            mode={status?.mode ?? 'demo'}
            activeIntrusions={activeIntrusions}
            zones={zones}
            latestZoneName={incidents[0]?.zone_name}
          />
          <ZoneEditor
            zones={zones}
            onAddPresetZone={handleAddPresetZone}
            onDeleteZone={handleDeleteZone}
          />
        </div>

        {/* Right Column: Incident Audit Log */}
        <div style={{ height: '100%' }}>
          <IncidentLog incidents={incidents} />
        </div>

      </div>

      {/* Track A Submission Verification Modal */}
      <TrackAEvidenceModal
        isOpen={isEvidenceOpen}
        onClose={() => setIsEvidenceOpen(false)}
      />

    </div>
  );
};
