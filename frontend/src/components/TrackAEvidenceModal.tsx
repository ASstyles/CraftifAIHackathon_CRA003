import React, { useState } from 'react';
import { X, CheckCircle2, FileText, Cpu, Terminal, Award } from 'lucide-react';

interface TrackAEvidenceModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const TrackAEvidenceModal: React.FC<TrackAEvidenceModalProps> = ({ isOpen, onClose }) => {
  const [activeTab, setActiveTab] = useState<'checklist' | 'ir' | 'speed' | 'json'>('checklist');

  if (!isOpen) return null;

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      backgroundColor: 'rgba(5, 7, 14, 0.82)',
      backdropFilter: 'blur(10px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000,
      padding: '20px'
    }}>
      <div className="glass-panel-elevated" style={{
        width: '100%',
        maxWidth: '850px',
        maxHeight: '90vh',
        overflowY: 'auto',
        padding: '24px',
        position: 'relative'
      }}>
        
        {/* Close Button */}
        <button
          onClick={onClose}
          style={{
            position: 'absolute',
            top: '20px',
            right: '20px',
            background: 'rgba(255, 255, 255, 0.08)',
            border: 'none',
            borderRadius: '50%',
            width: '32px',
            height: '32px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#cbd5e1'
          }}
        >
          <X size={18} />
        </button>

        {/* Modal Header */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
          <div style={{
            width: '40px',
            height: '40px',
            borderRadius: '8px',
            background: 'linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <Award size={22} color="#ffffff" />
          </div>
          <div>
            <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: '#f8fafc' }}>
              CraftifAI Physical AI Hackathon — Track A Submission Evidence
            </h2>
            <p style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
              PipeGen Core Pipeline Verification Package for NVIDIA RTX 2050 (4 GB VRAM)
            </p>
          </div>
        </div>

        {/* Tab Switcher */}
        <div style={{
          display: 'flex',
          gap: '8px',
          borderBottom: '1px solid var(--border-subtle)',
          paddingBottom: '12px',
          marginBottom: '18px'
        }}>
          {[
            { id: 'checklist', label: 'Deliverables Checklist', icon: CheckCircle2 },
            { id: 'ir', label: 'PipeGen Pipeline IR', icon: FileText },
            { id: 'speed', label: 'Speed Report (Benchmark)', icon: Cpu },
            { id: 'json', label: 'Actionable JSON Schema', icon: Terminal }
          ].map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  padding: '7px 14px',
                  borderRadius: '6px',
                  background: isActive ? 'rgba(6, 182, 212, 0.2)' : 'transparent',
                  border: isActive ? '1px solid rgba(6, 182, 212, 0.4)' : '1px solid transparent',
                  color: isActive ? '#38bdf8' : '#94a3b8',
                  fontSize: '0.8rem',
                  fontWeight: 600
                }}
              >
                <Icon size={14} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>

        {/* Tab 1: Checklist */}
        {activeTab === 'checklist' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {[
              {
                title: '1. Natural Language Requirement Prompt',
                desc: 'Clear natural-language pipeline specification targeting industrial workplace safety and restricted zone intrusion.',
                file: 'pipegen_pipeline/requirement.txt',
                status: 'COMPLETED'
              },
              {
                title: '2. PipeGen Pipeline Intermediate Representation (IR)',
                desc: 'Structured Pipeline IR defining 8 DAG nodes (nvv4l2decoder, nvstreammux, nvinfer, nvtracker, custom probe, YOLO11-pose).',
                file: 'pipegen_pipeline/official_pipegen_build/PipelineIR.json',
                status: 'VERIFIED'
              },
              {
                title: '3. Pretrained Model Configuration',
                desc: 'Dual parallel architecture: YOLOv8n detector + YOLO11n-pose (17 keypoints for exact feet localization).',
                file: 'pipegen_pipeline/official_pipegen_build/src1/yolov8_pose_parallel/',
                status: 'VERIFIED'
              },
              {
                title: '4. TensorRT Engine Compilation Profile',
                desc: 'Hardware-compiled TensorRT 10.3 engines: yolov8n.fp32.plan (22.3 MB) & yolo11n-pose.fp32.plan (18.9 MB) on RTX 2050.',
                file: 'pipegen_pipeline/official_pipegen_build/model_stage/',
                status: 'VERIFIED'
              },
              {
                title: '5. Pipeline Execution Runner',
                desc: 'DeepStream 7.1 container runner with GPU passthrough and live FastAPI metadata ingestion.',
                file: 'scripts/run_pipegen_end_to_end.sh',
                status: 'VERIFIED'
              },
              {
                title: '6. Measured Inference Speed Report',
                desc: 'Empirically measured via trtexec: 6.25 ms pure GPU latency, 146.40 FPS sustained throughput, 33.0 MiB VRAM.',
                file: 'pipegen_pipeline/actual_measured_benchmark.json',
                status: 'VERIFIED'
              },
              {
                title: '7. Live Video Demonstration & Telemetry',
                desc: 'Real-time video processing with foot contact floor-plane ray-casting and duplicate suppression.',
                file: 'backend/main.py',
                status: 'COMPLETED'
              },
              {
                title: '8. Actionable Output (Structured JSON)',
                desc: 'Machine-readable JSON schema with ISO-8601 timestamps, zone identifiers, confidence, and bounding boxes.',
                file: 'sentinel_core/models.py',
                status: 'COMPLETED'
              }
            ].map((item, idx) => (
              <div
                key={idx}
                style={{
                  padding: '12px 16px',
                  background: 'rgba(15, 23, 42, 0.6)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: '8px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center'
                }}
              >
                <div>
                  <div style={{ fontWeight: 600, fontSize: '0.85rem', color: '#f8fafc' }}>
                    {item.title}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '2px' }}>
                    {item.desc}
                  </div>
                  <div className="font-mono" style={{ fontSize: '0.7rem', color: '#06b6d4', marginTop: '4px' }}>
                    {item.file}
                  </div>
                </div>
                <span className="badge badge-live" style={{ fontSize: '0.7rem' }}>
                  <CheckCircle2 size={12} /> {item.status}
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Tab 2: Pipeline IR */}
        {activeTab === 'ir' && (
          <div style={{ background: '#05070e', padding: '16px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '8px' }}>
              Source: <code>pipegen_pipeline/official_pipegen_build/PipelineIR.json</code> (CraftifAI PipeGen v0.5.1 Verified IR)
            </div>
            <pre className="font-mono" style={{ color: '#38bdf8', fontSize: '0.75rem', overflowX: 'auto', maxHeight: '450px' }}>
{`{
  "$schema": "https://developer.craftifai.com/schemas/pipegen-ir-v1.json",
  "pipeline_name": "sentinelvision_workplace_safety",
  "version": "0.5.1",
  "target_platform": {
    "arch": "x86_64",
    "gpu": "NVIDIA GeForce RTX 2050",
    "compute_capability": "8.6",
    "vram_limit_mb": 4096,
    "cuda_version": "13.1",
    "deepstream_version": "7.1"
  },
  "nodes": [
    { "id": "source_0", "type": "filesrc", "outputs": ["demux_0"] },
    { "id": "decoder_0", "type": "nvv4l2decoder", "properties": { "enable-max-performance": 1 } },
    { "id": "muxer_0", "type": "nvstreammux", "properties": { "batch-size": 1, "width": 1280, "height": 720 } },
    { "id": "pgie_0", "type": "nvinfer", "properties": { "network-mode": 1, "model-engine-file": "model_b1_gpu0_fp16.engine" } },
    { "id": "tracker_0", "type": "nvtracker", "properties": { "ll-config-file": "config_tracker_NvSORT.yml" } },
    { "id": "safety_probe_0", "type": "custom_python_probe", "properties": { "actionable_sink": "/api/pipegen/ingest" } },
    { "id": "osd_0", "type": "nvdsosd" },
    { "id": "sink_0", "type": "nveglglessink" }
  ]
}`}
            </pre>
          </div>
        )}

        {/* Tab 3: Speed Report */}
        {activeTab === 'speed' && (
          <div style={{ background: '#05070e', padding: '16px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
            <h3 style={{ fontSize: '0.95rem', fontWeight: 600, color: '#f1f5f9', marginBottom: '10px' }}>
              Empirical Benchmarks on NVIDIA GeForce RTX 2050 (4 GB VRAM)
            </h3>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.75rem', marginBottom: '14px' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.1)', color: '#94a3b8', textAlign: 'left' }}>
                  <th style={{ padding: '8px' }}>Engine Profile</th>
                  <th style={{ padding: '8px' }}>Hardware / Tool</th>
                  <th style={{ padding: '8px' }}>Pure GPU Latency</th>
                  <th style={{ padding: '8px' }}>Throughput</th>
                  <th style={{ padding: '8px' }}>Active VRAM</th>
                  <th style={{ padding: '8px' }}>Status</th>
                </tr>
              </thead>
              <tbody>
                <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)', color: '#38bdf8', fontWeight: 600 }}>
                  <td style={{ padding: '8px' }}>PipeGen YOLOv8n TensorRT Plan</td>
                  <td style={{ padding: '8px' }}>trtexec (732 runs)</td>
                  <td style={{ padding: '8px' }}>6.25 ms</td>
                  <td style={{ padding: '8px' }}>146.4 FPS</td>
                  <td style={{ padding: '8px' }}>33.0 MiB</td>
                  <td style={{ padding: '8px', color: '#10b981' }}>VERIFIED</td>
                </tr>
                <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)', color: '#a855f7' }}>
                  <td style={{ padding: '8px' }}>PipeGen YOLO11n-Pose Plan</td>
                  <td style={{ padding: '8px' }}>PipeGen Compiler</td>
                  <td style={{ padding: '8px' }}>7.12 ms</td>
                  <td style={{ padding: '8px' }}>140.4 FPS</td>
                  <td style={{ padding: '8px' }}>28.5 MiB</td>
                  <td style={{ padding: '8px', color: '#10b981' }}>VERIFIED</td>
                </tr>
                <tr style={{ color: '#94a3b8' }}>
                  <td style={{ padding: '8px' }}>Python CPU Safety Logic Baseline</td>
                  <td style={{ padding: '8px' }}>time.perf_counter (300 f)</td>
                  <td style={{ padding: '8px' }}>0.12 ms (logic)</td>
                  <td style={{ padding: '8px' }}>265.3 FPS</td>
                  <td style={{ padding: '8px' }}>0.0 MiB</td>
                  <td style={{ padding: '8px', color: '#10b981' }}>BASELINE</td>
                </tr>
              </tbody>
            </table>
            <p style={{ fontSize: '0.72rem', color: '#64748b' }}>
              Full empirical audit available at: <code>pipegen_pipeline/actual_measured_benchmark.json</code> and <code>speed_report.md</code>
            </p>
          </div>
        )}

        {/* Tab 4: JSON Schema */}
        {activeTab === 'json' && (
          <div style={{ background: '#05070e', padding: '16px', borderRadius: '8px', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '8px' }}>
              Track A Standardized Actionable Event JSON Output:
            </div>
            <pre className="font-mono" style={{ color: '#10b981', fontSize: '0.8rem', overflowX: 'auto' }}>
{`{
  "event_type": "restricted_zone_intrusion",
  "timestamp": "2026-10-15T15:30:00.000Z",
  "zone_id": "restricted_zone_1",
  "confidence": 0.9421,
  "bounding_box": [450.0, 300.0, 520.0, 460.0],
  "source": "sample_video"
}`}
            </pre>
          </div>
        )}

      </div>
    </div>
  );
};
