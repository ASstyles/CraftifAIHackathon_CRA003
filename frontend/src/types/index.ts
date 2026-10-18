export interface SafetyIncident {
  incident_id?: string;
  action?: 'created' | 'updated' | 'resolved';
  update_seq?: number;
  is_update?: boolean;
  event_type: string;
  timestamp: string;
  zone_id: string;
  zone_name?: string;
  track_id?: number;
  confidence: number;
  bounding_box: number[];
  detection_point?: number[];
  contact_source?: string;
  source: string;
  risk_level: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
  dwell_time_seconds: number;
  pipegen_pipeline_id?: string;
  raw_pipegen_meta?: any;
}

export interface RestrictedZone {
  zone_id: string;
  name: string;
  polygon: number[][];
  color: string;
  risk_level: string;
  is_active: boolean;
  cooldown_seconds: number;
}

export interface GpuStatus {
  name: string;
  driver: string;
  memory_total_mb: number;
  memory_used_mb: number;
  memory_free_mb: number;
  temperature_c: number | null;
  utilization_pct: number | null;
  available: boolean;
}

export interface PipelineStatus {
  pipeline_id: string;
  pipeline_name: string;
  pipegen_version: string;
  mode: 'live' | 'demo';
  operational_status?: string;
  pipegen_installed: boolean;
  container_ready: boolean;
  container_status: string;
  live_pipeline_active?: boolean;
  last_pipegen_ingest_seconds_ago?: number | null;
  model_architecture: string;
  tensorrt_engine: string;
  pose_engine?: string;
  target_hardware: string;
  target_fps?: number;
  target_latency_ms?: number;
  current_fps: number | null;
  inference_latency_ms: number | null;
  pipeline_latency_ms?: number | null;
  ingest_latency_ms?: number | null;
  metrics_source?: string;
  inference_backend?: string;
  is_stale?: boolean;
  total_frames_processed: number;
  total_detections: number;
  total_incidents: number;
  active_intrusions: number;
  gpu: GpuStatus;
  active_zones_count: number;
}

export interface PerformanceMetrics {
  fps: number | null;
  inference_latency_ms: number | null;
  pipeline_latency_ms: number | null;
  ingest_latency_ms?: number | null;
  target_fps: number;
  target_latency_ms?: number;
  is_stale?: boolean;
  metrics_source?: string;
  inference_backend?: string;
  gpu_temperature_c: number | null;
  gpu_memory_used_mb: number | null;
  gpu_memory_total_mb: number | null;
  gpu_utilization_pct: number | null;
  total_frames_processed: number;
  mode: string;
}
