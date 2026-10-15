export interface SafetyIncident {
  event_type: string;
  timestamp: string;
  zone_id: string;
  zone_name?: string;
  track_id?: number;
  confidence: number;
  bounding_box: number[];
  detection_point?: number[];
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
  temperature_c: number;
  utilization_pct: number;
  available: boolean;
}

export interface PipelineStatus {
  pipeline_id: string;
  pipeline_name: string;
  pipegen_version: string;
  mode: 'live' | 'demo';
  pipegen_installed: boolean;
  container_ready: boolean;
  container_status: string;
  model_architecture: string;
  tensorrt_engine: string;
  target_hardware: string;
  current_fps: number;
  inference_latency_ms: number;
  total_frames_processed: number;
  total_detections: number;
  total_incidents: number;
  active_intrusions: number;
  gpu: GpuStatus;
  active_zones_count: number;
}

export interface PerformanceMetrics {
  fps: number;
  inference_latency_ms: number;
  pipeline_latency_ms: number;
  target_fps: number;
  gpu_temperature_c: number;
  gpu_memory_used_mb: number;
  gpu_memory_total_mb: number;
  gpu_utilization_pct: number;
  total_frames_processed: number;
  mode: string;
}
