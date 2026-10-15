// Shared TypeScript types for CricketCoach AI

export interface User {
  id: number
  email: string
  username: string
  full_name: string | null
  batting_hand: string
  experience_level: string
  role: string
  is_active: boolean
  created_at: string
}

export interface Token {
  access_token: string
  token_type: string
  user: User
}

export interface Video {
  id: number
  filename: string
  original_filename: string
  file_size_bytes: number
  duration_seconds: number | null
  fps: number | null
  width: number | null
  height: number | null
  batting_hand: string
  recording_angle: string
  created_at: string
}

export interface AnalysisJob {
  id: number
  video_id: number
  status: 'pending' | 'processing' | 'completed' | 'failed'
  progress: number
  error_message: string | null
  started_at: string | null
  completed_at: string | null
  created_at: string
}

export interface ShotPrediction {
  shot_label: string | null
  confidence: number | null
  all_scores: Record<string, number> | null
  model_available: boolean
}

export interface JointAngle {
  mean: number
  std: number
  min: number
  max: number
  n_frames: number
}

export interface PoseResult {
  frame_count: number | null
  sampled_frames: number | null
  landmarks_detected: boolean
  joint_angles_json: Record<string, JointAngle | null> | null
  movement_features_json: Record<string, number> | null
  annotated_frames_paths: string[] | null
  detection_confidence: number | null
}

export interface TechniqueObservation {
  id: number
  category: string
  observation: string
  measurement: string | null
  frame_timestamp: number | null
  reliability: 'high' | 'medium' | 'low'
  explanation: string
  source_type: 'measured_feature' | 'heuristic' | 'validated_reference'
  severity: 'good' | 'warning' | 'info'
  drill_id: number | null
}

export interface AnalysisResult {
  job: AnalysisJob
  video: Video
  prediction: ShotPrediction | null
  pose_result: PoseResult | null
  observations: TechniqueObservation[]
}

export interface PracticeDrill {
  id: number
  title: string
  shot_category: string | null
  technique_focus: string
  difficulty: 'beginner' | 'intermediate' | 'advanced'
  duration_minutes: number
  description: string
  steps: string[]
  equipment: string[] | null
  tips: string[] | null
}

export interface CompletedDrill {
  id: number
  drill_id: number
  notes: string | null
  rating: number | null
  duration_minutes: number | null
  completed_at: string
}

export interface ProgressSummary {
  total_analyses: number
  this_week_analyses: number
  total_drills_completed: number
  this_week_drills: number
  most_common_shot: string | null
  average_confidence: number | null
  observations_by_severity: Record<string, number>
  shot_distribution: Record<string, number>
  weekly_activity: Array<{ date: string; analyses: number }>
}

export interface ModelStatus {
  model_available: boolean
  model_name: string | null
  version: string | null
  accuracy: number | null
  macro_f1: number | null
  architecture: string | null
  test_samples: number | null
  inference_latency_ms: number | null
  dataset: string | null
  supported_shots: string[] | null
  training_samples: number | null
  message: string
}

export interface ShotGuide {
  id: string
  name: string
  category: string
  difficulty: string
  description: string
  key_points: string[]
  common_errors: string[]
  suitable_deliveries: string[]
}

export interface SessionNote {
  id: number
  title: string
  content: string
  tags: string[] | null
  created_at: string
}

export interface ApiError {
  detail: string
}
