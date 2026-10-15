import api from './api'
export { getApiErrorMessage } from './errors'
import type {
  Token, User, Video, AnalysisJob, AnalysisResult,
  PracticeDrill, CompletedDrill, ProgressSummary, ModelStatus,
  ShotGuide, SessionNote
} from '@/types'

// ── Auth ──────────────────────────────────────────────────────────────────────
export const authApi = {
  register: (data: { email: string; username: string; password: string; full_name?: string; batting_hand?: string; experience_level?: string }) =>
    api.post<Token>('/auth/register', data),
  login: (email: string, password: string) =>
    api.post<Token>('/auth/login', { email, password }),
  logout: () => api.post<void>('/auth/logout'),
  deleteMe: () => api.delete<void>('/auth/me'),
  getMe: () => api.get<User>('/auth/me'),
  updateMe: (data: Partial<User>) => api.put<User>('/auth/me', data),
}

// ── Videos ────────────────────────────────────────────────────────────────────
export const videosApi = {
  upload: (file: File, batting_hand: string, recording_angle: string, onProgress?: (p: number) => void) => {
    const form = new FormData()
    form.append('file', file)
    form.append('batting_hand', batting_hand)
    form.append('recording_angle', recording_angle)
    return api.post<Video>('/videos/upload', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (e) => {
        if (onProgress && e.total) {
          onProgress(Math.round((e.loaded * 100) / e.total))
        }
      },
    })
  },
  list: () => api.get<Video[]>('/videos/'),
  get: (id: number) => api.get<Video>(`/videos/${id}`),
  delete: (id: number) => api.delete(`/videos/${id}`),
  streamUrl: (id: number) => `/api/videos/${id}/stream`,
}

// ── Analysis ──────────────────────────────────────────────────────────────────
export const analysisApi = {
  start: (videoId: number) => api.post<AnalysisJob>(`/analysis/start/${videoId}`),
  getStatus: (jobId: number) => api.get<AnalysisJob>(`/analysis/${jobId}/status`),
  getResults: (jobId: number) => api.get<AnalysisResult>(`/analysis/${jobId}/results`),
  getHistory: (skip = 0, limit = 20) =>
    api.get<AnalysisResult[]>(`/analysis/history/list?skip=${skip}&limit=${limit}`),
}

// ── Drills ────────────────────────────────────────────────────────────────────
export const drillsApi = {
  list: (shot_category?: string, difficulty?: string) => {
    const params = new URLSearchParams()
    if (shot_category) params.append('shot_category', shot_category)
    if (difficulty) params.append('difficulty', difficulty)
    return api.get<PracticeDrill[]>(`/drills/?${params}`)
  },
  get: (id: number) => api.get<PracticeDrill>(`/drills/${id}`),
  complete: (id: number, data: { notes?: string; rating?: number; duration_minutes?: number }) =>
    api.post<CompletedDrill>(`/drills/${id}/complete`, data),
  listCompleted: () => api.get<CompletedDrill[]>('/drills/completed/list'),
}

// ── Progress ──────────────────────────────────────────────────────────────────
export const progressApi = {
  getSummary: () => api.get<ProgressSummary>('/progress/summary'),
}

// ── Model ─────────────────────────────────────────────────────────────────────
export const modelApi = {
  getStatus: () => api.get<ModelStatus>('/model/status'),
}

// ── Shots Guide ───────────────────────────────────────────────────────────────
export const shotsApi = {
  getGuide: () => api.get<ShotGuide[]>('/shots/guide'),
  getShot: (id: string) => api.get<ShotGuide>(`/shots/guide/${id}`),
}

// ── Notes ─────────────────────────────────────────────────────────────────────
export const notesApi = {
  list: () => api.get<SessionNote[]>('/notes/'),
  create: (data: { title: string; content: string; tags?: string[] }) =>
    api.post<SessionNote>('/notes/', data),
  delete: (id: number) => api.delete(`/notes/${id}`),
}
