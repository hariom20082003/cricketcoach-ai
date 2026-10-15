import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { useAuthStore } from '@/store/auth'
import { analysisApi, modelApi } from '@/services'
import type { AnalysisResult, ModelStatus } from '@/types'
import { Upload, History, Dumbbell, BookOpen, AlertCircle, TrendingUp } from 'lucide-react'

export default function DashboardPage() {
  const user = useAuthStore((s) => s.user)
  const [recentAnalyses, setRecentAnalyses] = useState<AnalysisResult[]>([])
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      analysisApi.getHistory(0, 3).then(r => setRecentAnalyses(r.data)),
      modelApi.getStatus().then(r => setModelStatus(r.data)),
    ]).finally(() => setLoading(false))
  }, [])

  const quickActions = [
    { to: '/upload', icon: Upload, label: 'Analyze a Shot', desc: 'Upload or record video' },
    { to: '/history', icon: History, label: 'View History', desc: 'Past analyses' },
    { to: '/drills', icon: Dumbbell, label: 'Practice Drills', desc: 'Exercises & tips' },
    { to: '/shots', icon: BookOpen, label: 'Shot Guide', desc: 'Learn cricket shots' },
  ]

  return (
    <div className="max-w-5xl mx-auto space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white">
          Welcome back, {user?.full_name || user?.username}! 👋
        </h1>
        <p className="text-white/50 mt-1">
          {user?.experience_level?.charAt(0).toUpperCase()}{user?.experience_level?.slice(1)} •{' '}
          {user?.batting_hand}-handed batsman
        </p>
      </div>

      {/* Model status banner */}
      {modelStatus && !modelStatus.model_available && (
        <div className="flex items-start gap-3 bg-amber-500/10 border border-amber-500/20 rounded-xl p-4">
          <AlertCircle className="text-amber-400 mt-0.5 shrink-0" size={18} />
          <div>
            <p className="text-amber-300 font-medium text-sm">Shot classification not available</p>
            <p className="text-amber-200/60 text-xs mt-0.5">
              {modelStatus.message} Pose analysis and coaching observations will still work.{' '}
              <Link to="/model-status" className="underline">Learn more →</Link>
            </p>
          </div>
        </div>
      )}

      {/* Quick actions */}
      <div>
        <h2 className="text-sm font-semibold text-white/40 uppercase tracking-wider mb-4">Quick actions</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {quickActions.map(({ to, icon: Icon, label, desc }) => (
            <Link
              key={to}
              to={to}
              className="card p-5 flex flex-col items-center text-center gap-3 hover:border-cricket-green/30 transition-colors group"
            >
              <div className="w-10 h-10 rounded-xl bg-cricket-green/10 group-hover:bg-cricket-green/20 flex items-center justify-center transition-colors">
                <Icon size={20} className="text-cricket-green" />
              </div>
              <div>
                <div className="text-white font-medium text-sm">{label}</div>
                <div className="text-white/40 text-xs mt-0.5">{desc}</div>
              </div>
            </Link>
          ))}
        </div>
      </div>

      {/* Recent analyses */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-semibold text-white/40 uppercase tracking-wider">Recent Analyses</h2>
          <Link to="/history" className="text-cricket-green text-sm hover:underline">View all →</Link>
        </div>

        {loading ? (
          <div className="card p-8 text-center text-white/40">Loading…</div>
        ) : recentAnalyses.length === 0 ? (
          <div className="card p-10 text-center">
            <TrendingUp size={40} className="text-white/20 mx-auto mb-3" />
            <p className="text-white/60 font-medium">No analyses yet</p>
            <p className="text-white/30 text-sm mt-1 mb-5">Upload your first batting video to get started.</p>
            <Link to="/upload" className="btn-primary px-6 py-2 text-sm">Upload Video</Link>
          </div>
        ) : (
          <div className="space-y-3">
            {recentAnalyses.map(({ job, video, prediction }) => (
              <Link
                key={job.id}
                to={`/results/${job.id}`}
                className="card p-4 flex items-center gap-4 hover:border-white/15 transition-colors"
              >
                <div className="w-10 h-10 rounded-lg bg-cricket-surface-light flex items-center justify-center shrink-0 text-lg">
                  🏏
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-white font-medium text-sm truncate">{video.original_filename}</div>
                  <div className="text-white/40 text-xs">
                    {new Date(job.created_at).toLocaleDateString()} •{' '}
                    {prediction?.model_available && prediction?.shot_label
                      ? prediction.shot_label.replace(/_/g, ' ')
                      : 'Shot classification unavailable'}
                  </div>
                </div>
                <span className={`badge text-xs ${job.status === 'completed' ? 'badge-green' : job.status === 'failed' ? 'badge-red' : 'badge-yellow'}`}>
                  {job.status}
                </span>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
