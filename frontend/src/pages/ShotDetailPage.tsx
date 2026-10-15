import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { shotsApi } from '@/services'
import type { ShotGuide } from '@/types'
import { CheckCircle, AlertTriangle, Zap, AlertCircle } from 'lucide-react'

export default function ShotDetailPage() {
  const { shotId } = useParams<{ shotId: string }>()
  const [shot, setShot] = useState<ShotGuide | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!shotId) return
    shotsApi.getShot(shotId)
      .then(r => setShot(r.data))
      .catch(e => setError(e.response?.data?.detail || 'Shot not found.'))
      .finally(() => setLoading(false))
  }, [shotId])

  if (loading) return <div className="text-center text-white/40 py-20">Loading…</div>
  if (error) return (
    <div className="card p-8 text-center">
      <AlertCircle size={32} className="text-red-400 mx-auto mb-3" />
      <p className="text-white/70">{error}</p>
      <Link to="/shots" className="btn-secondary mt-4 inline-block px-6 py-2">Back to Guide</Link>
    </div>
  )
  if (!shot) return null

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <div>
        <Link to="/shots" className="text-white/40 text-sm hover:text-white/70">← Back to Shot Guide</Link>
        <div className="flex items-center gap-4 mt-4">
          <div className="w-14 h-14 rounded-xl bg-cricket-green/10 flex items-center justify-center text-3xl">🏏</div>
          <div>
            <h1 className="text-2xl font-bold text-white">{shot.name}</h1>
            <div className="flex gap-2 mt-1">
              <span className="badge badge-green capitalize">{shot.difficulty}</span>
              <span className="badge bg-white/5 text-white/50 capitalize">{shot.category.replace(/_/g, ' ')}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Description */}
      <div className="card p-6">
        <p className="text-white/70 leading-relaxed">{shot.description}</p>
      </div>

      {/* Key points */}
      <div className="card p-6">
        <h2 className="font-semibold text-white mb-4 flex items-center gap-2">
          <CheckCircle size={16} className="text-cricket-green" /> Key Points
        </h2>
        <ol className="space-y-3">
          {shot.key_points.map((pt, i) => (
            <li key={i} className="flex gap-3">
              <span className="w-5 h-5 rounded-full bg-cricket-green/20 text-cricket-green text-xs font-bold flex items-center justify-center shrink-0 mt-0.5">{i + 1}</span>
              <p className="text-white/70 text-sm leading-relaxed">{pt}</p>
            </li>
          ))}
        </ol>
      </div>

      {/* Common errors */}
      <div className="card p-6">
        <h2 className="font-semibold text-white mb-4 flex items-center gap-2">
          <AlertTriangle size={16} className="text-amber-400" /> Common Errors
        </h2>
        <ul className="space-y-2">
          {shot.common_errors.map((err, i) => (
            <li key={i} className="flex gap-3 text-sm text-white/60">
              <span className="text-amber-400 shrink-0">!</span> {err}
            </li>
          ))}
        </ul>
      </div>

      {/* Suitable deliveries */}
      {shot.suitable_deliveries?.length > 0 && (
        <div className="card p-6">
          <h2 className="font-semibold text-white mb-4 flex items-center gap-2">
            <Zap size={16} className="text-cricket-green" /> Suitable Deliveries
          </h2>
          <div className="flex flex-wrap gap-2">
            {shot.suitable_deliveries.map((d, i) => (
              <span key={i} className="badge bg-cricket-green/10 text-cricket-green border border-cricket-green/20 text-xs">{d}</span>
            ))}
          </div>
        </div>
      )}

      <Link to="/drills" className="btn-secondary w-full py-3 flex items-center justify-center gap-2">
        Find drills for this shot →
      </Link>
    </div>
  )
}
