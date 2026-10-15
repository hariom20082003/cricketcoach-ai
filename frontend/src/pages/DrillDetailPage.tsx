import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { drillsApi } from '@/services'
import type { PracticeDrill } from '@/types'
import { Clock, Package, Lightbulb, CheckCircle, AlertCircle, Star } from 'lucide-react'

const DIFFICULTY_COLOR: Record<string, string> = {
  beginner: 'badge-green',
  intermediate: 'badge-yellow',
  advanced: 'badge-red',
}

export default function DrillDetailPage() {
  const { drillId } = useParams<{ drillId: string }>()
  const [drill, setDrill] = useState<PracticeDrill | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [completing, setCompleting] = useState(false)
  const [showForm, setShowForm] = useState(false)
  const [done, setDone] = useState(false)
  const [notes, setNotes] = useState('')
  const [rating, setRating] = useState(0)
  const [duration, setDuration] = useState('')

  useEffect(() => {
    if (!drillId) return
    drillsApi.get(Number(drillId))
      .then(r => setDrill(r.data))
      .catch(e => setError(e.response?.data?.detail || 'Drill not found.'))
      .finally(() => setLoading(false))
  }, [drillId])

  const handleComplete = async () => {
    setCompleting(true)
    try {
      await drillsApi.complete(Number(drillId), {
        notes: notes || undefined,
        rating: rating || undefined,
        duration_minutes: duration ? Number(duration) : undefined,
      })
      setDone(true)
      setShowForm(false)
    } catch (e: any) {
      setError(e.response?.data?.detail || 'Failed to mark drill as complete.')
    } finally {
      setCompleting(false)
    }
  }

  if (loading) return <div className="text-center text-white/40 py-20">Loading…</div>
  if (error) return (
    <div className="card p-8 text-center">
      <AlertCircle size={32} className="text-red-400 mx-auto mb-3" />
      <p className="text-white/70">{error}</p>
      <Link to="/drills" className="btn-secondary mt-4 inline-block px-6 py-2">Back to Drills</Link>
    </div>
  )
  if (!drill) return null

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <Link to="/drills" className="text-white/40 text-sm hover:text-white/70">← Back to Drills</Link>
        <h1 className="text-2xl font-bold text-white mt-3">{drill.title}</h1>
        <div className="flex flex-wrap gap-2 mt-2">
          <span className={`badge ${DIFFICULTY_COLOR[drill.difficulty]}`}>{drill.difficulty}</span>
          <span className="badge bg-white/5 text-white/50 flex items-center gap-1">
            <Clock size={11} /> {drill.duration_minutes} min
          </span>
          {drill.shot_category && (
            <span className="badge bg-white/5 text-white/50 capitalize">
              {drill.shot_category.replace(/_/g, ' ')}
            </span>
          )}
          <span className="badge bg-white/5 text-white/50 capitalize">{drill.technique_focus}</span>
        </div>
      </div>

      {/* Description */}
      <div className="card p-6">
        <p className="text-white/70 leading-relaxed">{drill.description}</p>
      </div>

      {/* Steps */}
      <div className="card p-6">
        <h2 className="font-semibold text-white mb-4">Steps</h2>
        <ol className="space-y-3">
          {drill.steps.map((step, i) => (
            <li key={i} className="flex gap-3">
              <span className="w-6 h-6 rounded-full bg-cricket-green/20 text-cricket-green text-xs font-bold flex items-center justify-center shrink-0 mt-0.5">
                {i + 1}
              </span>
              <p className="text-white/70 text-sm leading-relaxed">{step}</p>
            </li>
          ))}
        </ol>
      </div>

      {/* Equipment */}
      {drill.equipment && drill.equipment.length > 0 && (
        <div className="card p-6">
          <h2 className="font-semibold text-white mb-3 flex items-center gap-2">
            <Package size={16} className="text-white/40" /> Equipment
          </h2>
          <ul className="space-y-1">
            {drill.equipment.map((e, i) => (
              <li key={i} className="text-white/60 text-sm">• {e}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Tips */}
      {drill.tips && drill.tips.length > 0 && (
        <div className="card p-6">
          <h2 className="font-semibold text-white mb-3 flex items-center gap-2">
            <Lightbulb size={16} className="text-cricket-green" /> Tips
          </h2>
          <ul className="space-y-2">
            {drill.tips.map((tip, i) => (
              <li key={i} className="text-white/60 text-sm flex gap-2">
                <span className="text-cricket-green shrink-0">→</span> {tip}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Complete */}
      {done ? (
        <div className="card p-6 flex items-center gap-3 border-cricket-green/30 bg-cricket-green/5">
          <CheckCircle size={22} className="text-cricket-green" />
          <div>
            <p className="text-cricket-green font-semibold">Drill completed!</p>
            <p className="text-cricket-green/60 text-sm">Great work. Keep it up!</p>
          </div>
        </div>
      ) : showForm ? (
        <div className="card p-6 space-y-4">
          <h2 className="font-semibold text-white">Log completion</h2>
          <div>
            <label className="label">Rating</label>
            <div className="flex gap-2">
              {[1,2,3,4,5].map(n => (
                <button key={n} onClick={() => setRating(n)}>
                  <Star size={22} className={n <= rating ? 'text-amber-400 fill-amber-400' : 'text-white/20'} />
                </button>
              ))}
            </div>
          </div>
          <div>
            <label className="label">Duration (minutes)</label>
            <input type="number" className="input w-32" value={duration} onChange={e => setDuration(e.target.value)} placeholder="20" />
          </div>
          <div>
            <label className="label">Notes (optional)</label>
            <textarea className="input h-24 resize-none" value={notes} onChange={e => setNotes(e.target.value)} placeholder="How did it go?" />
          </div>
          <div className="flex gap-3">
            <button onClick={handleComplete} disabled={completing} className="btn-primary px-6 py-2">
              {completing ? 'Saving…' : 'Save'}
            </button>
            <button onClick={() => setShowForm(false)} className="btn-secondary px-6 py-2">Cancel</button>
          </div>
        </div>
      ) : (
        <button onClick={() => setShowForm(true)} className="btn-primary w-full py-3 flex items-center justify-center gap-2">
          <CheckCircle size={18} /> Mark as Complete
        </button>
      )}
    </div>
  )
}
