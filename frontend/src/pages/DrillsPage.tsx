import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { drillsApi } from '@/services'
import type { PracticeDrill } from '@/types'
import { Dumbbell, Clock, AlertCircle } from 'lucide-react'

const DIFFICULTY_COLOR: Record<string, string> = {
  beginner: 'badge-green',
  intermediate: 'badge-yellow',
  advanced: 'badge-red',
}

export default function DrillsPage() {
  const [drills, setDrills] = useState<PracticeDrill[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [filterShot, setFilterShot] = useState('')
  const [filterDiff, setFilterDiff] = useState('')

  useEffect(() => {
    drillsApi.list(filterShot || undefined, filterDiff || undefined)
      .then(r => setDrills(r.data))
      .catch(e => setError(e.response?.data?.detail || 'Failed to load drills.'))
      .finally(() => setLoading(false))
  }, [filterShot, filterDiff])

  const shots = Array.from(new Set(drills.map(d => d.shot_category).filter(Boolean))) as string[]

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Practice Drills</h1>
        <p className="text-white/40 text-sm mt-1">Structured exercises to improve your batting technique</p>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3">
        <select
          className="input w-auto text-sm py-2"
          value={filterShot}
          onChange={e => { setFilterShot(e.target.value); setLoading(true) }}
        >
          <option value="">All shots</option>
          {shots.map(s => <option key={s} value={s}>{s.replace(/_/g, ' ')}</option>)}
        </select>
        <select
          className="input w-auto text-sm py-2"
          value={filterDiff}
          onChange={e => { setFilterDiff(e.target.value); setLoading(true) }}
        >
          <option value="">All levels</option>
          <option value="beginner">Beginner</option>
          <option value="intermediate">Intermediate</option>
          <option value="advanced">Advanced</option>
        </select>
      </div>

      {error && (
        <div className="flex items-center gap-3 bg-red-500/10 border border-red-500/20 rounded-xl p-4">
          <AlertCircle size={16} className="text-red-400" />
          <p className="text-red-300 text-sm">{error}</p>
        </div>
      )}

      {loading ? (
        <div className="grid md:grid-cols-2 gap-4">
          {[1,2,3,4].map(i => <div key={i} className="card h-36 animate-pulse bg-cricket-surface/50" />)}
        </div>
      ) : drills.length === 0 ? (
        <div className="card p-14 text-center">
          <Dumbbell size={40} className="text-white/20 mx-auto mb-3" />
          <p className="text-white/60">No drills found for these filters.</p>
        </div>
      ) : (
        <div className="grid md:grid-cols-2 gap-4">
          {drills.map(drill => (
            <Link
              key={drill.id}
              to={`/drills/${drill.id}`}
              className="card p-5 flex flex-col gap-3 hover:border-white/15 transition-colors"
            >
              <div className="flex items-start justify-between gap-3">
                <h3 className="text-white font-semibold leading-tight">{drill.title}</h3>
                <span className={`badge text-xs shrink-0 ${DIFFICULTY_COLOR[drill.difficulty]}`}>
                  {drill.difficulty}
                </span>
              </div>
              <p className="text-white/50 text-sm leading-relaxed line-clamp-2">{drill.description}</p>
              <div className="flex flex-wrap gap-3 text-xs text-white/40">
                <span className="flex items-center gap-1">
                  <Clock size={12} /> {drill.duration_minutes} min
                </span>
                {drill.shot_category && (
                  <span className="capitalize">{drill.shot_category.replace(/_/g, ' ')}</span>
                )}
                <span className="capitalize">{drill.technique_focus}</span>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}
