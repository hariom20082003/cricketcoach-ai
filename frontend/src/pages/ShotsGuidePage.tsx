import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { shotsApi } from '@/services'
import type { ShotGuide } from '@/types'
import { BookOpen, ChevronRight } from 'lucide-react'

const DIFF_COLOR: Record<string, string> = {
  beginner: 'badge-green',
  intermediate: 'badge-yellow',
  advanced: 'badge-red',
}

export default function ShotsGuidePage() {
  const [shots, setShots] = useState<ShotGuide[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    shotsApi.getGuide()
      .then(r => setShots(r.data))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Shot Guide</h1>
        <p className="text-white/40 text-sm mt-1">Learn the fundamentals of cricket batting shots</p>
      </div>

      {loading ? (
        <div className="grid md:grid-cols-2 gap-4">
          {[1,2,3,4].map(i => <div key={i} className="card h-28 animate-pulse bg-cricket-surface/50" />)}
        </div>
      ) : (
        <div className="grid md:grid-cols-2 gap-4">
          {shots.map(shot => (
            <Link
              key={shot.id}
              to={`/shots/${shot.id}`}
              className="card p-5 flex items-center gap-4 hover:border-white/15 transition-colors"
            >
              <div className="w-12 h-12 rounded-xl bg-cricket-green/10 flex items-center justify-center text-2xl shrink-0">
                🏏
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <h3 className="text-white font-semibold text-sm">{shot.name}</h3>
                  <span className={`badge text-xs ${DIFF_COLOR[shot.difficulty] || 'badge-green'}`}>{shot.difficulty}</span>
                </div>
                <p className="text-white/40 text-xs mt-0.5 capitalize">{shot.category.replace(/_/g, ' ')}</p>
              </div>
              <ChevronRight size={16} className="text-white/25 shrink-0" />
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}
