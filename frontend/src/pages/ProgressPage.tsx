import { useEffect, useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend } from 'recharts'
import { progressApi } from '@/services'
import type { ProgressSummary } from '@/types'
import { TrendingUp, AlertCircle } from 'lucide-react'
import { Link } from 'react-router-dom'

const COLORS = ['#22c55e', '#4ade80', '#86efac', '#bbf7d0', '#dcfce7', '#f0fdf4', '#16a34a']

function StatCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="card p-5">
      <div className="text-2xl font-bold text-white">{value}</div>
      <div className="text-white/50 text-sm mt-1">{label}</div>
      {sub && <div className="text-cricket-green text-xs mt-0.5">{sub}</div>}
    </div>
  )
}

export default function ProgressPage() {
  const [summary, setSummary] = useState<ProgressSummary | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    progressApi.getSummary()
      .then(r => setSummary(r.data))
      .catch(e => setError(e.response?.data?.detail || 'Failed to load progress.'))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="text-center text-white/40 py-20">Loading…</div>
  if (error) return (
    <div className="card p-8 text-center">
      <AlertCircle size={32} className="text-red-400 mx-auto mb-3" />
      <p className="text-white/70">{error}</p>
    </div>
  )
  if (!summary) return null

  if (summary.total_analyses === 0) {
    return (
      <div className="max-w-lg mx-auto mt-20 text-center">
        <TrendingUp size={48} className="text-white/20 mx-auto mb-4" />
        <h2 className="text-xl font-bold text-white">No progress data yet</h2>
        <p className="text-white/40 mt-2 mb-6">Upload and analyse a batting video to start tracking progress.</p>
        <Link to="/upload" className="btn-primary px-8 py-3">Upload a Video</Link>
      </div>
    )
  }

  const weeklyData = summary.weekly_activity.map(w => ({
    date: new Date(w.date).toLocaleDateString('en', { weekday: 'short' }),
    analyses: w.analyses,
  }))

  const shotData = Object.entries(summary.shot_distribution).map(([name, count]) => ({
    name: name.replace(/_/g, ' '),
    value: count,
  }))

  const severityData = [
    { name: 'Good', value: summary.observations_by_severity.good ?? 0, color: '#22c55e' },
    { name: 'Info', value: summary.observations_by_severity.info ?? 0, color: '#60a5fa' },
    { name: 'Warning', value: summary.observations_by_severity.warning ?? 0, color: '#fbbf24' },
  ]

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white">Progress Analytics</h1>
        <p className="text-white/40 text-sm mt-1">Your batting journey at a glance</p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Total Analyses" value={summary.total_analyses} sub={`+${summary.this_week_analyses} this week`} />
        <StatCard label="Drills Completed" value={summary.total_drills_completed} sub={`+${summary.this_week_drills} this week`} />
        <StatCard
          label="Most Common Shot"
          value={summary.most_common_shot ? summary.most_common_shot.replace(/_/g, ' ') : '–'}
        />
        <StatCard
          label="Avg. Model Score"
          value={summary.average_confidence != null ? `${(summary.average_confidence * 100).toFixed(0)}%` : '–'}
          sub={summary.average_confidence != null ? 'Uncalibrated classifier score' : undefined}
        />
      </div>

      {/* Weekly activity */}
      <div className="card p-6">
        <h2 className="font-semibold text-white mb-4">Weekly Activity</h2>
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={weeklyData}>
            <XAxis dataKey="date" tick={{ fill: '#ffffff40', fontSize: 12 }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: '#ffffff40', fontSize: 12 }} axisLine={false} tickLine={false} allowDecimals={false} />
            <Tooltip contentStyle={{ background: '#1a2332', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }} />
            <Bar dataKey="analyses" fill="#22c55e" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Shot distribution */}
        {shotData.length > 0 && (
          <div className="card p-6">
            <h2 className="font-semibold text-white mb-4">Shot Distribution</h2>
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={shotData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
                  {shotData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Legend wrapperStyle={{ color: '#ffffff70', fontSize: 12 }} />
                <Tooltip contentStyle={{ background: '#1a2332', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Observations by severity */}
        <div className="card p-6">
          <h2 className="font-semibold text-white mb-4">Feedback Breakdown</h2>
          <div className="space-y-3 mt-2">
            {severityData.map(({ name, value, color }) => {
              const total = severityData.reduce((s, d) => s + d.value, 0)
              const pct = total > 0 ? Math.round((value / total) * 100) : 0
              return (
                <div key={name}>
                  <div className="flex justify-between text-xs text-white/50 mb-1">
                    <span>{name}</span>
                    <span>{value} ({pct}%)</span>
                  </div>
                  <div className="h-2 bg-white/10 rounded-full">
                    <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, backgroundColor: color }} />
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}
