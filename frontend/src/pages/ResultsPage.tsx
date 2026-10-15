import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { analysisApi } from '@/services'
import type { AnalysisResult } from '@/types'
import { CheckCircle, AlertTriangle, Info, AlertCircle, ChevronRight } from 'lucide-react'

const SEVERITY_ICON = {
  good: <CheckCircle size={16} className="text-cricket-green shrink-0" />,
  warning: <AlertTriangle size={16} className="text-amber-400 shrink-0" />,
  info: <Info size={16} className="text-blue-400 shrink-0" />,
}

const RELIABILITY_COLOR = {
  high: 'badge-green',
  medium: 'badge-yellow',
  low: 'badge-red',
}

const SOURCE_LABEL = {
  measured_feature: 'Measured',
  heuristic: 'Heuristic',
  validated_reference: 'Reference',
}

const SHOT_LABEL_NAMES: Record<string, string> = {
  cover: 'Cover Drive', defense: 'Defense Shot', flick: 'Flick Shot',
  hook: 'Hook Shot', late_cut: 'Late Cut', lofted: 'Lofted Drive',
  pull: 'Pull Shot', square_cut: 'Square Cut', straight: 'Straight Drive',
  sweep: 'Sweep Shot',
}

function shotName(label: string) {
  return SHOT_LABEL_NAMES[label] ?? label.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

function formatAngle(v: number | null | undefined) {
  if (v == null) return '–'
  return `${v.toFixed(1)}°`
}

function angleColor(name: string, val: number) {
  // Simple heuristic ranges — flagging only obvious extremes
  if (name.includes('elbow') && (val < 90 || val > 170)) return 'text-amber-400'
  if (name.includes('knee') && (val < 120 || val > 175)) return 'text-amber-400'
  return 'text-cricket-green'
}

export default function ResultsPage() {
  const { jobId } = useParams<{ jobId: string }>()
  const [data, setData] = useState<AnalysisResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!jobId) return
    analysisApi.getResults(Number(jobId))
      .then(r => setData(r.data))
      .catch(e => setError(e.response?.data?.detail || 'Failed to load results.'))
      .finally(() => setLoading(false))
  }, [jobId])

  if (loading) return <div className="text-center text-white/40 py-20">Loading results…</div>
  if (error) return (
    <div className="max-w-md mx-auto mt-20 text-center">
      <AlertCircle size={40} className="text-red-400 mx-auto mb-4" />
      <p className="text-white font-medium">{error}</p>
      <Link to="/history" className="btn-secondary mt-4 inline-block px-6 py-2">View History</Link>
    </div>
  )
  if (!data) return null

  const { job, video, prediction, pose_result, observations } = data

  const scoreData = prediction?.all_scores
    ? Object.entries(prediction.all_scores)
        .map(([name, score]) => ({ name: shotName(name), score: Math.round(score * 100) }))
        .sort((a, b) => b.score - a.score)
    : []

  const angles = pose_result?.joint_angles_json ?? {}

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Analysis Results</h1>
          <p className="text-white/40 text-sm mt-1">{video.original_filename} · {new Date(job.created_at).toLocaleString()}</p>
        </div>
        <div className="flex gap-3">
          <Link to="/upload" className="btn-secondary px-4 py-2 text-sm">Analyse another</Link>
          <Link to="/history" className="btn-secondary px-4 py-2 text-sm">History</Link>
        </div>
      </div>

      {/* Shot Classification */}
      <div className="card p-6">
        <h2 className="font-semibold text-white mb-4">Shot Classification</h2>
        {!prediction || !prediction.model_available ? (
          <div className="flex items-start gap-3 bg-amber-500/10 border border-amber-500/20 rounded-lg p-4">
            <Info size={16} className="text-amber-400 mt-0.5 shrink-0" />
            <div>
              <p className="text-amber-300 text-sm font-medium">No trained model available</p>
              <p className="text-amber-200/60 text-xs mt-0.5">
                Shot classification is unavailable in this backend instance. Pose analysis can still be reviewed.{' '}
                <Link to="/model-status" className="underline">See Model Status →</Link>
              </p>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex items-center gap-4">
              <div className="text-3xl">🏏</div>
              <div>
                <div className="text-xl font-bold text-white capitalize">
                  {prediction.shot_label ? shotName(prediction.shot_label) : 'Unknown'}
                </div>
                {prediction.confidence != null && (
                  <div className="text-cricket-green font-medium text-sm">
                    {(prediction.confidence * 100).toFixed(1)}% model score
                  </div>
                )}
              </div>
            </div>
            {scoreData.length > 0 && (
              <div>
                <p className="text-white/40 text-xs mb-3">Class scores · uncalibrated</p>
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={scoreData} layout="vertical">
                    <XAxis type="number" domain={[0, 100]} tick={{ fill: '#ffffff40', fontSize: 11 }} />
                    <YAxis type="category" dataKey="name" width={110} interval={0} tick={{ fill: '#ffffff70', fontSize: 11 }} />
                    <Tooltip contentStyle={{ background: '#1a2332', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }} />
                    <Bar dataKey="score" radius={4}>
                      {scoreData.map((entry, i) => (
                        <Cell key={i} fill={i === 0 ? '#22c55e' : '#ffffff20'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        )}
        {prediction?.model_available && (
          <p className="text-white/35 text-xs mt-3">
            The visual classifier identifies the shot from sampled video frames. Technique observations below are generated separately from pose measurements and coaching heuristics.
          </p>
        )}
      </div>

      {/* Pose Analysis */}
      {pose_result && (
        <div className="card p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-semibold text-white">Pose Analysis</h2>
            <span className={`badge text-xs ${pose_result.landmarks_detected ? 'badge-green' : 'badge-red'}`}>
              {pose_result.landmarks_detected ? 'Landmarks detected' : 'No landmarks'}
            </span>
          </div>

          {pose_result.detection_confidence != null && (
            <div className="mb-4">
              <div className="flex justify-between text-xs text-white/40 mb-1">
                <span>Detection confidence</span>
                <span>{(pose_result.detection_confidence * 100).toFixed(0)}%</span>
              </div>
              <div className="h-1.5 bg-white/10 rounded-full">
                <div
                  className="h-full bg-cricket-green rounded-full"
                  style={{ width: `${(pose_result.detection_confidence * 100).toFixed(0)}%` }}
                />
              </div>
            </div>
          )}

          {Object.keys(angles).length > 0 && (
            <div>
              <p className="text-white/40 text-xs mb-3">Joint angles (mean across frames)</p>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                {Object.entries(angles).map(([name, stats]) => (
                  stats && (
                    <div key={name} className="bg-cricket-surface-light rounded-lg p-3">
                      <div className="text-white/40 text-xs capitalize">{name.replace(/_/g, ' ')}</div>
                      <div className={`text-base font-bold mt-0.5 ${angleColor(name, stats.mean)}`}>
                        {formatAngle(stats.mean)}
                      </div>
                      <div className="text-white/25 text-xs">{stats.n_frames} frames</div>
                    </div>
                  )
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Technique Observations */}
      <div className="card p-6">
        <h2 className="font-semibold text-white mb-4">
          Technique Observations
          <span className="ml-2 text-white/30 font-normal text-sm">({observations.length})</span>
        </h2>

        {observations.length === 0 ? (
          <p className="text-white/40 text-sm">No observations generated.</p>
        ) : (
          <div className="space-y-3">
            {observations.map((obs) => (
              <div
                key={obs.id}
                className={`rounded-xl border p-4 ${
                  obs.severity === 'good' ? 'border-cricket-green/20 bg-cricket-green/5' :
                  obs.severity === 'warning' ? 'border-amber-500/20 bg-amber-500/5' :
                  'border-blue-500/20 bg-blue-500/5'
                }`}
              >
                <div className="flex items-start gap-3">
                  {SEVERITY_ICON[obs.severity]}
                  <div className="flex-1 min-w-0">
                    <div className="flex flex-wrap items-center gap-2 mb-1">
                      <span className="text-white font-medium text-sm">{obs.observation}</span>
                      <span className="badge badge-yellow text-xs capitalize">{obs.category.replace(/_/g, ' ')}</span>
                    </div>
                    <p className="text-white/55 text-sm leading-relaxed">{obs.explanation}</p>

                    {obs.measurement && (
                      <code className="block text-xs text-cricket-green/70 font-mono bg-black/20 rounded px-2 py-1 mt-2">
                        {obs.measurement}
                      </code>
                    )}

                    <div className="flex flex-wrap gap-2 mt-3">
                      <span className={`badge text-xs ${RELIABILITY_COLOR[obs.reliability]}`}>
                        {obs.reliability} reliability
                      </span>
                      <span className="badge text-xs bg-white/5 text-white/40">
                        {SOURCE_LABEL[obs.source_type]}
                      </span>
                      {obs.drill_id && (
                        <Link
                          to={`/drills/${obs.drill_id}`}
                          className="flex items-center gap-1 text-xs text-cricket-green hover:underline"
                        >
                          View drill <ChevronRight size={12} />
                        </Link>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
