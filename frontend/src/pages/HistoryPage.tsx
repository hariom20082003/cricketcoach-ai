import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { analysisApi, videosApi } from '@/services'
import type { AnalysisResult } from '@/types'
import { History, ChevronRight, AlertCircle, Trash2 } from 'lucide-react'

export default function HistoryPage() {
  const [history, setHistory] = useState<AnalysisResult[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [page, setPage] = useState(0)
  const [hasMore, setHasMore] = useState(true)
  const [deletingId, setDeletingId] = useState<number | null>(null)
  const LIMIT = 10

  const load = async (skip: number) => {
    try {
      const res = await analysisApi.getHistory(skip, LIMIT)
      if (skip === 0) {
        setHistory(res.data)
      } else {
        setHistory(prev => [...prev, ...res.data])
      }
      if (res.data.length < LIMIT) setHasMore(false)
    } catch (e: any) {
      setError(e.response?.data?.detail || 'Failed to load history.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load(0) }, [])

  const loadMore = () => {
    const next = page + 1
    setPage(next)
    load(next * LIMIT)
  }

  const deleteVideo = async (videoId: number, filename: string) => {
    const confirmed = window.confirm(
      `Permanently delete ${filename} and its saved analysis, prediction, pose measurements, and observations?`,
    )
    if (!confirmed) return

    setDeletingId(videoId)
    setError('')
    try {
      await videosApi.delete(videoId)
      setHistory(current => current.filter(item => item.video.id !== videoId))
    } catch (e: any) {
      setError(e.response?.data?.detail || 'Failed to delete this video.')
    } finally {
      setDeletingId(null)
    }
  }

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Analysis History</h1>
        <p className="text-white/40 text-sm mt-1">All your previous batting analyses</p>
      </div>

      {error && (
        <div className="flex items-center gap-3 bg-red-500/10 border border-red-500/20 rounded-xl p-4">
          <AlertCircle size={16} className="text-red-400 shrink-0" />
          <p className="text-red-300 text-sm">{error}</p>
        </div>
      )}

      {loading ? (
        <div className="space-y-3">
          {[1,2,3].map(i => (
            <div key={i} className="card p-4 animate-pulse h-20 bg-cricket-surface/50" />
          ))}
        </div>
      ) : history.length === 0 ? (
        <div className="card p-14 text-center">
          <History size={40} className="text-white/20 mx-auto mb-3" />
          <p className="text-white/60 font-medium">No analyses yet</p>
          <p className="text-white/30 text-sm mt-1 mb-5">Upload your first video to get started.</p>
          <Link to="/upload" className="btn-primary px-6 py-2">Upload Video</Link>
        </div>
      ) : (
        <>
          <div className="space-y-3">
            {history.map(({ job, video, prediction, observations }) => (
              <div key={job.id} className="card p-4 flex items-center gap-4 hover:border-white/15 transition-colors">
                <div className="w-10 h-10 rounded-lg bg-cricket-surface-light flex items-center justify-center shrink-0 text-lg">
                  🏏
                </div>
                <Link to={`/results/${job.id}`} className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <p className="text-white font-medium text-sm truncate">{video.original_filename}</p>
                    <span className={`badge text-xs shrink-0 ${job.status === 'completed' ? 'badge-green' : job.status === 'failed' ? 'badge-red' : 'badge-yellow'}`}>
                      {job.status}
                    </span>
                  </div>
                  <div className="text-white/40 text-xs mt-0.5 flex flex-wrap gap-2">
                    <span>{new Date(job.created_at).toLocaleDateString()}</span>
                    <span>·</span>
                    <span className="capitalize">
                      {prediction?.model_available && prediction?.shot_label
                        ? prediction.shot_label.replace(/_/g, ' ')
                        : 'Classification unavailable'}
                    </span>
                    {prediction?.confidence != null && (
                      <><span>·</span><span>{(prediction.confidence * 100).toFixed(0)}% model score</span></>
                    )}
                    {observations?.length > 0 && (
                      <><span>·</span><span>{observations.length} observations</span></>
                    )}
                  </div>
                </Link>
                <ChevronRight size={16} className="text-white/25 shrink-0" />
                <button
                  type="button"
                  aria-label={`Delete ${video.original_filename}`}
                  title="Delete video and analysis"
                  disabled={deletingId === video.id}
                  onClick={() => deleteVideo(video.id, video.original_filename)}
                  className="p-2 rounded-lg text-white/35 hover:text-red-300 hover:bg-red-500/10 disabled:opacity-40 transition-colors"
                >
                  <Trash2 size={16} />
                </button>
              </div>
            ))}
          </div>

          {hasMore && (
            <button onClick={loadMore} className="btn-secondary w-full py-3">
              Load more
            </button>
          )}
        </>
      )}
    </div>
  )
}
