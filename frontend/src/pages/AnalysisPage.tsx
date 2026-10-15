import { useEffect, useState, useRef } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import { analysisApi } from '@/services'
import type { AnalysisJob } from '@/types'
import { AlertCircle, X } from 'lucide-react'

const STATUS_MESSAGES: Record<string, string[]> = {
  pending: ['Queued for processing…', 'Waiting to start…'],
  processing: [
    'Extracting video frames…',
    'Running MediaPipe pose estimation…',
    'Detecting body landmarks…',
    'Computing joint angles…',
    'Running shot classification…',
    'Generating technique observations…',
    'Finalising results…',
  ],
}

export default function AnalysisPage() {
  const { jobId } = useParams<{ jobId: string }>()
  const navigate = useNavigate()
  const [job, setJob] = useState<AnalysisJob | null>(null)
  const [error, setError] = useState('')
  const [msgIdx, setMsgIdx] = useState(0)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const msgRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    if (!jobId) return

    const poll = async () => {
      try {
        const res = await analysisApi.getStatus(Number(jobId))
        setJob(res.data)
        if (res.data.status === 'completed') {
          clearInterval(intervalRef.current!)
          navigate(`/results/${jobId}`)
        } else if (res.data.status === 'failed') {
          clearInterval(intervalRef.current!)
          setError(res.data.error_message || 'Analysis failed.')
        }
      } catch (e: any) {
        clearInterval(intervalRef.current!)
        setError(e.response?.data?.detail || 'Error checking status.')
      }
    }

    poll()
    intervalRef.current = setInterval(poll, 2000)

    // Rotate status messages
    msgRef.current = setInterval(() => {
      setMsgIdx(i => i + 1)
    }, 2500)

    return () => {
      clearInterval(intervalRef.current!)
      clearInterval(msgRef.current!)
    }
  }, [jobId, navigate])

  const progress = job?.progress ?? 0
  const status = job?.status ?? 'pending'
  const msgs = STATUS_MESSAGES[status] || STATUS_MESSAGES.processing
  const currentMsg = msgs[msgIdx % msgs.length]

  const circumference = 2 * Math.PI * 54

  if (error) {
    return (
      <div className="max-w-md mx-auto mt-20 text-center space-y-6">
        <div className="w-16 h-16 rounded-full bg-red-500/10 flex items-center justify-center mx-auto">
          <AlertCircle size={32} className="text-red-400" />
        </div>
        <div>
          <h2 className="text-xl font-bold text-white">Analysis failed</h2>
          <p className="text-white/50 mt-2 text-sm">{error}</p>
        </div>
        <div className="flex gap-3 justify-center">
          <Link to="/upload" className="btn-primary px-6 py-2">Try again</Link>
          <Link to="/history" className="btn-secondary px-6 py-2">History</Link>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-md mx-auto mt-16 text-center space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white">Analysing your shot…</h1>
        <p className="text-white/40 text-sm mt-1">This usually takes 15–60 seconds</p>
      </div>

      {/* Circular progress */}
      <div className="relative w-36 h-36 mx-auto">
        <svg className="w-full h-full -rotate-90" viewBox="0 0 120 120">
          <circle cx="60" cy="60" r="54" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="8" />
          <circle
            cx="60" cy="60" r="54"
            fill="none" stroke="#22c55e" strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={circumference - (circumference * progress) / 100}
            className="transition-all duration-500"
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-2xl font-bold text-white">{progress}%</span>
        </div>
      </div>

      <div>
        <p className="text-cricket-green font-medium animate-pulse">{currentMsg}</p>
        <p className="text-white/30 text-xs mt-2">Job #{jobId}</p>
      </div>

      <button
        onClick={() => navigate('/upload')}
        className="flex items-center gap-2 text-white/40 hover:text-white/70 text-sm mx-auto transition-colors"
      >
        <X size={14} /> Cancel
      </button>
    </div>
  )
}
