import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { modelApi } from '@/services'
import type { ModelStatus } from '@/types'
import { Cpu, CheckCircle, AlertTriangle, AlertCircle } from 'lucide-react'

const SHOT_NAMES: Record<string, string> = {
  cover: 'Cover Drive', defense: 'Defense Shot', flick: 'Flick Shot',
  hook: 'Hook Shot', late_cut: 'Late Cut', lofted: 'Lofted Drive',
  pull: 'Pull Shot', square_cut: 'Square Cut', straight: 'Straight Drive',
  sweep: 'Sweep Shot',
}

export default function ModelStatusPage() {
  const [status, setStatus] = useState<ModelStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    modelApi.getStatus()
      .then(r => setStatus(r.data))
      .catch(e => setError(e.response?.data?.detail || 'Failed to load model status.'))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="text-center text-white/40 py-20">Loading…</div>
  if (error) return (
    <div className="card p-8 text-center">
      <AlertCircle size={32} className="text-red-400 mx-auto mb-3" />
      <p className="text-white/70">{error}</p>
    </div>
  )
  if (!status) return null

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Model Status</h1>
        <p className="text-white/40 text-sm mt-1">ML classifier status and training instructions</p>
      </div>

      {/* Status banner */}
      <div className={`flex items-start gap-4 rounded-xl border p-5 ${
        status.model_available
          ? 'bg-cricket-green/10 border-cricket-green/20'
          : 'bg-amber-500/10 border-amber-500/20'
      }`}>
        {status.model_available
          ? <CheckCircle size={24} className="text-cricket-green shrink-0 mt-0.5" />
          : <AlertTriangle size={24} className="text-amber-400 shrink-0 mt-0.5" />}
        <div>
          <p className={`font-semibold ${status.model_available ? 'text-cricket-green' : 'text-amber-300'}`}>
            {status.model_available ? 'Model is active' : 'No model trained yet'}
          </p>
          <p className={`text-sm mt-1 ${status.model_available ? 'text-cricket-green/70' : 'text-amber-200/60'}`}>
            {status.message}
          </p>
        </div>
      </div>

      {/* Model details */}
      {status.model_available && (
        <div className="card p-6">
          <h2 className="font-semibold text-white mb-4 flex items-center gap-2">
            <Cpu size={16} className="text-cricket-green" /> Model Details
          </h2>
          <div className="space-y-3 text-sm">
            {[
              ['Name', status.model_name],
              ['Version', status.version],
              ['Architecture', status.architecture],
              ['Held-out Test Accuracy', status.accuracy != null ? `${(status.accuracy * 100).toFixed(1)}%` : null],
              ['Held-out Macro F1', status.macro_f1 != null ? status.macro_f1.toFixed(3) : null],
              ['Training Samples', status.training_samples?.toString()],
              ['Held-out Test Clips', status.test_samples?.toString()],
              ['Median Inference', status.inference_latency_ms != null ? `${status.inference_latency_ms.toFixed(1)} ms/video` : null],
              ['Dataset', status.dataset],
            ].map(([label, value]) => value && (
              <div key={label} className="flex justify-between">
                <span className="text-white/40">{label}</span>
                <span className="text-white/80">{value}</span>
              </div>
            ))}
          </div>

          {status.supported_shots && (
            <div className="mt-4">
              <p className="text-white/40 text-xs mb-2">Supported shots</p>
              <div className="flex flex-wrap gap-2">
                {status.supported_shots.map(s => (
                  <span key={s} className="badge badge-green text-xs">{SHOT_NAMES[s] ?? s.replace(/_/g, ' ')}</span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Model and training details */}
      <div className="card p-6">
        <h2 className="font-semibold text-white mb-4">Visual shot model</h2>
        <p className="text-white/60 text-sm mb-4 leading-relaxed">
          The classifier samples eight RGB frames and feeds ImageNet EfficientNet-B0 embeddings into a trained bidirectional GRU. Pose landmarks are processed separately for technique observations. Class scores are uncalibrated.
        </p>

        <ol className="space-y-4">
          {[
            {
              n: 1,
              title: 'Prepare the dataset manifest',
              body: '.venv/bin/python ml_pipeline/prepare_dataset.py --input-dir dataset/external/cricket-shot --output-dir dataset/prepared/cricket-shot',
            },
            {
              n: 2,
              title: 'Check split readiness',
              body: '.venv/bin/python ml_pipeline/train_model.py --manifest dataset/prepared/cricket-shot/cricket_dataset_manifest.csv --check-only',
            },
            {
              n: 3,
              title: 'Train and evaluate',
              body: '.venv/bin/python ml_pipeline/train_visual_model.py --manifest dataset/prepared/cricket-shot/cricket_dataset_manifest.csv',
            },
            {
              n: 4,
              title: 'Load the saved artifact',
              body: 'Restart the backend. Model Status reports whether the compatible weights and metadata loaded.',
            },
          ].map(({ n, title, body }) => (
            <li key={n} className="flex gap-4">
              <span className="w-7 h-7 rounded-full bg-cricket-green/20 text-cricket-green font-bold text-xs flex items-center justify-center shrink-0 mt-0.5">
                {n}
              </span>
              <div>
                <p className="text-white font-medium text-sm">{title}</p>
                <pre className="text-white/50 text-xs mt-1 whitespace-pre-wrap font-mono leading-relaxed">{body}</pre>
              </div>
            </li>
          ))}
        </ol>
        <p className="text-white/40 text-xs mt-5 leading-relaxed">
          The Hugging Face dataset card declares Apache-2.0, while the acknowledged source videos remain subject to their original rights holders. Check footage rights before redistribution or commercial use.
        </p>
      </div>

      <div className="flex gap-3">
        <Link to="/shots" className="btn-secondary flex-1 py-2.5 text-center">Shot Guide →</Link>
        <Link to="/upload" className="btn-primary flex-1 py-2.5 text-center">Upload Video →</Link>
      </div>
    </div>
  )
}
