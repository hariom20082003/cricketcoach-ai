import { useState, useRef, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { Upload, Video, X, AlertCircle, CheckCircle } from 'lucide-react'
import { videosApi, analysisApi } from '@/services'
import { useAuthStore } from '@/store/auth'

const ACCEPTED = ['video/mp4', 'video/quicktime', 'video/avi', 'video/webm', 'video/x-msvideo']
const MAX_SIZE_MB = 100

function formatSize(bytes: number) {
  return bytes < 1024 * 1024
    ? `${(bytes / 1024).toFixed(1)} KB`
    : `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export default function UploadPage() {
  const navigate = useNavigate()
  const user = useAuthStore((s) => s.user)
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [battingHand, setBattingHand] = useState(user?.batting_hand || 'right')
  const [recordingAngle, setRecordingAngle] = useState('side-on')
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const [fileError, setFileError] = useState('')
  const [recording, setRecording] = useState(false)
  const [webcamStream, setWebcamStream] = useState<MediaStream | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const inputRef = useRef<HTMLInputElement>(null)

  const validateFile = (f: File): string => {
    if (!ACCEPTED.includes(f.type) && !f.name.match(/\.(mp4|mov|avi|webm)$/i)) {
      return 'Unsupported file type. Please upload MP4, MOV, AVI, or WebM.'
    }
    if (f.size > MAX_SIZE_MB * 1024 * 1024) return `File too large. Maximum size is ${MAX_SIZE_MB}MB.`
    return ''
  }

  const selectFile = (f: File) => {
    const err = validateFile(f)
    if (err) { setFileError(err); return }
    setFileError('')
    setFile(f)
    setPreview(URL.createObjectURL(f))
  }

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const f = e.dataTransfer.files[0]
    if (f) selectFile(f)
  }, [])

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (f) selectFile(f)
  }

  const clearFile = () => {
    setFile(null)
    if (preview) URL.revokeObjectURL(preview)
    setPreview(null)
    setFileError('')
    if (inputRef.current) inputRef.current.value = ''
  }

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
      setWebcamStream(stream)
      if (videoRef.current) videoRef.current.srcObject = stream
    } catch {
      setError('Cannot access webcam. Please check browser permissions.')
    }
  }

  const startRecording = () => {
    if (!webcamStream) return
    chunksRef.current = []
    const mr = new MediaRecorder(webcamStream, { mimeType: 'video/webm' })
    mr.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data) }
    mr.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: 'video/webm' })
      const f = new File([blob], `webcam_${Date.now()}.webm`, { type: 'video/webm' })
      selectFile(f)
      webcamStream.getTracks().forEach(t => t.stop())
      setWebcamStream(null)
      setRecording(false)
    }
    mr.start()
    mediaRecorderRef.current = mr
    setRecording(true)
  }

  const stopRecording = () => mediaRecorderRef.current?.stop()

  const handleSubmit = async () => {
    if (!file) return
    setError('')
    setUploading(true)
    try {
      const uploadRes = await videosApi.upload(file, battingHand, recordingAngle, setUploadProgress)
      const videoId = uploadRes.data.id
      const jobRes = await analysisApi.start(videoId)
      navigate(`/analysis/${jobRes.data.id}`)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Upload failed. Please try again.')
      setUploading(false)
      setUploadProgress(0)
    }
  }

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Analyse a Shot</h1>
        <p className="text-white/50 mt-1">Upload a video or record from your webcam</p>
      </div>

      {error && (
        <div className="flex items-start gap-3 bg-red-500/10 border border-red-500/20 rounded-xl p-4">
          <AlertCircle className="text-red-400 shrink-0 mt-0.5" size={16} />
          <p className="text-red-300 text-sm">{error}</p>
        </div>
      )}

      {/* File picker / drag zone */}
      {!file && !webcamStream && (
        <div
          className="border-2 border-dashed border-white/15 rounded-xl p-10 text-center hover:border-cricket-green/40 transition-colors cursor-pointer"
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
        >
          <Upload size={36} className="text-white/30 mx-auto mb-3" />
          <p className="text-white/70 font-medium">Drag a video here, or click to browse</p>
          <p className="text-white/30 text-sm mt-1">MP4, MOV, AVI, WebM · Max {MAX_SIZE_MB}MB</p>
          <input ref={inputRef} type="file" accept="video/*" className="hidden" onChange={handleFileInput} />
          {fileError && <p className="text-red-400 text-sm mt-3">{fileError}</p>}
        </div>
      )}

      {/* Webcam */}
      {webcamStream && (
        <div className="card overflow-hidden">
          <video ref={videoRef} autoPlay muted className="w-full rounded-t-xl" />
          <div className="p-4 flex gap-3">
            {!recording
              ? <button onClick={startRecording} className="btn-primary px-5 py-2 text-sm">● Start Recording</button>
              : <button onClick={stopRecording} className="btn-secondary px-5 py-2 text-sm">■ Stop Recording</button>
            }
          </div>
        </div>
      )}

      {/* Webcam button */}
      {!file && !webcamStream && (
        <button onClick={startWebcam} className="btn-secondary w-full py-3 flex items-center justify-center gap-2">
          <Video size={18} /> Record from Webcam
        </button>
      )}

      {/* File preview */}
      {file && preview && (
        <div className="card overflow-hidden">
          <video src={preview} controls className="w-full max-h-64 bg-black" />
          <div className="p-4 flex items-center justify-between">
            <div>
              <p className="text-white font-medium text-sm">{file.name}</p>
              <p className="text-white/40 text-xs">{formatSize(file.size)}</p>
            </div>
            <button onClick={clearFile} className="p-2 rounded-lg hover:bg-white/5 text-white/40 hover:text-white">
              <X size={18} />
            </button>
          </div>
        </div>
      )}

      {/* Options */}
      <div className="card p-6 space-y-4">
        <h2 className="text-white font-semibold">Recording details</h2>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="label">Batting hand</label>
            <select className="input" value={battingHand} onChange={e => setBattingHand(e.target.value)}>
              <option value="right">Right-handed</option>
              <option value="left">Left-handed</option>
            </select>
          </div>
          <div>
            <label className="label">Recording angle</label>
            <select className="input" value={recordingAngle} onChange={e => setRecordingAngle(e.target.value)}>
              <option value="side-on">Side-on</option>
              <option value="front-on">Front-on</option>
              <option value="overhead">Overhead</option>
              <option value="unknown">Unknown</option>
            </select>
          </div>
        </div>
      </div>

      {/* Upload progress */}
      {uploading && (
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-white/60">Uploading…</span>
            <span className="text-cricket-green">{uploadProgress}%</span>
          </div>
          <div className="h-2 bg-white/10 rounded-full overflow-hidden">
            <div
              className="h-full bg-cricket-green rounded-full transition-all duration-200"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
        </div>
      )}

      <button
        onClick={handleSubmit}
        disabled={!file || uploading}
        className="btn-primary w-full py-3 flex items-center justify-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed"
      >
        {uploading ? 'Uploading…' : (
          <><CheckCircle size={18} /> Submit for Analysis</>
        )}
      </button>
    </div>
  )
}
