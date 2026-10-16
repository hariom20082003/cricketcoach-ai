import { Link } from 'react-router-dom'
import { Camera, Brain, TrendingUp, Dumbbell, ArrowRight } from 'lucide-react'

const features = [
  {
    icon: Camera,
    title: 'Shot Classification',
    desc: 'Upload or record your batting. A trained visual-temporal model classifies the shot from sampled video frames.',
  },
  {
    icon: Brain,
    title: 'Pose Analysis',
    desc: 'MediaPipe Pose extracts body landmarks to measure joint angles and movement patterns.',
  },
  {
    icon: TrendingUp,
    title: 'Technique Coaching',
    desc: 'Automated pose measurements and rule-based suggestions flag possible areas to review. They are not expert-validated coaching judgments.',
  },
  {
    icon: Dumbbell,
    title: 'Practice Drills',
    desc: 'Curated drills mapped to your weaknesses. Track completions and build consistency.',
  },
]

export default function LandingPage() {
  const backendConfigured = Boolean(import.meta.env.VITE_API_URL?.trim())

  return (
    <div className="min-h-screen bg-cricket-charcoal text-white">
      {!backendConfigured && (
        <div role="status" className="bg-amber-400 px-6 py-3 text-center text-sm font-medium text-cricket-charcoal">
          Frontend preview only — account, upload, and analysis features will be available after the backend is deployed.
        </div>
      )}

      {/* Navbar */}
      <nav className="flex items-center justify-between px-8 py-5 border-b border-white/5">
        <div className="flex items-center gap-2 font-bold text-lg">
          <span>🏏</span> CricketCoach <span className="text-cricket-green">AI</span>
        </div>
        <div className="flex gap-3">
          {backendConfigured && (
            <>
              <Link to="/login" className="btn-secondary text-sm px-5 py-2">Sign In</Link>
              <Link to="/register" className="btn-primary text-sm px-5 py-2">Get Started</Link>
            </>
          )}
        </div>
      </nav>

      {/* Hero */}
      <section className="max-w-4xl mx-auto text-center px-6 py-24">
        <div className="inline-block bg-cricket-green/10 border border-cricket-green/20 text-cricket-green text-xs font-semibold px-4 py-1.5 rounded-full mb-6">
          ML-Powered Batting Analysis
        </div>
        <h1 className="text-5xl md:text-6xl font-bold mb-6 leading-tight">
          Master Your Batting<br />
          <span className="text-cricket-green">Technique with AI</span>
        </h1>
        <p className="text-xl text-white/60 mb-10 max-w-2xl mx-auto">
          Upload a batting video, identify the shot, inspect pose landmarks, and receive
          automated pose measurements and rule-based suggestions. These observations can help you review a clip, but are not expert-validated coaching advice.
        </p>
        {backendConfigured ? (
          <>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link to="/register" className="btn-primary text-base px-8 py-3 flex items-center gap-2 justify-center">
                Get Started Free <ArrowRight size={18} />
              </Link>
              <Link to="/login" className="btn-secondary text-base px-8 py-3">
                Sign In
              </Link>
            </div>
            <p className="text-white/30 text-sm mt-6">
              Ten shot classes · visual-temporal model · uncalibrated class scores.
            </p>
          </>
        ) : (
          <p className="text-amber-200/80 text-sm mt-6">
            The interactive coaching app is not connected yet.
          </p>
        )}
      </section>

      {/* Honest disclaimer */}
      <section className="max-w-3xl mx-auto px-6 mb-16">
        <div className="bg-cricket-surface border border-white/10 rounded-xl p-6 text-sm text-white/60 space-y-1">
          <p className="font-semibold text-white/80 mb-2">Transparency</p>
          <p>✓ Pose landmarks are measured from video — not estimated from rules.</p>
          <p>✓ Shot classification uses a trained visual-temporal model. Its class scores are uncalibrated.</p>
          <p>✓ Technique observations come from automated pose measurements and rule-based suggestions; they have not been validated by cricket coaches.</p>
          <p>✓ Pose estimation alone cannot confirm bat face, ball contact, or timing.</p>
        </div>
      </section>

      {/* Features */}
      <section className="max-w-5xl mx-auto px-6 py-16">
        <h2 className="text-3xl font-bold text-center mb-12">Everything a beginner needs</h2>
        <div className="grid md:grid-cols-2 gap-6">
          {features.map(({ icon: Icon, title, desc }) => (
            <div key={title} className="card p-6 flex gap-4">
              <div className="w-10 h-10 rounded-lg bg-cricket-green/15 flex items-center justify-center shrink-0">
                <Icon size={20} className="text-cricket-green" />
              </div>
              <div>
                <h3 className="font-semibold text-white mb-1">{title}</h3>
                <p className="text-white/55 text-sm leading-relaxed">{desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="text-center px-6 py-20">
        <h2 className="text-3xl font-bold mb-4">Ready to improve your batting?</h2>
        {backendConfigured ? (
          <>
            <p className="text-white/50 mb-8">Create a free account and upload your first video in minutes.</p>
            <Link to="/register" className="btn-primary text-base px-10 py-3 inline-flex items-center gap-2">
              Start Analysing <ArrowRight size={18} />
            </Link>
          </>
        ) : (
          <p className="text-amber-200/80">Interactive coaching will be available after the backend is deployed.</p>
        )}
      </section>

      {/* Footer */}
      <footer className="border-t border-white/5 px-8 py-6 text-center text-white/30 text-sm">
        <div>CricketCoach AI — Engineering College ML Project. Not affiliated with any cricket board.</div>
        <Link to="/privacy" className="inline-block mt-2 text-cricket-green/70 hover:text-cricket-green hover:underline">Privacy Notice</Link>
      </footer>
    </div>
  )
}
