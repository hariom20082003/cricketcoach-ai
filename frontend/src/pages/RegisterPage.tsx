import { useState, FormEvent } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuthStore } from '@/store/auth'
import { authApi } from '@/services'
import { getApiErrorMessage } from '@/services/errors'

export default function RegisterPage() {
  const navigate = useNavigate()
  const setAuth = useAuthStore((s) => s.setAuth)
  const [form, setForm] = useState({
    email: '', username: '', password: '', full_name: '',
    batting_hand: 'right', experience_level: 'beginner',
  })
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const set = (k: string) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
    setForm((f) => ({ ...f, [k]: e.target.value }))

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    if (form.password.length < 8) { setError('Password must be at least 8 characters.'); return }
    if (new TextEncoder().encode(form.password).length > 72) {
      setError('Password must be no more than 72 UTF-8 bytes.')
      return
    }
    setError('')
    setLoading(true)
    try {
      const res = await authApi.register(form)
      setAuth(res.data.user, res.data.access_token)
      navigate('/dashboard')
    } catch (err: any) {
      setError(getApiErrorMessage(err, 'Registration failed.'))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-cricket-charcoal flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <div className="text-4xl mb-2">🏏</div>
          <h1 className="text-2xl font-bold text-white">Create an account</h1>
          <p className="text-white/50 text-sm mt-1">Start your batting analysis journey</p>
        </div>

        <div className="card p-8">
          {error && (
            <div className="bg-red-500/10 border border-red-500/20 text-red-400 text-sm rounded-lg px-4 py-3 mb-5">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="label">Email *</label>
                <input type="email" className="input" placeholder="you@example.com"
                  value={form.email} onChange={set('email')} required />
              </div>
              <div>
                <label className="label">Username *</label>
                <input type="text" className="input" placeholder="batsman99"
                  value={form.username} onChange={set('username')} required minLength={3} />
              </div>
            </div>

            <div>
              <label className="label">Full Name (optional)</label>
              <input type="text" className="input" placeholder="Sachin Tendulkar"
                value={form.full_name} onChange={set('full_name')} />
            </div>

            <div>
              <label className="label">Password *</label>
                <input type="password" className="input" placeholder="At least 8 characters"
                  value={form.password} onChange={set('password')} required minLength={8} />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="label">Batting Hand</label>
                <select className="input" value={form.batting_hand} onChange={set('batting_hand')}>
                  <option value="right">Right-handed</option>
                  <option value="left">Left-handed</option>
                </select>
              </div>
              <div>
                <label className="label">Experience</label>
                <select className="input" value={form.experience_level} onChange={set('experience_level')}>
                  <option value="beginner">Beginner</option>
                  <option value="intermediate">Intermediate</option>
                  <option value="advanced">Advanced</option>
                </select>
              </div>
            </div>

            <button type="submit" className="btn-primary w-full py-3 mt-2" disabled={loading}>
              {loading ? 'Creating account…' : 'Create Account'}
            </button>
          </form>

          <p className="text-white/35 text-xs mt-5">
            By creating an account, you acknowledge the <Link to="/privacy" className="text-cricket-green hover:underline">Privacy Notice</Link>.
          </p>

          <p className="text-center text-white/40 text-sm mt-6">
            Already have an account?{' '}
            <Link to="/login" className="text-cricket-green hover:underline">Sign in</Link>
          </p>
        </div>
      </div>
    </div>
  )
}
