import { useState, FormEvent } from 'react'
import { useAuthStore } from '@/store/auth'
import { authApi } from '@/services'
import { User, CheckCircle, Trash2 } from 'lucide-react'
import { useNavigate } from 'react-router-dom'

export default function ProfilePage() {
  const { user, updateUser } = useAuthStore()
  const clearAuth = useAuthStore((state) => state.clearAuth)
  const navigate = useNavigate()
  const [form, setForm] = useState({
    full_name: user?.full_name || '',
    batting_hand: user?.batting_hand || 'right',
    experience_level: user?.experience_level || 'beginner',
  })
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState('')
  const [deletingAccount, setDeletingAccount] = useState(false)

  const set = (k: string) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
    setForm(f => ({ ...f, [k]: e.target.value }))

  const handleSave = async (e: FormEvent) => {
    e.preventDefault()
    setSaving(true); setError(''); setSaved(false)
    try {
      const res = await authApi.updateMe(form)
      updateUser(res.data)
      setSaved(true)
      setTimeout(() => setSaved(false), 3000)
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to save profile.')
    } finally {
      setSaving(false)
    }
  }

  const handleDeleteAccount = async () => {
    const confirmed = window.confirm('Permanently delete your account, videos, analysis results, notes, and progress? This cannot be undone.')
    if (!confirmed) return
    setDeletingAccount(true)
    setError('')
    try {
      await authApi.deleteMe()
      clearAuth()
      navigate('/')
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Account deletion failed. Please try again.')
    } finally {
      setDeletingAccount(false)
    }
  }

  return (
    <div className="max-w-xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Profile</h1>
        <p className="text-white/40 text-sm mt-1">Manage your account settings</p>
      </div>

      {/* Avatar */}
      <div className="card p-6 flex items-center gap-5">
        <div className="w-16 h-16 rounded-full bg-cricket-green/20 flex items-center justify-center">
          <User size={28} className="text-cricket-green" />
        </div>
        <div>
          <p className="text-white font-semibold">{user?.username}</p>
          <p className="text-white/40 text-sm">{user?.email}</p>
          <p className="text-white/30 text-xs mt-0.5">
            Member since {user?.created_at ? new Date(user.created_at).toLocaleDateString() : '–'}
          </p>
        </div>
      </div>

      {/* Edit form */}
      <div className="card p-6">
        <h2 className="font-semibold text-white mb-5">Edit Profile</h2>

        {error && (
          <div className="bg-red-500/10 border border-red-500/20 text-red-400 text-sm rounded-lg px-4 py-3 mb-4">
            {error}
          </div>
        )}
        {saved && (
          <div className="bg-cricket-green/10 border border-cricket-green/20 text-cricket-green text-sm rounded-lg px-4 py-3 mb-4 flex items-center gap-2">
            <CheckCircle size={15} /> Profile saved!
          </div>
        )}

        <form onSubmit={handleSave} className="space-y-4">
          <div>
            <label className="label">Full Name</label>
            <input type="text" className="input" value={form.full_name} onChange={set('full_name')} placeholder="Sachin Tendulkar" />
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
              <label className="label">Experience Level</label>
              <select className="input" value={form.experience_level} onChange={set('experience_level')}>
                <option value="beginner">Beginner</option>
                <option value="intermediate">Intermediate</option>
                <option value="advanced">Advanced</option>
              </select>
            </div>
          </div>
          <button type="submit" className="btn-primary px-8 py-2.5" disabled={saving}>
            {saving ? 'Saving…' : 'Save Changes'}
          </button>
        </form>
      </div>

      <div className="card p-6 border border-red-500/20">
        <h2 className="font-semibold text-red-300 mb-2">Delete account and data</h2>
        <p className="text-white/45 text-sm mb-4">
          Permanently removes your account, uploaded videos, saved analyses, notes, and progress from application storage.
        </p>
        <button
          type="button"
          onClick={handleDeleteAccount}
          disabled={deletingAccount}
          className="inline-flex items-center gap-2 rounded-lg border border-red-500/30 px-4 py-2 text-sm text-red-300 hover:bg-red-500/10 disabled:opacity-50"
        >
          <Trash2 size={15} />
          {deletingAccount ? 'Deleting account…' : 'Delete my account and data'}
        </button>
      </div>

      {/* Password section */}
      <div className="card p-6">
        <h2 className="font-semibold text-white mb-2">Password</h2>
        <p className="text-white/40 text-sm">
          Password changes and reset are not available in this demonstration. Contact the demonstration operator if you lose access.
        </p>
      </div>

      {/* Account info */}
      <div className="card p-6">
        <h2 className="font-semibold text-white mb-3">Account</h2>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-white/40">Email</span>
            <span className="text-white/70">{user?.email}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-white/40">Role</span>
            <span className="text-white/70 capitalize">{user?.role}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-white/40">Status</span>
            <span className="text-cricket-green">Active</span>
          </div>
        </div>
      </div>
    </div>
  )
}
