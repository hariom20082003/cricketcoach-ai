import { Outlet, Link, useLocation, useNavigate } from 'react-router-dom'
import { useAuthStore } from '@/store/auth'
import { authApi } from '@/services'
import {
  LayoutDashboard, Upload, History, TrendingUp, Dumbbell,
  BookOpen, User, Cpu, LogOut, Menu, X, ChevronRight
} from 'lucide-react'
import { useState } from 'react'

const NAV_ITEMS = [
  { path: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/upload', label: 'Analyze Shot', icon: Upload },
  { path: '/history', label: 'History', icon: History },
  { path: '/progress', label: 'Progress', icon: TrendingUp },
  { path: '/drills', label: 'Drills', icon: Dumbbell },
  { path: '/shots', label: 'Shot Guide', icon: BookOpen },
  { path: '/model-status', label: 'Model Status', icon: Cpu },
]

export default function Layout() {
  const { user, clearAuth } = useAuthStore()
  const location = useLocation()
  const navigate = useNavigate()
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const handleLogout = async () => {
    try {
      await authApi.logout()
    } catch {
      // Always clear local credentials even if the API is temporarily unreachable.
    } finally {
      clearAuth()
      navigate('/login')
    }
  }

  return (
    <div className="flex h-screen bg-cricket-charcoal overflow-hidden">
      {/* Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-50 w-64 bg-cricket-navy border-r border-white/5 transform transition-transform duration-200 lg:relative lg:translate-x-0 ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* Logo */}
        <div className="flex items-center gap-3 px-6 py-5 border-b border-white/5">
          <div className="w-8 h-8 rounded-lg bg-cricket-green flex items-center justify-center">
            <span className="text-white font-bold text-sm">🏏</span>
          </div>
          <div>
            <div className="font-bold text-white text-sm">CricketCoach</div>
            <div className="text-cricket-green text-xs font-medium">AI</div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-4 py-4 space-y-1 overflow-y-auto">
          {NAV_ITEMS.map(({ path, label, icon: Icon }) => {
            const active = location.pathname === path || location.pathname.startsWith(path + '/')
            return (
              <Link
                key={path}
                to={path}
                onClick={() => setSidebarOpen(false)}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-150 group ${
                  active
                    ? 'bg-cricket-green/20 text-cricket-green-light border border-cricket-green/20'
                    : 'text-white/60 hover:text-white hover:bg-white/5'
                }`}
              >
                <Icon size={18} className={active ? 'text-cricket-green' : 'text-white/40 group-hover:text-white/70'} />
                {label}
                {active && <ChevronRight size={14} className="ml-auto text-cricket-green/60" />}
              </Link>
            )
          })}
        </nav>

        {/* User section */}
        <div className="border-t border-white/5 p-4">
          <Link
            to="/profile"
            onClick={() => setSidebarOpen(false)}
            className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-white/5 transition-colors mb-1"
          >
            <div className="w-8 h-8 rounded-full bg-cricket-green/20 flex items-center justify-center">
              <User size={16} className="text-cricket-green" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-sm font-medium text-white truncate">{user?.username}</div>
              <div className="text-xs text-white/40 truncate">{user?.experience_level}</div>
            </div>
          </Link>
          <button
            onClick={handleLogout}
            className="flex items-center gap-3 px-3 py-2 rounded-lg w-full text-white/50 hover:text-red-400 hover:bg-red-500/10 transition-colors text-sm"
          >
            <LogOut size={16} />
            Sign out
          </button>
        </div>
      </aside>

      {/* Overlay for mobile sidebar */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main content */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Top bar (mobile) */}
        <header className="lg:hidden flex items-center gap-4 px-4 py-3 bg-cricket-navy border-b border-white/5">
          <button
            onClick={() => setSidebarOpen(true)}
            className="p-2 rounded-lg hover:bg-white/10 text-white/70 hover:text-white"
          >
            <Menu size={20} />
          </button>
          <div className="flex items-center gap-2">
            <span>🏏</span>
            <span className="font-bold text-white text-sm">CricketCoach AI</span>
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-y-auto p-4 lg:p-8">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
