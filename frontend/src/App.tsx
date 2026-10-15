import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { useAuthStore } from '@/store/auth'
import Layout from '@/components/layout/Layout'
import LandingPage from '@/pages/LandingPage'
import LoginPage from '@/pages/LoginPage'
import RegisterPage from '@/pages/RegisterPage'
import DashboardPage from '@/pages/DashboardPage'
import UploadPage from '@/pages/UploadPage'
import AnalysisPage from '@/pages/AnalysisPage'
import ResultsPage from '@/pages/ResultsPage'
import HistoryPage from '@/pages/HistoryPage'
import ProgressPage from '@/pages/ProgressPage'
import DrillsPage from '@/pages/DrillsPage'
import DrillDetailPage from '@/pages/DrillDetailPage'
import ShotsGuidePage from '@/pages/ShotsGuidePage'
import ShotDetailPage from '@/pages/ShotDetailPage'
import ProfilePage from '@/pages/ProfilePage'
import ModelStatusPage from '@/pages/ModelStatusPage'
import PrivacyPage from '@/pages/PrivacyPage'

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const user = useAuthStore((s) => s.user)
  if (!user) return <Navigate to="/login" replace />
  return <>{children}</>
}

function PublicOnlyRoute({ children }: { children: React.ReactNode }) {
  const user = useAuthStore((s) => s.user)
  if (user) return <Navigate to="/dashboard" replace />
  return <>{children}</>
}

export default function App() {
  return (
    <BrowserRouter basename={import.meta.env.BASE_URL}>
      <Routes>
        {/* Public */}
        <Route path="/" element={<LandingPage />} />
        <Route path="/privacy" element={<PrivacyPage />} />
        <Route path="/login" element={<PublicOnlyRoute><LoginPage /></PublicOnlyRoute>} />
        <Route path="/register" element={<PublicOnlyRoute><RegisterPage /></PublicOnlyRoute>} />

        {/* Protected */}
        <Route element={<ProtectedRoute><Layout /></ProtectedRoute>}>
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/analysis/:jobId" element={<AnalysisPage />} />
          <Route path="/results/:jobId" element={<ResultsPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/progress" element={<ProgressPage />} />
          <Route path="/drills" element={<DrillsPage />} />
          <Route path="/drills/:drillId" element={<DrillDetailPage />} />
          <Route path="/shots" element={<ShotsGuidePage />} />
          <Route path="/shots/:shotId" element={<ShotDetailPage />} />
          <Route path="/profile" element={<ProfilePage />} />
          <Route path="/model-status" element={<ModelStatusPage />} />
        </Route>

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
