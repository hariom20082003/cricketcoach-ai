import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { User } from '@/types'

interface AuthState {
  user: User | null
  token: string | null
  setAuth: (user: User, token: string) => void
  clearAuth: () => void
  updateUser: (user: Partial<User>) => void
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      token: null,
      setAuth: (user, token) => {
        localStorage.setItem('access_token', token)
        set({ user, token })
      },
      clearAuth: () => {
        localStorage.removeItem('access_token')
        set({ user: null, token: null })
      },
      updateUser: (data) =>
        set((state) => ({ user: state.user ? { ...state.user, ...data } : null })),
    }),
    {
      name: 'cricket-auth',
      partialize: (state) => ({ user: state.user, token: state.token }),
    }
  )
)
