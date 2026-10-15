import axios from 'axios'

const configuredApiOrigin = import.meta.env.VITE_API_URL?.replace(/\/$/, '')

const api = axios.create({
  baseURL: configuredApiOrigin ? `${configuredApiOrigin}/api` : '/api',
  timeout: 60000,
})

// Request interceptor: attach JWT token
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Response interceptor: handle 401
api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem('access_token')
      localStorage.removeItem('user')
      localStorage.removeItem('cricket-auth')
      const requestPath = err.config?.url || ''
      const isCredentialRequest = requestPath.endsWith('/auth/login') || requestPath.endsWith('/auth/register')
      if (!isCredentialRequest) {
        window.location.assign(`${import.meta.env.BASE_URL}login`)
      }
    }
    return Promise.reject(err)
  }
)

export default api
