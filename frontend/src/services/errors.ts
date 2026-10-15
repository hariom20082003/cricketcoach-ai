export function getApiErrorMessage(error: unknown, fallback: string): string {
  const detail = (error as any)?.response?.data?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => typeof item?.msg === 'string' ? item.msg : '')
      .filter(Boolean)
    if (messages.length) return messages.join(' ')
  }
  return fallback
}
