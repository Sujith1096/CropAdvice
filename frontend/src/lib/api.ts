import { supabase } from "./supabase"

export interface ApiFetchOptions extends RequestInit {
  timeoutMs?: number
}

export async function parseApiError(response: Response, fallback = "Request failed"): Promise<string> {
  try {
    const text = await response.text()
    if (!text) return `${fallback} (HTTP ${response.status})`
    try {
      const data = JSON.parse(text)
      return data.detail || data.message || fallback
    } catch {
      return `${fallback} (HTTP ${response.status})`
    }
  } catch {
    return fallback
  }
}

export async function apiFetch(endpoint: string, options: ApiFetchOptions = {}) {
  const baseUrl = import.meta.env.VITE_API_URL || "http://localhost:8000"
  const { timeoutMs = 45000, signal: externalSignal, ...fetchOptions } = options

  const { data: { session } } = await supabase.auth.getSession()
  const token = session?.access_token

  const headers = new Headers(fetchOptions.headers || {})

  if (token && !headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${token}`)
  }

  const controller = new AbortController()
  let timedOut = false
  const timeoutId = window.setTimeout(() => {
    timedOut = true
    controller.abort()
  }, timeoutMs)

  const onExternalAbort = () => controller.abort()
  if (externalSignal) {
    if (externalSignal.aborted) {
      controller.abort()
    } else {
      externalSignal.addEventListener("abort", onExternalAbort, { once: true })
    }
  }

  try {
    const response = await fetch(`${baseUrl}${endpoint}`, {
      ...fetchOptions,
      headers,
      signal: controller.signal,
    })

    if (response.status === 401) {
      if (window.location.pathname !== "/login" && window.location.pathname !== "/register") {
        await supabase.auth.signOut()
        window.location.href = "/login"
      }
    }

    return response
  } catch (err: any) {
    if (timedOut && err?.name === "AbortError") {
      throw new Error("Request timed out. The server may be waking up or busy — please try again.")
    }
    throw err
  } finally {
    window.clearTimeout(timeoutId)
    externalSignal?.removeEventListener("abort", onExternalAbort)
  }
}
