/**
 * Settings > Developer > Crewmates -- the `dashboard.crewmate_threads` toggle and
 * the hook the Members page reads the same flag through.
 *
 * Off by default; the switch PATCHes the config key and the shared config query
 * is refetched so the flag's readers move with it; a refused save says so.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, fireEvent, waitFor, renderHook } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import React from 'react'

vi.mock('../../api/client', () => ({
  api: {
    kirocrewConfig: vi.fn(),
    patchConfig: vi.fn(),
  },
}))

import { api } from '../../api/client'
import { CrewmatesSection } from './CrewmatesSection'
import { useCrewmateThreadsFlag } from '../../hooks/useCrewmateThreadsFlag'

let qc: QueryClient
const wrap = (ui: React.ReactElement) => render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>)

beforeEach(() => {
  vi.clearAllMocks()
  qc = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  vi.mocked(api.kirocrewConfig).mockResolvedValue({})
  vi.mocked(api.patchConfig).mockResolvedValue({ ok: true })
})

describe('useCrewmateThreadsFlag', () => {
  it('is off until the config says otherwise, and only for an explicit true', async () => {
    const { result } = renderHook(() => useCrewmateThreadsFlag(), {
      wrapper: ({ children }) => <QueryClientProvider client={qc}>{children}</QueryClientProvider>,
    })
    expect(result.current).toBe(false)
    await waitFor(() => expect(api.kirocrewConfig).toHaveBeenCalled())
    expect(result.current).toBe(false)

    qc.clear()
    vi.mocked(api.kirocrewConfig).mockResolvedValue({ dashboard: { crewmate_threads: true } })
    const on = renderHook(() => useCrewmateThreadsFlag(), {
      wrapper: ({ children }) => <QueryClientProvider client={qc}>{children}</QueryClientProvider>,
    })
    await waitFor(() => expect(on.result.current).toBe(true))
  })
})

describe('CrewmatesSection', () => {
  it('renders the reply-threads switch off by default and saves the flag on toggle', async () => {
    wrap(<CrewmatesSection />)
    const toggle = await screen.findByRole('switch', { name: /Reply threads on crewmate chat messages/ })
    await waitFor(() => expect(toggle).not.toBeDisabled())
    expect(toggle).toHaveAttribute('aria-checked', 'false')
    fireEvent.click(toggle)
    await waitFor(() => expect(api.patchConfig).toHaveBeenCalledWith('dashboard.crewmate_threads', true))
    // The shared config query is refetched, so the Members page's hook sees the new value.
    await waitFor(() => expect(api.kirocrewConfig).toHaveBeenCalledTimes(2))
  })

  it('a refused save is said in plain words and the switch stays where the server has it', async () => {
    vi.mocked(api.patchConfig).mockRejectedValue(new Error('boom'))
    wrap(<CrewmatesSection />)
    const toggle = await screen.findByRole('switch', { name: /Reply threads on crewmate chat messages/ })
    await waitFor(() => expect(toggle).not.toBeDisabled())
    fireEvent.click(toggle)
    await screen.findByText("Couldn't save the crewmate setting. Try again.")
    expect(toggle).toHaveAttribute('aria-checked', 'false')
  })
})
