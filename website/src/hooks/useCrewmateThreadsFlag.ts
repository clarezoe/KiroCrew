/**
 * `dashboard.crewmate_threads` -- whether reply threads on crewmate chat messages
 * are on for this gateway. A server-side config flag (`config/sections.py`),
 * off by default, read from the shared `['kirocrewConfig']` cache so the Members
 * page and the Settings toggle agree without a second request.
 *
 * Off (or before the config has loaded, or when the read fails): `false`. The
 * routes behind the feature answer 404 while it is off, so a control drawn on
 * a stale `true` would only reach a refusal; defaulting closed is the honest
 * reading of "not known to be on".
 */
import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'

export const CREWMATE_THREADS_CONFIG_KEY = 'dashboard.crewmate_threads'

type KirocrewConfigThreads = { dashboard?: { crewmate_threads?: boolean } }

export function useCrewmateThreadsFlag(): boolean {
  const { data } = useQuery<KirocrewConfigThreads, Error, boolean>({
    queryKey: ['kirocrewConfig'],
    queryFn: () => api.kirocrewConfig(),
    select: (c) => c?.dashboard?.crewmate_threads === true,
  })
  return data === true
}
