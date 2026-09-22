/**
 * Settings > Developer > **Crewmates**: the server-side switches for crewmate
 * features. One today -- `dashboard.crewmate_threads`, reply threads on crewmate
 * chat messages (RFC crewmates launch, screen 07), off by default.
 *
 * Sits under its own heading, right after Feature Previews, because that is
 * where the Crew Members preview card lives -- the one door to the `/members`
 * page this flag changes. The heading is visual only: config has no crewmate
 * parent flag (Crew Members is a client-side preview flag, `previewFlags.ts`),
 * so this toggle nests under a section title rather than under a parent key.
 * Whether crewmate features should share a parent flag is out of scope here.
 *
 * Server-side, like the Power toggle on the Chat tab: the value is PATCHed to
 * `dashboard.crewmate_threads` and read back through the shared
 * `['kirocrewConfig']` query the Members page reads the flag from
 * (`useCrewmateThreadsFlag`), so the Reply in thread control appears or goes
 * on the next render without a reload.
 */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { api } from '../../api/client'
import ErrorNotice from '../../components/ErrorNotice'
import { SettingsCard, SettingsSection, SettingsToggle } from '../../components/settings'
import { CREWMATE_THREADS_CONFIG_KEY } from '../../hooks/useCrewmateThreadsFlag'
import { i18nT } from '../../i18n/t'

type KirocrewConfigShape = { dashboard?: { crewmate_threads?: boolean } }

export function CrewmatesSection() {
  const qc = useQueryClient()
  const cfgQ = useQuery<KirocrewConfigShape>({
    queryKey: ['kirocrewConfig'],
    queryFn: () => api.kirocrewConfig(),
  })
  const [saveError, setSaveError] = useState('')
  const threads = cfgQ.data?.dashboard?.crewmate_threads === true
  const threadsMut = useMutation({
    mutationFn: (v: boolean) => api.patchConfig(CREWMATE_THREADS_CONFIG_KEY, v),
    onSuccess: () => {
      setSaveError('')
      void qc.invalidateQueries({ queryKey: ['kirocrewConfig'] })
    },
    onError: () => setSaveError(i18nT('pages.settings.crewmatesSection.failed_to_save')),
  })

  return (
    <SettingsSection title={i18nT('pages.settings.crewmatesSection.title')}>
      {saveError && (
        <ErrorNotice message={saveError} onDismiss={() => setSaveError('')} className="mb-2" />
      )}
      <SettingsCard>
        <SettingsToggle
          label={i18nT('pages.settings.crewmatesSection.reply_threads')}
          description={i18nT('pages.settings.crewmatesSection.reply_threads_desc')}
          checked={threads}
          onChange={(v) => threadsMut.mutate(v)}
          disabled={!cfgQ.isSuccess || threadsMut.isPending}
          configKey={CREWMATE_THREADS_CONFIG_KEY}
        />
      </SettingsCard>
    </SettingsSection>
  )
}
