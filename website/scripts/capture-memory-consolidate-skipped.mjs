/**
 * Screenshot harness for the Memory tab's "Summarize now" tally when a target is
 * a Temporary or Incognito session.
 *
 * Runs the REAL built SPA (website/dist) behind the shared in-process static
 * server with every /api/** answered from fixtures (gateway-free).
 *
 * The scenario: `api.sessions` lists three transcripts, one of them a Slack
 * thread the user marked `!incognito`. `POST /api/memory/consolidate` refuses that
 * one with `403 {code: "restricted_target_session"}` -- the refusal the memory
 * modes promise, not a failure of the press. The tally must count it as SKIPPED
 * and keep the success tone; with a fourth session whose request genuinely fails
 * (500), the failed tally names both counts and renders through ErrorNotice,
 * with the failed session named under a label; with five failing sessions every
 * one is named, in the notice, in the DOM.
 *
 * Frames:
 *   <prefix>-1-skipped.png         2 ok + 1 refused  -> ok tone, "1 skipped as temporary or incognito"
 *   <prefix>-2-failed-skipped.png  1 ok + 1 refused + 1 failed -> ErrorNotice banner, "Failed: <key>", stays until dismissed
 *   <prefix>-3-failed-many.png     1 ok + 5 failed -> every failed key listed under the tally, the dismiss control apart from the list
 *
 * There is no hover frame: the list is complete in the DOM, so there is no
 * tooltip to hover (a native `title` would not render in a screenshot anyway --
 * it is browser chrome, not page content).
 *
 * Asserts as well as shoots. Pass `--expect-stale` to invert the assertion and
 * capture the BEFORE frame from a base-branch dist (`--dist <dir>`), where the
 * same refusal read "1 failed" in the danger tone.
 *
 * Usage: node scripts/capture-memory-consolidate-skipped.mjs [outDir] [prefix] [--dist <dir>] [--expect-stale]
 */
import { chromium } from 'playwright'
import { mkdirSync } from 'node:fs'
import { serveDist } from './lib/serve-dist.mjs'
import { logPageProblems, stubDashboardApi, json } from './lib/stub-dashboard-api.mjs'

const positional = process.argv.slice(2).filter((a, i, all) => !a.startsWith('--') && all[i - 1] !== '--dist')
const OUT = positional[0] || '../temp-screenshots/memory-consolidate-skipped'
const PREFIX = positional[1] || 'after'
const distIdx = process.argv.indexOf('--dist')
const DIST = distIdx > -1 ? process.argv[distIdx + 1] : undefined
const EXPECT_STALE = process.argv.includes('--expect-stale')

mkdirSync(OUT, { recursive: true })

// The transcript stems `list_sessions` hands out. The Slack thread is named by
// its filename stem, exactly as the tab posts it; its `!incognito` flag lives in
// the session map, so no row-side `memory_mode` could have filtered it out.
const OK_KEYS = ['dashboard:chat-release-notes', 'dashboard:chat-perf-triage']
const INCOGNITO_STEM = 'slack_1785861252.833429'
const FAILING_KEY = 'dashboard:chat-flaky-span'
// Five more whose requests fail, for the many-failures frame: two dashboard
// chats, a Telegram thread stem, a Discord thread stem and a cron session --
// the key shapes the list really carries.
const FAILING_KEYS = [
  FAILING_KEY, 'dashboard:chat-mesh-dns', 'telegram_7781120043', 'discord_1418921033711624344', 'cron:nightly-digest',
]

// The route's own refusal body (api_memory_consolidate).
const REFUSAL = {
  error: 'Consolidation is not allowed for an incognito session: it leaves no durable memory.',
  code: 'restricted_target_session',
}

const session = (key) => ({ key, title: key, messages: 12, agent: 'kirocrew', memory_mode: 'persistent' })

async function main() {
  const { srv, base } = await serveDist(DIST)
  const { LD_LIBRARY_PATH: _mise, ...browserEnv } = process.env
  const browser = await chromium.launch({ env: browserEnv })
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 }, deviceScaleFactor: 2 })
  const page = await context.newPage()

  // Mutable fixture state the harness flips between frames.
  let sessions = [...OK_KEYS, INCOGNITO_STEM].map(session)
  const posted = []
  await stubDashboardApi(page, {
    folders: [], slots: [],
    extra: async (path, route) => {
      if (path === '/api/sessions') { await json(route, { sessions }); return true }
      if (path === '/api/memory/consolidate') {
        const body = route.request().postDataJSON()
        posted.push(body.key)
        if (body.key === INCOGNITO_STEM) { await json(route, REFUSAL, 403); return true }
        if (FAILING_KEYS.includes(body.key)) { await json(route, { error: 'consolidation retry backoff' }, 500); return true }
        await json(route, { ok: true, key: body.key }); return true
      }
      if (path === '/api/memory/settings') { await json(route, { history_idle_hours: 3, history_max_days: 90, migrated: true }); return true }
      if (path === '/api/memory/preferences' || path === '/api/memory/projects' || path === '/api/memory/history') {
        await json(route, { content: '' }); return true
      }
      if (path === '/api/lessons') { await json(route, { lessons: [] }); return true }
      if (path === '/api/memory/stores') {
        await json(route, { stores: [{ name: 'default', is_default: true, lineage: 'v1', exists: true }], active: 'default' }); return true
      }
      if (path === '/api/memory/retired') { await json(route, { retired: [] }); return true }
      if (path === '/api/memory/backups') { await json(route, { backups: [] }); return true }
      if (path === '/api/memory/carve') { await json(route, { counts: {} }); return true }
      if (path === '/api/memory/stats') { await json(route, { entries: 0, size_bytes: 0, provider: 'local' }); return true }
      if (path === '/api/memory/embedding-status') {
        await json(route, {
          enabled: true, provider: 'local', model_available: true, server_healthy: true,
          setup_step: 'ready', model_id: 'qwen3-embedding:0.6b', model_dim: 1024,
          model_source: 'default', model_path: '', reembed: { step: 'idle', done: 0, total: 0, error: '' },
        })
        return true
      }
      if (path.startsWith('/api/memory/')) { await json(route, {}); return true }
      return false
    },
  })
  logPageProblems(page)

  await page.goto(`${base}/settings/overview?view=memory`, { waitUntil: 'domcontentloaded' })
  const button = page.getByRole('button', { name: 'Summarize now' })
  await button.waitFor({ state: 'visible', timeout: 20_000 })
  // The card that holds the button: the tally renders beside it, and the card
  // is the frame a reader needs to place it.
  const card = button.locator('xpath=ancestor::*[contains(@class,"card-glow")][1]')

  async function press(name, expectText, { expectAlert, forbidText, expectNamed = [] }) {
    posted.length = 0
    await button.click()
    const tally = page.getByText(expectText, { exact: false })
    await tally.first().waitFor({ state: 'visible', timeout: 15_000 })
    const shown = (await tally.first().textContent()) ?? ''
    const alerts = page.getByRole('alert')
    const alertCount = await alerts.count()
    const alertText = alertCount ? ((await alerts.first().textContent()) ?? '') : ''
    console.log(`${name}: posted ${JSON.stringify(posted)}; shows ${JSON.stringify(shown)}; alerts=${alertCount} ${JSON.stringify(alertText)}`)
    if (forbidText && shown.includes(forbidText)) throw new Error(`${name}: tally still reads ${JSON.stringify(forbidText)}`)
    if (expectAlert && alertCount === 0) throw new Error(`${name}: the failed tally did not render through ErrorNotice (no role="alert")`)
    if (!expectAlert && alertCount !== 0) throw new Error(`${name}: an error surface rendered for a refusal the user asked for`)
    if (expectNamed.length > 0) {
      // Every failed key in the DOM, under its label, and none behind a tooltip.
      const keys = alerts.first().getByTestId('consolidate-failed-keys')
      const listed = (await keys.textContent()) ?? ''
      for (const key of expectNamed) if (!listed.includes(key)) throw new Error(`${name}: the failed list does not name ${key}: ${JSON.stringify(listed)}`)
      if (!alertText.includes('Failed:')) throw new Error(`${name}: the key list carries no label`)
      if (await alerts.first().locator('[title]').count()) throw new Error(`${name}: part of the list rides a title tooltip`)
      // The dismiss control is the banner's own, beside the text block -- not
      // the element right after the last key.
      const adjacent = await keys.evaluate(el => el.nextElementSibling?.tagName === 'BUTTON' || el.parentElement?.querySelector(':scope > button') !== null)
      if (adjacent) throw new Error(`${name}: the dismiss control sits right against the key list`)
    }
    await page.mouse.move(40, 40)
    await card.screenshot({ path: `${OUT}/${PREFIX}-${name}.png` })
    // The success tally clears itself after 4 s; the failure notice stays until
    // dismissed. Wait it out either way so the next press starts clean.
    await page.waitForTimeout(4_500)
    if (expectAlert) {
      if ((await alerts.count()) === 0) throw new Error(`${name}: the failure notice vanished on its own`)
      await alerts.first().getByRole('button', { name: 'Dismiss' }).click()
      if (await alerts.count()) throw new Error(`${name}: the failure notice did not dismiss`)
    }
  }

  if (EXPECT_STALE) {
    // The base branch counted the refusal as a failure, in the danger tone, on
    // every press -- the regression this harness exists to disprove.
    await press('1-skipped', '(1 failed)', { expectAlert: false, forbidText: 'skipped' })
  } else {
    await press('1-skipped', '(1 skipped as temporary or incognito)', { expectAlert: false, forbidText: 'failed' })
    sessions = [OK_KEYS[0], INCOGNITO_STEM, FAILING_KEY].map(session)
    await press('2-failed-skipped', '(1 failed, 1 skipped as temporary or incognito)', { expectAlert: true, expectNamed: [FAILING_KEY] })
    sessions = [OK_KEYS[0], ...FAILING_KEYS].map(session)
    await press('3-failed-many', '1/6 sessions (5 failed)', { expectAlert: true, expectNamed: FAILING_KEYS })
  }

  await browser.close()
  srv.close()
  console.log(`wrote ${OUT}/${PREFIX}-*.png`)
}

main().catch(err => { console.error(err); process.exit(1) })
