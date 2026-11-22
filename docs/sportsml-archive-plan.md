# Archiving sportsml and moving its traffic to Run It Back

Runbook for the day `mefortunato/sportsml` is retired. Nothing here runs on its own: work through
the phases in order and tick each box as you go. Where a step needs a value that isn't in either
repo (the public runitback URL), it says `RUNITBACK_URL`. Fill that in first.

## What we're working with

| | sportsml (old) | runitback (new) |
|---|---|---|
| Hosting | GitHub Pages, static SvelteKit build (`web/`, `adapter-static`, `404.html` fallback for deep links) | Cloud Run (`deploy-cloud-run.yml`), React app served by `runitback.server` |
| URL | `https://mefortunato.github.io/sportsml/...` (`BASE_PATH` from `actions/configure-pages`, no CNAME) | `RUNITBACK_URL` (the Cloud Run service URL or its custom domain) |
| Data refresh | `site.yml`, about 10 scheduled runs a week plus pushes to `web/**` | `refresh-predictions.yml` → predictions bucket |
| Training | `train.yml` (manual) → `webdata` release | `train-predictions.yml` |

GitHub Pages can't send a server-side 301, so the redirect has to happen in the page itself. That
works out, because we want a notice and a countdown anyway rather than a silent redirect.

### Redirect target

The redirect is global: **every sportsml page, whatever its path or query string, sends readers to
runitback's home page** (`RUNITBACK_URL/`). There's no per-page mapping, so nothing depends on the
two sites' routes or game ids matching. Readers pick their sport from runitback's home page.

## Phase 0: go / no-go checks (runitback)

Don't start the cutover until every box is ticked.

- [ ] **Every sport sportsml shows is published on runitback.** sportsml exports `nfl, cfb, nba,
      cbb, wnba, wcbb, nhl, mlb`. At the time of writing, runitback's README lists `nhl` as not yet
      ported to `runitback.wager`, and says site pages for the newly ported sports "come later". Check
      `RUNITBACK_URL/api/.../sports` (whatever `api.sports()` calls) lists every sport. If one is still
      missing, decide whether to wait for it or cut over without it.
- [ ] **runitback's home page links every sport**, since that's where every redirected reader
      lands.
- [ ] **runitback's backtests reproduce sportsml's production Brier** (plan package S4, the
      `experiments/<sport>/README.md` write-ups). This is the "the new site is at least as good"
      check.
- [ ] **runitback's data refresh is keeping up:** `refresh-predictions.yml` has run green for at
      least a week, and upcoming games on runitback match sportsml's.
- [ ] **runitback is public:** the Cloud Run service allows unauthenticated access, and
      `RUNITBACK_URL` loads in a private browser window.
- [ ] **Decide the countdown length.** This plan uses **10 seconds**.

## Phase 1: the moved notice (sportsml PR)

One PR on `mefortunato/sportsml`, titled for example "Site: notice that sportsml has moved to Run
It Back". Merging it to `main` redeploys the site through `site.yml`, since it changes `web/**`.

### Behaviour

On every page of the GitHub Pages build:

1. A full-width notice opens on top of the page (a modal dialog with the page dimmed behind it):

   > **SportsML has moved to Run It Back**
   >
   > The predictions, rankings, backtests and "Beat the model" now live at
   > **runitback** (link), with the same pages and more.
   >
   > Taking you to Run It Back in **10**…
   >
   > [ Go now ]   [ Stay on the archived SportsML site ]
   >
   > _This site is archived and no longer updated. Its predictions stopped on <freeze date>.
   > SportsML will eventually be removed; please update your bookmarks._

2. The countdown ticks down once a second. At 0 the page goes to runitback's home page with
   `location.replace` (so Back doesn't bounce the reader into the redirect again).
3. **Go now** redirects immediately.
4. **Stay on the archived SportsML site** stops the timer and closes the dialog. The reader has to
   click it before the countdown ends. Pressing Escape or clicking outside the dialog does **not**
   count as staying: the dialog can't be dismissed any other way.
5. Once someone has chosen to stay, the countdown doesn't come back on later pages in that tab
   (`sessionStorage`, key `sportsml-stay`). A slim banner stays pinned at the top of every page
   instead: "SportsML is archived and will eventually be removed. **Go to Run It Back →**" (the
   link goes to runitback's home page). A new tab or visit shows the countdown again.
   That's deliberate: staying is a one-visit choice, not a permanent opt-out.
6. Storage access is wrapped in `try/catch`. If `sessionStorage` is blocked, the countdown just
   shows on every page load, which is still correct.

Not shown in the hash-routed artifact build (`npm run build:artifact`, `VITE_SPORTSML_TARGET ===
'artifact'`): it runs in a frame where a redirect would navigate the host.

### Accessibility

- The dialog is a native `<dialog>` opened with `showModal()`, labelled by its heading, with focus
  on **Stay** to start (the button a keyboard or screen-reader user needs before time runs out).
- The countdown number is `aria-hidden`. An `aria-live="polite"` region announces only "Redirecting
  in 10 seconds" when the dialog opens and "5 seconds" partway through, not every tick.
- Both buttons are at least 44 px tall; the dialog is readable at 320 px width; colors use the
  site's existing theme tokens, so light and dark both work.

### Files

- `web/src/lib/moved.ts`: the constants, and the stay flag's read/write wrapped in `try/catch`
  (unit-tested in `web/src/lib/moved.test.ts`, including a storage that throws):

  ```ts
  // the home page: every sportsml page redirects here, whatever its path
  export const RUNITBACK_HOME = new URL('/', import.meta.env.VITE_RUNITBACK_URL).toString();
  export const COUNTDOWN_SECONDS = 10;
  ```

  The build fails if `VITE_RUNITBACK_URL` is unset, rather than shipping a broken link.
- `web/src/lib/components/MovedNotice.svelte`: the dialog, timer and stay banner, all pointing at
  `RUNITBACK_HOME`. The timer starts in `$effect` (client only: the prerendered HTML ships the
  dialog closed, and a reader without JavaScript gets the static banner with the link).
- `web/src/routes/+layout.svelte`: render `<MovedNotice />` above the header when not the artifact
  build.
- `web/src/app.html`: add `<meta name="robots" content="noindex">` so search engines drop the old
  pages.
- `.github/workflows/site.yml`: pass `VITE_RUNITBACK_URL` to the Build step, as a repository
  variable (`vars.RUNITBACK_URL`) rather than a hard-coded host.
- `README.md`: a line at the top: "SportsML has moved to [Run It Back](RUNITBACK_URL). This
  repository is archived and will eventually be removed."

### Checks before merging

- [ ] `cd web && npm test && npm run check && npm run build`, with the summary lines pasted in the PR.
- [ ] Preview the build (`npm run preview` with `BASE_PATH=/sportsml`) and test in a browser, on
      desktop and at phone width:
  - [ ] the countdown lands on runitback's home page at 0, from the home page and from a deep link
        (e.g. a game page, with a query string);
  - [ ] **Stay** stops it, the slim banner appears, and moving to another page doesn't restart it;
  - [ ] a new tab shows the countdown again;
  - [ ] Escape and clicking outside don't dismiss it;
  - [ ] keyboard only: Tab reaches both buttons, Enter on Stay works;
  - [ ] light and dark themes.

## Phase 2: cutover day

1. [ ] **Freeze the data.** Run `site.yml` once by hand (`workflow_dispatch`) for a last full
       export, and write the date down as the freeze date (it goes in the notice).
2. [ ] Set the repository variable `RUNITBACK_URL` on sportsml (Settings → Secrets and variables →
       Actions → Variables), and set the freeze date in the notice text.
3. [ ] **Merge the Phase 1 PR.** `site.yml` redeploys on the push. As it's a `web/`-only change, it
       reuses the deployed data (the frozen export) instead of fetching new data.
4. [ ] Check the live site: `https://mefortunato.github.io/sportsml/` and a deep link such as
       `.../sportsml/nfl/rankings` (served through the 404.html fallback) both show the notice and
       land on `RUNITBACK_URL/`.
5. [ ] **Stop sportsml's schedules:** delete the `schedule:` block from `site.yml` and disable the
       Site and Train workflows (Actions → workflow → "Disable workflow"). This stops spending The
       Odds API quota (`ODDS_API_KEY`) on a site nobody's meant to use. If runitback needs that key,
       it now has the whole free tier.
6. [ ] Remove sportsml's secrets runitback doesn't share (`ODDS_API_KEY`, `CBBD_API_KEY` on
       sportsml only), once step 5 is done.
7. [ ] Point every link at runitback: the GitHub repo "Website" field, profile READMEs, any posts or
       docs that link to `mefortunato.github.io/sportsml`.
8. [ ] **Archive the repository** (Settings → General → Danger Zone → Archive). GitHub Pages keeps
       serving the last deployment from an archived repo, but no workflow can run and nothing can be
       pushed. Do this last: unarchive first if the notice needs fixing.

Rollback: if runitback breaks after the cutover, unarchive sportsml, revert the Phase 1 merge on
`main` (`site.yml` redeploys the frozen site without the notice), and re-enable the workflows.

## Phase 3: removal (later)

The notice tells readers sportsml will eventually be removed. When traffic to the old URL has died
down (watch for referrals from `mefortunato.github.io` on runitback, or give it a fixed period,
e.g. a season):

1. [ ] Unarchive, replace the Pages deployment with a single redirect page (no app, no data, no
       Stay button): a short "SportsML has moved" note, `<meta http-equiv="refresh">` and a JS
       redirect to runitback's home page, also used as `404.html` so every old URL is covered.
       Re-archive.
2. [ ] Later still: unpublish GitHub Pages for the repo (Settings → Pages), and optionally delete
       the `webdata` and `backtest-cache` releases to free their storage. The repo stays archived
       as source history, or gets deleted outright if that's the decision then.

## Optional: welcome on runitback

A small runitback change, not needed for the cutover: the redirect appends `?from=sportsml`, and
runitback shows a one-time "Welcome from SportsML: same pages, new home" note on its home page and strips the
parameter with `history.replaceState`. It's a frontend-only change (`web/`), so it doesn't touch any
production settings, but it does deploy on merge to `main` through `deploy-cloud-run.yml`.
