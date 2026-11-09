# Project instructions (paste into Project settings > Memory > Project instructions)

This project brings the NFL player-based wager model (play-by-play foundation embeddings + player
values + projected-lineup `pvc` terms) to NBA, WNBA, men's and women's college basketball, NHL, MLB
and college football in `mefortunato/runitback`.

The plan and every work package (IDs like S1, N2, H4) are in `docs/multisport/PLAN.md`. Until it is
on `main`, read it with:
`git fetch origin claude/ecstatic-lovelace-opcpo7 && git show origin/claude/ecstatic-lovelace-opcpo7:docs/multisport/PLAN.md`
Read the NFL write-up `experiments/nfl/README.md` before starting a package.

- **Base branch:** branch from `claude/focused-brahmagupta-34i6ys` (the NFL player model) until it
  is merged to `main`, then from `main`. If a package depends on another that is not merged yet,
  branch from that package's branch and say so in the PR.
- **One thread per work package, one draft PR per thread.** The PR title starts with the package ID,
  e.g. `[N2] NBA player values`. Base the PR on the branch you started from.
- **Do the package as specified in PLAN.md.** If the data or results contradict the plan, stop and
  report what you found instead of redesigning the package.
- **Before calling work done:** run `pytest` (or the relevant `tests/` subset for slow suites) and
  paste the summary line. For experiment packages, report the Brier deltas with 95% CIs against the
  sport's production baseline, and the swap test, in `experiments/<sport>/README.md`, in the NFL
  write-up's format.
- **No leakage:** every fit, PCA and projected lineup uses only data before the game it is for.
- **Refactors of NFL code** (S1, S2, S3) must leave NFL outputs identical; prove it with a test.
- **Don't push changes to `experiments/nfl/*.yaml` or `.github/workflows/*`** unless the package calls
  for it. Pushes to `claude/**` branches that touch them start multi-hour GitHub Actions runs.
- **Heavy training** (fold models) goes in a GitHub Actions matrix workflow modeled on
  `.github/workflows/foundation-experiment.yml`, not in the thread's container.
- **If something is missing** (an API key such as `CBBD_API_KEY`, a blocked host, a repository),
  say exactly what in your first message and stop. Don't mock or substitute data.
- **Ask me first** before merging, force-pushing, deleting branches, or changing production
  settings (`RUNITBACK_*` variables, deploy workflows).
- **Reference:** the original per-sport loaders are in `mefortunato/sportsml` under
  `src/sportsml/<sport>/`.
