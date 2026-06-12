# CIKA Circle — Interactive Engine & Economics

A single-page, dependency-free interactive model of the CIKA bid-allocation rotating
circle. **Every number is computed live by the actual simulation engine running in the
browser** — there is no backend and no pre-baked data.

It is built to pitch the mechanism internally: how the engine works, how it stays solvent,
the per-slot pricing, and the economics for **members, CIKA, and a bank partner**.

## What's inside

| File | Role |
|---|---|
| `engine.js` | Faithful JS port of the Python `bidrosca_simulator.py` (Version B, yield-led). Single-path + Monte Carlo. Money-conserving, verified against the Python within MC noise. |
| `app.js` | UI: controls, KPIs, canvas charts (no chart library), money split, slot table, option comparison. |
| `index.html` / `styles.css` | The page. |

## Sections

1. **How it works** — animated walkthrough of one month (contributions → auction → allocation → reserve → distribution).
2. **Live simulator** — sliders for N, contribution, cycles, treasury yield, member share, bank line, payment discipline; runs a real in-browser Monte Carlo and shows failure rate, member-outcome histogram, and the reserve / bank-line path.
3. **Who earns what** — per-circle and portfolio-scaled split across Members / CIKA / Bank.
4. **Slot pricing** — the auto-solved per-slot asking-price schedule (guarantees "quitting never pays" at every slot).
5. **CIKA & bank options** — the three separable bank roles (treasury manager / liquidity backstop / distribution) with a live configuration comparison.

## Run locally

```bash
cd cika-rosca-web
python3 -m http.server 8000
# open http://localhost:8000
```

(Must be served over HTTP, not opened as a `file://` URL, because it uses ES modules.)

## Deploy on GitHub Pages

1. Push this folder to a repository.
2. In **Settings → Pages**, set the source to the branch and (if the folder is at the repo
   root) `/root`, or move these files to `/docs` and select `/docs`.
3. The site is fully static — no build step.

## Compliance note

Terminology on the page is engineering-oriented and illustrative. Consumer-facing language
(avoiding "investment", "savings", "return", "interest", "deposit") is handled separately and
is **not** the wording in this internal model.

## Methodology / honesty notes

- **Money is conserved.** Member net positions reconcile exactly to the fintech take plus any
  outstanding bank advance (verified in the Python engine to 0.0000).
- **Central scenario** = UEMOA-realistic: per-member on-time payment drawn around the discipline
  slider, plus two lean-season dips. Tail behaviour is shown via p95/p99, not hidden.
- **"Failure rate"** = probability the reserve *and* the bank line are both exhausted in a path.
- Defaults: N=12, 2 cycles, 8% gross treasury yield, 85% member share, 300k bank line.
