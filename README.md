# GTA VI launch-day Twitch forecast

Entry for `gta6-twitch-ccu-forecast-01`: a program that reads one case
(`/input/<id>.in`) and writes five quantiles (5/25/50/75/95%) of Twitch
concurrent viewers for each of hours 2–24 of the release day
(`/output/<id>.out`). Standard library only, offline, deterministic.

## Method

An analog forecaster over the past launches shipped in each case's
`comparables`:

1. For every comparable, take its release-day curve as log-ratios to its own
   hour-1 count. Keep only launches whose hour 1 was already live (hour-1
   viewers ≥ 5% of the day peak) — the rolling midnight-local launches, like
   GTA VI. Simultaneous global unlocks are excluded: their hour 1 predates
   the unlock, so hour-1-anchored ratios are meaningless.
2. Per hour, take scale-weighted quantiles of the ratios: analogs whose
   hour-1 count is close to this case's hour-1 print count more
   (weight `exp(-|ln h1 - ln p1|)`). Anchor on the observed hour-1 print.
3. Add a shrunk heat adjustment: a launch's heat is `ln(hour 1) - ln(median
   pre-release hourly viewers)`; a 0.5-shrunk per-hour least-squares slope on
   heat shifts the forecast, with the target's heat winsorised to the
   analogs' 10th–90th percentile range and the shift capped at ±0.5 log
   units.
4. Widen the 5%/95% quantiles to at least the unweighted Harrell-Davis
   quantiles, so the 90% band stays honest when weighting concentrates.

If fewer than 5 analogs qualify, or anything unexpected happens, the program
falls back to the house persistence baseline (flat median, spread widening
as `0.15 * sqrt(h - 1)`), and always writes a valid forecast file.

## Validation

Leave-one-out over the 29 rolling-midnight comparables in the kit
(each scored against its own hours 2–24, forecasting from the other
comparables only):

| forecast | pinball loss | typical miss (log) | 90% coverage |
|---|---|---|---|
| house persistence baseline | 0.360 | 0.537 | 0.757 |
| this program | 0.264 | 0.393 | 0.907 |

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

where `input/` holds the kit's `p00.in`–`p03.in` practice cases.
