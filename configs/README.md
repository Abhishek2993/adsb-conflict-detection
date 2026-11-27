# Configuration variants

Pass any of these with `--config`, e.g.
`python -m conflict_detection run --scenario data/sample --out results/x --config configs/robust_resolution.json`.
Keys not listed keep their defaults from `src/conflict_detection/config.py`
(which implement the requirements: 5 NM / 1,000 ft, 300 s horizon, Kalman predictor, CPA rule).

| File | Change | Why |
|---|---|---|
| `robust_resolution.json` | accept a manoeuvre only if the predicted margin is at least 1.1 x the minima | absorbs mismatch between the prediction model and the simulator (see README, BlueSky results) |
| `altimetry_tolerance_200ft.json` | vertical threshold 800 ft instead of 1,000 ft | sensitivity study: treats aircraft within 200 ft of adjacent flight levels as level, i.e. ignores LoS that exist only because of 25 ft altitude quantisation and altimetry noise |
| `level_snap_200fpm.json` | vertical rates below 200 ft/min are treated as level flight | ablation: removes vertical-rate noise on level aircraft |
