# Learning notes: how this project works and why it is built this way

These notes explain every part of the pipeline in plain language, with the maths
written out, so that you can explain and defend each decision. The measured
results are in the [README](../README.md#5-evaluation-results); this document
explains what they mean.

---

## 1. The problem in one paragraph

Two aircraft have *lost separation* when they are closer than **5 NM
horizontally and less than 1,000 ft vertically at the same time**. Either
condition alone is fine: two aircraft 1 NM apart but 2,000 ft apart vertically
are separated, and so are two aircraft at the same level 6 NM apart. A
*conflict* is a **predicted** loss of separation. A short-term conflict detector
looks a few minutes ahead (here 5 minutes) and warns early enough for a
controller or pilot to act. This project builds such a detector from ADS-B
data, measures how well it works against what really happened, proposes simple
manoeuvres to fix each conflict, and checks those manoeuvres in an independent
simulator (BlueSky).

### Vocabulary

| Term | Meaning |
|---|---|
| ADS-B | Automatic Dependent Surveillance–Broadcast: aircraft broadcast their GNSS position, barometric altitude, velocity and identity about twice per second. OpenSky collects these. |
| icao24 | 24-bit transponder address, the unique aircraft id in ADS-B. |
| Barometric altitude | Altitude from the static-pressure altimeter, referenced to the standard pressure 1013.25 hPa. All aircraft above the transition altitude fly *pressure levels* (flight levels), so separation is defined in barometric altitude. |
| Geometric altitude | GNSS height above the WGS84 ellipsoid. It differs from barometric altitude by hundreds of feet depending on the weather, so mixing the two is a hazard (see the README hazard table). |
| RVSM | Reduced Vertical Separation Minimum: between FL290 and FL410, flight levels are only 1,000 ft apart, so **aircraft on adjacent levels sit exactly at the vertical minimum**. This matters a lot for the real-data results. |
| CPA | Closest Point of Approach: the moment two straight-line trajectories are closest. tCPA = time until then, dCPA = horizontal distance then. |
| Look-ahead horizon | How far ahead the detector predicts (default 300 s). |
| Warning time | How long before the actual loss of separation the first alert was raised. |
| CAS / TAS / GS | Calibrated, true and ground airspeed. ADS-B reports ground speed; BlueSky takes CAS. Without wind data we assume TAS = GS and convert TAS to CAS with the standard atmosphere (`units.tas_to_cas`). |

---

## 2. The pipeline at a glance

```
ingest -> clean -> predict -> CPA (+ pre-filter) -> detect -> evaluate -> resolve -> BlueSky
```

| Stage | Module | One-sentence job |
|---|---|---|
| Ingest | `ingest.py` | Bring any source (sample parquet, traffic, OpenSky, synthetic) into one schema. |
| Clean | `clean.py` | Drop impossible reports, convert to a local Cartesian frame, build a common 5 s grid. |
| Predict | `predict.py` | Estimate each aircraft's current position and velocity, causally, and extrapolate in a straight line. |
| CPA | `cpa.py` | For every candidate pair, compute when and how close they will pass. |
| Detect | `detect.py` | Flag pairs predicted to violate both minima inside the horizon; merge alerts into events. |
| Evaluate | `detect.py` | Compare alerts with actual losses of separation: TP, missed, false alarms, warning time. |
| Resolve | `resolve.py` | Find the smallest speed/heading/level change that removes the predicted conflict. |
| Verify | `bluesky_bridge.py` | Re-fly each encounter in BlueSky with and without the manoeuvre. |

---

## 3. Coordinates: why a local East-North frame, and why altitude is kept separate

Latitude and longitude are angles, and a degree of longitude shrinks with
latitude (at 47° N it is 76 km, not 111 km). Distances and velocities are much
easier in metres, so every position is projected into a **local East-North-Up
(ENU) frame** centred on the scenario (`geo.LocalFrame`): convert lat/lon to
Earth-centred (ECEF) coordinates on the WGS84 ellipsoid, subtract the centre,
and rotate into East/North/Up axes.

Only **East and North** are used. The vertical coordinate is the reported
barometric altitude, *not* the ENU Up axis. Reason: the Earth curves away from
the tangent plane. A point on the ground 150 km from the centre is about
`d^2 / (2R) = 150,000^2 / (2 x 6,371,000) = 1,766 m` *below* the plane. Using
ENU Up would invent up to 1.8 km of fake vertical separation between aircraft
on the same flight level. Separation is defined in pressure altitude, so we
keep pressure altitude.

How accurate is the flat East/North approximation? The unit test
`test_planar_distance_matches_ellipsoid_chord` compares it with the exact
straight-line distance between two points about 155 NM apart and requires
agreement within 0.05 NM; the inverse projection round-trips to 5 cm at 300 km.
Both are far below the 5 NM scale that matters.

---

## 4. Cleaning, and why prediction must be causal

ADS-B data contains outliers: position jumps from decoding errors, altitude
spikes (one report in the Paris sample says 123,100 ft), duplicates, and
missing fields. `clean.clean_reports` applies simple physical gates:

* **Range:** altitude between 1,000 and 50,000 ft (removes ground traffic and nonsense).
* **Implied speed gate:** a report is rejected if reaching it from the *last
  accepted* report would need more than 700 kt plus a 250 m noise allowance.
* **Implied vertical-rate gate:** same idea with 8,000 ft/min plus 100 ft.
* **Recovery:** if 5 reports in a row are rejected, the reference itself was
  probably the outlier, so the gate re-anchors on the new data.
* **Velocity sanity:** absurd speeds are blanked (NaN), not dropped, because the position may still be good.
* **Short tracks** (< 12 reports) are dropped.

Every gate compares a report only with **earlier** reports. This is the
**causality principle**, and it runs through the whole design: an alert at
time *t* must only use data available at *t*. If we smoothed or interpolated
using later reports, the detector would look better in evaluation than it
could ever be in operation (look-ahead leakage).

That is why there are **two products**:

1. **Cleaned raw reports**, consumed in time order by the predictors.
2. **A common 5 s grid** (`clean.resample`), interpolated between reports.
   Interpolation uses the next report, so it is *not causal*. It is used only
   as the best estimate of the truth: to find actual losses of separation,
   to evaluate predictions and to draw plots.

Two details of the resampling matter:

* **No bridging of gaps.** If two consecutive reports are more than 60 s apart,
  no grid points are created between them. Inventing a straight line through
  a 5-minute gap could invent or hide a loss of separation.
* **Velocities are interpolated as East/North components**, not as track
  angles: averaging 359° and 1° gives 180°, which is exactly backwards.

---

## 5. The CPA maths (relative motion)

### 5.1 Derivation

Put aircraft A at the origin and look at B relative to A:

* relative position now: **r₀ = p_B − p_A**
* relative velocity: **v = v_B − v_A**

If both fly straight at constant velocity, the relative position after time *t* is

```
r(t) = r₀ + v t
```

The squared horizontal distance is a quadratic in t:

```
|r(t)|² = (r₀ + v t)·(r₀ + v t) = |r₀|² + 2 t (r₀·v) + t² |v|²
```

A quadratic with positive leading coefficient |v|² has its minimum where its
derivative is zero:

```
d/dt |r(t)|² = 2 (r₀·v) + 2 t |v|² = 0     =>     tCPA = −(r₀·v) / |v|²
```

Plug it back in to get the miss distance:

```
dCPA = |r₀ + v tCPA|        (equivalently  dCPA² = |r₀|² − (r₀·v)² / |v|²)
```

The vertical separation at that same instant is `|Δz₀ + Δv_z · tCPA|`.

Interpretation:

* **r₀·v < 0** means the aircraft are closing, so tCPA > 0 (CPA is in the future).
* **tCPA < 0** means they are already diverging; the closest point is behind them.
* **|v| ≈ 0** (same velocity) means the distance never changes; tCPA is
  undefined, and the code defines it as 0 so dCPA = current distance.

Everything is vectorised with numpy over all candidate pairs at once (`cpa.compute_cpa`).

### 5.2 Worked examples (these are the unit tests)

**Head-on.** A at the origin eastbound at 480 kt; B 40 NM east and 2 NM north,
westbound at 480 kt. Relative velocity is 960 kt = 16 NM/min, straight at A.
tCPA = 40 NM / 960 kt = 1/24 h = **150 s**, dCPA = the lateral offset = **2 NM**.

**Crossing.** A 20 NM west of a crossing point heading east at 400 kt; B 20 NM
south heading north at 400 kt. Both reach the crossing point after
20/400 h = **180 s**, so dCPA = **0**: a collision course.
With B starting 17 NM south instead: r₀ = (20, −17) NM, v = (−400, 400) kt,
r₀·v = −14,800, |v|² = 320,000, tCPA = 0.04625 h = **166.5 s**, and
r₀ + v·tCPA = (1.5, 1.5) NM, so dCPA = **2.12 NM**.

**Overtaking.** B is 10 NM behind A and 3 NM to the side, 200 kt faster.
tCPA = 10/200 h = **180 s**, dCPA = **3 NM**. Overtaking conflicts develop
slowly, so their warning time is short relative to their duration; this shows
up in the synthetic results.

### 5.3 The detection rule, and its blind spot

The rule in the requirements flags a conflict when

```
0 < tCPA ≤ horizon   and   dCPA < 5 NM   and   |Δz at tCPA| < 1,000 ft
```

It checks the vertical separation **only at the horizontal CPA instant**. For
level aircraft that is exact. For a climbing aircraft it is not: the vertical
closest point can happen at a different time from the horizontal one. Test
`test_cpa_rule_vs_interval_rule_for_level_crossing` builds such a case: a 2 NM
miss, horizontally inside 5 NM from 132.8 s to 167.2 s, while B climbs through
A's level and is within 1,000 ft between 84 s and 144 s. Both minima are
violated together between 132.8 s and 144 s, but at the horizontal CPA (150 s)
B is already 1,200 ft above, so the CPA rule stays silent.

The alternative **interval rule** (`--rule interval`, `cpa.los_interval`)
solves the problem exactly:

* horizontal intrusion: `|r₀ + v t|² < D²` is a quadratic inequality, true between its two roots;
* vertical intrusion: `|Δz + Δv_z t| < H` is a linear inequality, true on an interval;
* a loss of separation needs both, so intersect the two intervals and check
  whether the intersection overlaps `[0, horizon]`.

The README compares both rules on real and synthetic data.

### 5.4 The boundary

"Less than 5 NM" and "less than 1,000 ft" are strict. Floating point makes the
boundary treacherous: 38,000 ft − 37,000 ft converted to metres comes out as
304.8000000000011 m, and other level pairs can land just *below* 304.8 m.
Without care, aircraft on adjacent flight levels would randomly count as losing
separation. All comparisons go through one function, `cpa.in_los`, which
applies a 1 µm tolerance. The boundary tests check 4.99 NM (conflict) against
5.01 NM and exactly 5.00 NM (no conflict), and 999 ft against 1,000 and 1,001 ft.

---

## 6. Avoiding O(n²): the swept-segment KD-tree pre-filter

Checking every pair of *n* aircraft costs n(n−1)/2 CPA computations: 3,003
pairs for 78 aircraft, but 12.5 million for 5,000 (a busy continental sector
picture). Most pairs are hundreds of miles apart and can never conflict within
5 minutes. `cpa.candidate_pairs` discards them without losing any real
conflict:

1. Over the horizon T each aircraft sweeps a straight segment of length |v|T.
   Give it a midpoint m and half-length h = |v|T/2.
2. If two aircraft come within D (5 NM) of each other at some moment inside
   the horizon, those two positions lie on their segments. By the triangle
   inequality the midpoints are then at most **h_i + h_j + D** apart.
3. A KD-tree on the midpoints returns all pairs within 2·max(h) + D in roughly
   O(n log n); an exact per-pair check with h_i + h_j + D removes the rest.
4. A vertical test discards pairs whose altitude bands over the horizon
   (current altitude to altitude at T) never come within 1,000 ft.

Because the test is a necessary condition for a conflict, the filter is
**lossless**: `test_prefilter_is_lossless` compares it with brute force on
random traffic, and `test_prefilter_gives_identical_detections_on_sample`
checks the detections on the real sample are identical with and without it.
The README reports the measured pruning and a scaling benchmark.

---

## 7. Prediction and the Kalman filter

### 7.1 Two predictors

* **Baseline, constant velocity (cv):** take the latest report, move it forward
  by its age with its own reported velocity, then extrapolate in a straight line.
* **Improved, Kalman (kalman):** run a constant-velocity Kalman filter over
  all reports so far, then extrapolate the filtered state in a straight line.

Both are straight-line predictors. The Kalman filter does not predict turns;
it gives a **better estimate of the current state**, especially velocity,
which is what matters for a 5-minute extrapolation: a 1 m/s velocity error
becomes 300 m after 300 s, and a 100 ft/min vertical-rate error becomes 500 ft,
which is half the vertical minimum.

### 7.2 The filter

**State** (per axis x, y, z): position and velocity, `s = [p, v]`.

**Motion model** (constant velocity):

```
p(k+1) = p(k) + v(k) Δt
v(k+1) = v(k)                    F = [[1, Δt], [0, 1]]
```

Real aircraft accelerate a little, so the model is wrong by a random
acceleration *a* of standard deviation σ_a held over each step. Its effect is
`Δp = a Δt²/2`, `Δv = a Δt`, so the process-noise covariance is

```
Q = σ_a² [[Δt⁴/4, Δt³/2],
          [Δt³/2, Δt²  ]]
```

(the *discrete white-noise acceleration* model).

**Measurements:** ADS-B gives position and velocity, so `z = [p, v]`,
`H = I` (or `H = [1, 0]` when velocity is missing), with noise covariance R from
the expected sensor accuracy.

**Cycle** at each report:

```
predict:  s⁻ = F s            P⁻ = F P Fᵀ + Q
update:   K  = P⁻ Hᵀ (H P⁻ Hᵀ + R)⁻¹       (Kalman gain)
          s  = s⁻ + K (z − H s⁻)           (blend prediction and measurement)
          P  = (I − K H) P⁻ (I − K H)ᵀ + K R Kᵀ   (Joseph form)
```

Intuition: the gain K is a weight between 0 and 1. If the measurement is noisy
compared with the prediction (R large, P small), K is small and the filter
trusts its prediction; if the prediction is uncertain, K is large and the
filter follows the measurement. The Joseph form of the covariance update gives
the same result as the textbook `(I − KH)P⁻` but stays symmetric and positive
definite under rounding (`test_covariance_stays_symmetric_positive_definite`).

**Tuning** (`config.KalmanConfig`):

| Parameter | Value | Reason |
|---|---|---|
| σ_a horizontal | 0.5 m/s² | airliners in cruise rarely exceed this outside manoeuvres |
| σ_a vertical | 0.3 m/s² | level-offs and climb-rate changes are gentle |
| position noise, horizontal | 50 m | ADS-B NACp 8 means 95 % of fixes within 92.6 m |
| position noise, vertical | 15 m | 25 ft (7.6 m) quantisation plus altimetry noise |
| velocity noise | 2 m/s horizontal, 1.5 m/s vertical | the barometric vertical rate is the noisiest field |

**Design details:**

* **Decoupled axes.** F, Q, H and R are block-diagonal per axis, so a 6-state
  filter is mathematically identical to three 2-state filters. The code uses
  the three small filters: same answer, easier to read.
* **Restart after gaps** longer than 60 s: a stale estimate extrapolated
  through a long gap would bias the new track.
* **Forward filter only, no smoother.** A Rauch–Tung–Striebel smoother would
  give nicer tracks, but it uses future data and would break causality.
* **Staleness.** If an aircraft's latest report is more than 30 s old it is
  excluded from that evaluation time. That avoids alerts on ghosts, at the
  cost of possibly missing a conflict (see the hazard table).

### 7.3 When does the filter help?

On the **noisy synthetic** data (40 m position noise, noisy vertical rate) the
filter cuts the vertical prediction error several-fold and removes some false
alarms. On the **real Swiss sample** it barely changes anything. Real ADS-B
velocities come from GNSS and are already good, and the dominant prediction
error comes from manoeuvres (turns, level-offs) that no constant-velocity model
can foresee. The honest conclusion: filtering fixes *measurement* error, not
*intent* error.

---

## 8. Detection output and the severity score

At each evaluation time (every 10 s) the detector produces one alert row per
flagged pair: time, pair, callsigns, tCPA, dCPA, vertical separation at CPA,
current distances, the time the predicted LoS would start, the predicted CPA
positions, and a **severity score**:

```
pen_h    = 1 − dCPA / 5 NM                   (how deep the horizontal intrusion is)
pen_v    = 1 − |Δz_CPA| / 1,000 ft           (how deep the vertical intrusion is)
urgency  = 1 − 0.5 · tCPA / horizon          (1 = now, 0.5 = at the horizon)
severity = sqrt(pen_h · pen_v) · urgency     (0 … 1)
```

The geometric mean is only high if *both* penetrations are high, which mirrors
the definition of a loss of separation (both minima at once). A head-on
collision geometry that is imminent scores 1.0.

Consecutive alerts for the same pair (less than 30 s apart) are merged into a
**conflict event**, which is what a controller would see as one alert. The
event keeps the geometry of its first alert.

---

## 9. Evaluation metrics

Accuracy is useless here: in almost every (time, pair) combination nothing
happens. The project counts **events** instead.

* **Actual LoS:** found on the interpolated 5 s grid with the same `in_los`
  definition. An actual LoS is *evaluable* only if both aircraft had been
  tracked for at least 60 s before it started. If an aircraft appears in the
  data already inside 5 NM of another, no predictor could have warned.
* **True positive (TP):** an evaluable LoS with at least one alert for the
  same pair raised *strictly before* the LoS started, and at most
  horizon + one evaluation step before. An alert raised after the LoS started
  is too late and does not count.
* **Missed conflict:** an evaluable LoS without such an alert.
* **False alarm:** an alert event that is not followed, within the horizon
  after its last alert, by an actual LoS of the same pair (and does not overlap one).
* **Recall** (probability of detection) = TP / actual LoS.
* **Precision** = (alert events − false alarms) / alert events.
* **Warning time** = LoS start − first alert. Longer is better, but it is
  capped by the horizon.
* **False alarms per flight hour** normalises for traffic volume.

The fundamental trade-off: a longer horizon or larger thresholds give more
warning and fewer misses, but more false alarms, because straight-line
predictions get worse with time and because controllers routinely resolve
conflicts with clearances we cannot see. Missed conflicts are the
safety-critical error; false alarms are the usability error (too many and
people stop trusting the tool).

---

## 10. Resolution manoeuvres

For each conflict event the resolver takes the predicted states at the first
alert and tries this ladder on **either** aircraft:

| Manoeuvre | Values | Cost | Why this cost |
|---|---|---|---|
| speed | ±10 % | 1 | keeps route and level, invisible to other traffic layers, but slow-acting |
| heading | ±15° | 2 | small lateral deviation, keeps the level |
| level | ±1,000 ft | 3 | moves the aircraft into another traffic layer, needs its own check |
| heading | ±30° | 4 | largest lateral deviation |

It stops at the first cost level where something works and, among those
options, picks the largest separation margin. "Works" means, over the whole
horizon:

1. the pair never violates both minima, and
2. **the manoeuvring aircraft does not create a new conflict with any third
   aircraft** (secondary-conflict check, `test_secondary_conflict_rejects_otherwise_valid_manoeuvre`).

Manoeuvres are flown with simple but realistic kinematics, not as instant jumps:

* **Turns** at the rate given by a 25° bank: `ω = g·tan(φ)/V`. At 450 kt that
  is 1.13°/s, so a 30° turn takes about 27 s. A "standard rate" 3°/s turn is
  not realistic for airliners at cruise speed, and BlueSky also uses bank-angle-limited turns.
* **Speed changes** at about 1 kt/s.
* **Level changes** at 1,500 ft/min, stopping at the target level.
* The other aircraft keeps its straight line; we do not know its intent.

**Buffer (`resolution.required_margin`, default 1.0).** The BlueSky check
showed that resolutions which restore *exactly* 5 NM in our model can fail in
the simulator, because the simulator's aircraft fly slightly differently (speed
envelope limits, acceleration). A buffer of 1.1 requires 5.5 NM horizontally
*unless* the aircraft are vertically separated. The vertical minimum is
deliberately not buffered: a one-level change ends exactly 1,000 ft apart,
which is the standard and very precisely flown ATC solution. A combined
buffer would have made every level change impossible. That design error was
caught by a unit test (`test_buffer_does_not_block_level_changes`).

---

## 11. BlueSky verification

BlueSky is an open-source air traffic simulator (TU Delft) with its own
aircraft performance model (OpenAP). Re-flying each encounter there tests the
resolutions against dynamics we did not write ourselves.

For each event `bluesky_bridge.py` writes two scenario files: unresolved, and
with the resolution command. Each file:

* turns off BlueSky's own conflict detection and resolution (`ASAS OFF`,
  `RESO OFF`), so BlueSky does not "help";
* creates both aircraft at their predicted states at the first alert
  (`CRE acid A333 lat lon heading altitude CAS`). Climbing or descending
  aircraft get `ALT target VS` so they keep their vertical rate;
* creates a state logger (`CRELOG`) at 1 s and stops itself with `HOLD`
  after horizon + 60 s.

BlueSky runs headless (`bluesky.init(mode="sim", detached=True)`) in a
**separate Python process**, because it keeps global state and prints a lot.
The logs are parsed back (`read_log`) and the minimum separation is measured
with the same `in_los` definition.

Two outcomes are reported per event:

* **Conflict reproduced:** the unresolved run loses separation in BlueSky.
  If it does not, the predicted conflict depended on something BlueSky does
  not reproduce. On the real sample the reason is almost always the same:
  BlueSky holds a level aircraft (|vertical rate| < 100 ft/min, which we
  create without a climb command) *exactly* level, whereas our predictor
  extrapolated a noise-level vertical rate of a few tens of ft/min for 5
  minutes. Of the level pairs, 0 of 43 that started 1,000 ft or more apart
  lost separation in BlueSky, against 25 of 27 that started just under
  1,000 ft; all 12 pairs with a climbing or descending aircraft were
  reproduced. Note that this does **not** make BlueSky a false-alarm
  detector: it reproduced about half of the confirmed events (17 of 42) and
  about half of the false alarms (20 of 40), because the real-data events
  are adjacent-level pairs whose fate depends on tens of feet.
* **Resolution success:** the resolved run keeps separation.

Why the A333: ADS-B does not carry the aircraft type, and BlueSky needs one.
Among the common types in OpenAP, the A333 has the highest ceiling (41,100 ft).
An A320 created at FL370 in BlueSky sinks to its 36,286 ft ceiling, which would
change the geometry. Encounters above the ceiling are exported but marked not simulable.

---

## 12. What the real data taught

On the Swiss sample, **every** actual loss of separation is a pair on adjacent
flight levels whose measured vertical separation dips just below 1,000 ft
(925 to 987.5 ft). Under RVSM, aircraft on adjacent levels are *exactly* at the
vertical minimum, so 25 ft altitude quantisation and altimetry noise decide
whether a level pair "loses separation". Operationally these are not
incidents: altimetry tolerances are about ±200 ft. Most predicted false alarms
are the same pairs with ~50 ft/min of vertical-rate noise extrapolated over
5 minutes.

Two things follow, and both are in the README:

1. A sensitivity run with an altimetry tolerance (vertical threshold 800 ft,
   `configs/altimetry_tolerance_200ft.json`) shows how many alerts survive
   when quantisation artefacts are excluded.
2. A good synthetic test set is essential: it is the only place where the
   ground truth contains real, deep losses of separation with known geometry.

---

## 13. Design decisions (and the alternatives)

| Decision | Alternatives considered | Why this one |
|---|---|---|
| SI units internally, aviation units at the edges | work in NM/ft/kt throughout | one conversion module, tested once; no mixed-unit bugs |
| Local ENU East/North + barometric altitude | haversine on lat/lon; full 3-D ENU | fast vector maths; avoids the 1.8 km curvature artefact |
| Causal predictor on raw reports, interpolated grid only for truth | predict from the interpolated grid | no look-ahead leakage, evaluation stays honest |
| Straight-line (CV) extrapolation | turn-aware / intent-based models | the requirement; simple, explainable; failures are measurable (turning encounter) |
| Kalman CV filter, decoupled axes, forward only | α-β filter; IMM; RTS smoother | standard, explainable, causal; IMM would be the next step for turns |
| Analytic CPA | sample the predicted trajectories every 10 s | exact, vectorised, no sampling error; sampling is kept for plots and resolution |
| CPA-point rule by default, interval rule as option | interval rule by default | the requirement defines the CPA rule; the interval rule is reported as an improvement |
| Swept-segment KD-tree pre-filter | uniform grid hashing; no filter | lossless by construction, O(n log n), one scipy call |
| One `in_los` predicate with 1 µm tolerance | inline comparisons | single definition of LoS everywhere; fixes the floating-point boundary bug |
| Event-level metrics with "evaluable" LoS | per-timestep accuracy / ROC | matches how alerts are used; excludes cases no predictor could catch |
| Ordered manoeuvre ladder with costs + secondary-conflict check | optimisation (e.g. MVP, velocity obstacles) | transparent, deterministic, matches simple ATC practice |
| BlueSky in a subprocess, A333 type, own CD&R off | in-process; real types | isolation; type must cover the cruise levels; we test *our* resolution, not BlueSky's |
| Strict 5 NM / 1,000 ft by default, tolerance as a sensitivity run | build tolerance into the definition | the requirement is the requirement; the data issue is shown, not hidden |

---

## 14. Questions you may be asked

**Why is tCPA = −(r·v)/|v|²?** It minimises the quadratic |r₀ + vt|²; set the
derivative 2r₀·v + 2t|v|² to zero.

**What if both aircraft have the same velocity?** |v| = 0, the distance is
constant, tCPA is undefined; the code sets it to 0 and dCPA = current distance.

**Why not just check the distance every 10 s along the prediction?** Sampling
can step over a short intrusion (a head-on pair closes 1.3 NM in 5 s); the
analytic CPA has no sampling error and is cheaper.

**Why did the Kalman filter not help on real data?** Because ADS-B velocity is
already accurate; the remaining error is intent (turns, level-offs), which a
constant-velocity model cannot know.

**What is the biggest weakness?** No intent information: we extrapolate
straight lines while aircraft follow routes and clearances. That causes both
false alarms (a descent that will level off) and late alerts (a turn into
traffic, the `turn_into_conflict` encounter).

**How would you improve it?** Use intent (flight plan, Mode S selected
altitude: the "selected altitude" field in ADS-B would remove most level-off
false alarms), an IMM filter with a turn model, probabilistic conflict
detection (propagate the Kalman covariance into a conflict probability), wind
data for the TAS/GS conversion, and more real data with actual incidents.

**Is the KD-tree filter really lossless?** Yes: two positions inside D at the
same time imply the segment midpoints are within h_i + h_j + D (triangle
inequality); the vertical band test is likewise necessary. It is tested
against brute force.

**Why does BlueSky not reproduce many predicted conflicts on real data?**
Most of them exist only because our predictor extrapolates a small,
noise-level vertical rate over 5 minutes, while BlueSky holds the level
exactly: none of the 43 level pairs that started at least 1,000 ft apart lost
separation in BlueSky. It is not a clean false-alarm test, though: it
reproduced roughly half of both the confirmed events and the false alarms,
because on real data both are adjacent-level pairs decided by tens of feet.

**Barometric or geometric altitude?** Barometric, because that is what
flight levels and the vertical minimum are defined in. Mixing the two can
create or hide errors of hundreds of feet.

**Is this an operational tool?** No. It has no intent, no wind, no
certification, simplified manoeuvres and a small dataset. It is a
demonstrator of the method and of how to evaluate it.
