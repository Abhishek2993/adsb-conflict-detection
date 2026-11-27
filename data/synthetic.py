"""Generate the synthetic encounter scenarios under data/synthetic/.

Usage:
    python data/synthetic.py            # writes data/synthetic/{noisy,clean}/

Two versions of the same nine encounters are written:

* ``clean``: perfect ADS-B (no noise), for checking the maths end to end;
* ``noisy``: position noise (40 m, 1 sigma), noisy velocities and 25 ft
  altitude quantisation, for evaluating the predictors.

Each directory has ``trajectories.parquet``, ``metadata.json`` and
``truth.csv`` (designed and measured CPA values, actual LoS interval).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from conflict_detection.ingest import Scenario, save_scenario
from conflict_detection.synthetic import NoiseModel, generate_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "synthetic")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    for label, noise in (("clean", None), ("noisy", NoiseModel())):
        traj, truth = generate_scenario(noise=noise, seed=args.seed)
        meta = {
            "name": f"synthetic_{label}",
            "source": "conflict_detection.synthetic.generate_scenario",
            "seed": args.seed,
            "noise": None if noise is None else noise.__dict__,
            "altitude_reference": "barometric",
        }
        out = save_scenario(
            Scenario(name=meta["name"], trajectories=traj, metadata=meta, truth=truth),
            args.out / label,
        )
        print(f"wrote {out} ({traj.icao24.nunique()} aircraft, {len(traj)} reports)")


if __name__ == "__main__":
    main()
