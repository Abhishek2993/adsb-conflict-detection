# Data

| Directory | Content | Source |
|---|---|---|
| `sample/` | 1 hour of en-route traffic over Switzerland, 2018-08-01 14:00–15:00 UTC: 79 aircraft, 7,513 ADS-B reports at 10 s intervals. Barometric altitude. | Cut from the `switzerland` collection bundled with the [`traffic`](https://traffic-viz.github.io/) library. The data was recorded by the [OpenSky Network](https://opensky-network.org). |
| `synthetic/clean/` | Nine designed two-aircraft encounters with perfect measurements (1 s reports). | `python data/synthetic.py` |
| `synthetic/noisy/` | Same encounters with 40 m position noise, noisy velocities and 25 ft altitude quantisation. | `python data/synthetic.py` |

Every scenario directory contains `trajectories.parquet` (schema in
`src/conflict_detection/ingest.py`) and `metadata.json`; the synthetic ones add
`truth.csv` with the designed and measured closest point of approach and the
true loss-of-separation interval of each encounter.

## Rebuilding

```bash
pip install -e ".[traffic]"
python -m conflict_detection build-sample --out data/sample      # from traffic's bundled data
python data/synthetic.py                                         # synthetic scenarios
```

## OpenSky (optional)

```bash
export OPENSKY_USERNAME=...   # never commit credentials
export OPENSKY_PASSWORD=...
python -m conflict_detection fetch-opensky --start "2024-05-02 12:00" --stop "2024-05-02 13:00" \
    --bounds 5.9 45.8 10.5 47.8 --out data/opensky/swiss_20240502
```

If you publish results that use OpenSky data, cite: M. Schäfer, M. Strohmeier, V. Lenders,
I. Martinovic and M. Wilhelm, "Bringing Up OpenSky: A Large-scale ADS-B Sensor Network for
Research", IPSN 2014.
