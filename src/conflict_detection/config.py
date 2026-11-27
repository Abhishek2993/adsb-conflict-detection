"""Run configuration: separation minima, look-ahead horizon and filter tuning.

Every number that changes the behaviour of the pipeline lives here, so that a
run can be reproduced from the ``config.json`` written next to its results.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

from .units import FT, NM

PredictorName = Literal["cv", "kalman"]
DetectionRule = Literal["cpa", "interval"]


@dataclass(frozen=True)
class SeparationMinima:
    """Separation standard. A loss of separation needs BOTH to be violated."""

    horizontal_nm: float = 5.0
    vertical_ft: float = 1000.0

    @property
    def horizontal_m(self) -> float:
        """Horizontal minimum in metres."""
        return self.horizontal_nm * NM

    @property
    def vertical_m(self) -> float:
        """Vertical minimum in metres."""
        return self.vertical_ft * FT


@dataclass(frozen=True)
class CleaningConfig:
    """Parameters of the outlier filter and the common time grid."""

    resample_s: float = 5.0
    max_gap_s: float = 60.0
    min_altitude_ft: float = 1000.0
    max_altitude_ft: float = 50000.0
    max_groundspeed_kt: float = 700.0
    max_vertical_rate_fpm: float = 8000.0
    altitude_spike_ft: float = 1500.0
    min_points: int = 12


@dataclass(frozen=True)
class KalmanConfig:
    """Constant-velocity Kalman filter tuning (white-noise acceleration model)."""

    accel_noise_h: float = 0.5  # horizontal acceleration spectral density [m/s^2]
    accel_noise_v: float = 0.3  # vertical acceleration spectral density [m/s^2]
    pos_noise_h_m: float = 50.0  # ADS-B horizontal position noise, 1 sigma
    pos_noise_v_m: float = 15.0  # barometric altitude noise (25 ft quantisation), 1 sigma
    vel_noise_h_ms: float = 2.0  # reported ground velocity noise, 1 sigma
    vel_noise_v_ms: float = 1.5  # reported vertical rate noise, 1 sigma
    use_velocity_measurements: bool = True


@dataclass(frozen=True)
class ResolutionConfig:
    """Manoeuvre ladder used by the resolver."""

    heading_changes_deg: tuple[float, ...] = (15.0, 30.0)
    altitude_changes_ft: tuple[float, ...] = (1000.0,)
    speed_changes_frac: tuple[float, ...] = (0.10,)
    bank_angle_deg: float = 25.0  # turn rate = g tan(bank) / V, as in BlueSky
    accel_kt_s: float = 1.0
    climb_rate_fpm: float = 1500.0
    check_step_s: float = 2.0


@dataclass(frozen=True)
class Config:
    """Top-level configuration of a pipeline run."""

    separation: SeparationMinima = field(default_factory=SeparationMinima)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    resolution: ResolutionConfig = field(default_factory=ResolutionConfig)
    horizon_s: float = 300.0
    prediction_step_s: float = 10.0
    eval_step_s: float = 10.0
    predictor: PredictorName = "kalman"
    rule: DetectionRule = "cpa"
    stale_after_s: float = 30.0
    event_merge_gap_s: float = 30.0
    min_track_before_los_s: float = 60.0

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as a JSON-serialisable dictionary."""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Write the configuration to a JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Build a configuration from a (possibly partial) dictionary."""
        nested = {
            "separation": SeparationMinima,
            "cleaning": CleaningConfig,
            "kalman": KalmanConfig,
            "resolution": ResolutionConfig,
        }
        kwargs: dict[str, Any] = {}
        names = {f.name for f in fields(cls)}
        for key, value in data.items():
            if key not in names:
                raise KeyError(f"unknown configuration key: {key}")
            if key in nested and isinstance(value, dict):
                sub = dict(value)
                for k, v in sub.items():
                    if isinstance(v, list):
                        sub[k] = tuple(v)
                kwargs[key] = nested[key](**sub)
            else:
                kwargs[key] = value
        return cls(**kwargs)
