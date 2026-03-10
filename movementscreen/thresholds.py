"""Global threshold configuration for compensation detection.

Values are loaded from the database at runtime with an in-process TTL cache,
falling back to hardcoded defaults when no DB override exists.

Caching strategy
----------------
A single ``_ThresholdCache`` instance lives in this module.  On each analysis
request the server calls ``get_thresholds(pool)`` which:

1. Returns the cached ``ThresholdConfig`` immediately if it is younger than
   ``ttl_seconds`` (default 60 s).
2. Otherwise queries the ``threshold_config`` table, applies any overrides on
   top of the dataclass defaults, stores the result, and returns it.

When an admin writes a new value the server calls ``invalidate_threshold_cache()``
so the very next analysis request picks up the change without waiting for the TTL.

AWS / multi-instance note
-------------------------
Each process keeps its own in-memory cache.  On a single instance (or a single
ECS task) this is seamless.  When running multiple instances the worst-case lag
between an admin update and all instances seeing it is one TTL period.  For a
value that changes rarely this is acceptable.  If you need instant cross-instance
consistency, replace ``_ThresholdCache`` below with a Redis-backed equivalent
(e.g. via ``aiocache`` + ElastiCache) — the public API (``get_thresholds`` /
``invalidate_threshold_cache``) stays the same.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, fields as dc_fields
from typing import Optional

import asyncpg


# ---------------------------------------------------------------------------
# Human-readable descriptions (used by the admin API)
# ---------------------------------------------------------------------------

DESCRIPTIONS: dict[str, str] = {
    "knee_valgus_mild": (
        "Knee frontal angle (°) at or below which mild valgus is flagged "
        "(e.g. 178° = 2° inward from straight)"
    ),
    "knee_valgus_moderate": (
        "Knee frontal angle (°) at or below which moderate valgus is flagged"
    ),
    "knee_valgus_severe": (
        "Knee frontal angle (°) at or below which severe valgus is flagged"
    ),
    "trunk_lean_mild": "Forward trunk lean (°) above which mild lean is flagged",
    "trunk_lean_moderate": "Forward trunk lean (°) above which moderate lean is flagged",
    "trunk_lean_severe": "Forward trunk lean (°) above which severe lean is flagged",
    "ankle_df_mild": (
        "Ankle dorsiflexion angle (°) below which mild restriction / heel rise is flagged"
    ),
    "ankle_df_moderate": (
        "Ankle dorsiflexion angle (°) below which moderate restriction / heel rise is flagged"
    ),
    "lateral_shift_mild": (
        "Lateral trunk shift (normalised image coords) above which mild shift is flagged"
    ),
    "lateral_shift_moderate": (
        "Lateral trunk shift (normalised image coords) above which moderate shift is flagged"
    ),
    "lateral_shift_severe": (
        "Lateral trunk shift (normalised image coords) above which severe shift is flagged"
    ),
    "asymmetry_mild": "Bilateral asymmetry ratio above which mild asymmetry is flagged",
    "asymmetry_moderate": "Bilateral asymmetry ratio above which moderate asymmetry is flagged",
    "asymmetry_severe": "Bilateral asymmetry ratio above which severe asymmetry is flagged",
    "lateral_flexion_mild": "Lateral trunk tilt (°) above which mild lateral flexion is flagged",
    "lateral_flexion_moderate": "Lateral trunk tilt (°) above which moderate lateral flexion is flagged",
    "lateral_flexion_severe": "Lateral trunk tilt (°) above which severe lateral flexion is flagged",
    "spine_curve_mild": (
        "Spine segmental deviation from 180° (°) above which mild curvature is flagged"
    ),
    "spine_curve_moderate": (
        "Spine segmental deviation from 180° (°) above which moderate curvature is flagged"
    ),
    "spine_curve_severe": (
        "Spine segmental deviation from 180° (°) above which severe curvature is flagged"
    ),
    "upper_trunk_mild": (
        "Upper trunk (ear→shoulder) angle from vertical (°) above which mild flexion is flagged"
    ),
    "upper_trunk_moderate": (
        "Upper trunk (ear→shoulder) angle from vertical (°) above which moderate flexion is flagged"
    ),
    "upper_trunk_severe": (
        "Upper trunk (ear→shoulder) angle from vertical (°) above which severe flexion is flagged"
    ),
    "head_forward_mild": (
        "Head forward offset (normalised) above which mild forward head posture is flagged "
        "(lateral view only)"
    ),
    "head_forward_moderate": (
        "Head forward offset (normalised) above which moderate forward head posture is flagged "
        "(lateral view only)"
    ),
    "head_forward_severe": (
        "Head forward offset (normalised) above which severe forward head posture is flagged "
        "(lateral view only)"
    ),
    # Squat-specific trunk lean
    "squat_trunk_lean_mild": (
        "Squat: trunk lean (°) above which mild excessive forward lean is flagged. "
        "Optimal squat trunk lean is 20–40°, so this should be set above 40°."
    ),
    "squat_trunk_lean_moderate": (
        "Squat: trunk lean (°) above which moderate excessive forward lean is flagged."
    ),
    "squat_trunk_lean_severe": (
        "Squat: trunk lean (°) above which severe excessive forward lean is flagged."
    ),
    # Tibial angle (lateral view DF proxy)
    "tibial_angle_restricted_mild": (
        "Tibial angle (°) below which mild dorsiflexion restriction is flagged (lateral view). "
        "Optimal at squat depth: 30–40°."
    ),
    "tibial_angle_restricted_severe": (
        "Tibial angle (°) below which severe dorsiflexion restriction is flagged (lateral view)."
    ),
    # Pelvic tilt (anterior view)
    "pelvic_tilt_mild": (
        "Pelvic tilt (°) above which mild hip-level asymmetry is flagged (anterior view)."
    ),
    "pelvic_tilt_moderate": (
        "Pelvic tilt (°) above which moderate hip-level asymmetry is flagged (anterior view)."
    ),
    "pelvic_tilt_severe": (
        "Pelvic tilt (°) above which severe hip-level asymmetry is flagged (anterior view)."
    ),
}


# ---------------------------------------------------------------------------
# ThresholdConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class ThresholdConfig:
    """All configurable compensation thresholds with their hardcoded defaults."""

    # Knee valgus (frontal angle; lower = more collapsed knee)
    # 180° = perfectly straight alignment in the frontal plane
    knee_valgus_mild: float = 177.0      # ~3° inward — early warning
    knee_valgus_moderate: float = 173.0  # ~7° inward — clinically significant
    knee_valgus_severe: float = 165.0    # ~15° inward — major collapse

    # Forward trunk lean
    trunk_lean_mild: float = 20.0
    trunk_lean_moderate: float = 35.0
    trunk_lean_severe: float = 50.0

    # Ankle dorsiflexion / heel rise
    ankle_df_mild: float = 100.0
    ankle_df_moderate: float = 90.0

    # Lateral trunk shift (normalised image coords)
    lateral_shift_mild: float = 0.02
    lateral_shift_moderate: float = 0.05
    lateral_shift_severe: float = 0.08

    # Bilateral asymmetry ratio [0-1]
    asymmetry_mild: float = 0.10
    asymmetry_moderate: float = 0.20
    asymmetry_severe: float = 0.35

    # Lateral spinal flexion (degrees)
    lateral_flexion_mild: float = 5.0
    lateral_flexion_moderate: float = 10.0
    lateral_flexion_severe: float = 15.0

    # Spine segmental curvature (deviation from 180°)
    spine_curve_mild: float = 10.0
    spine_curve_moderate: float = 15.0
    spine_curve_severe: float = 20.0

    # Upper trunk / cervicothoracic angle from vertical
    upper_trunk_mild: float = 15.0
    upper_trunk_moderate: float = 25.0
    upper_trunk_severe: float = 35.0

    # Head forward offset (normalised image coords, lateral view)
    head_forward_mild: float = 0.03
    head_forward_moderate: float = 0.05
    head_forward_severe: float = 0.07

    # Squat-specific trunk lean (optimal squat lean is 20–40°, so flag above 40°+)
    squat_trunk_lean_mild: float = 45.0
    squat_trunk_lean_moderate: float = 55.0
    squat_trunk_lean_severe: float = 65.0

    # Tibial angle restriction (lateral view): lower = more restricted DF
    tibial_angle_restricted_mild: float = 28.0
    tibial_angle_restricted_severe: float = 18.0

    # Pelvic tilt from horizontal (anterior view)
    pelvic_tilt_mild: float = 3.0
    pelvic_tilt_moderate: float = 6.0
    pelvic_tilt_severe: float = 10.0

    # ------------------------------------------------------------------
    @classmethod
    def from_db_overrides(cls, overrides: dict[str, float]) -> "ThresholdConfig":
        """Build a ThresholdConfig applying DB overrides on top of defaults."""
        cfg = cls()
        valid = {f.name for f in dc_fields(cfg)}
        for key, value in overrides.items():
            if key in valid:
                setattr(cfg, key, float(value))
        return cfg

    def as_dict(self) -> dict[str, float]:
        return {f.name: getattr(self, f.name) for f in dc_fields(self)}

    def default_for(self, key: str) -> float:
        """Return the hardcoded default for a given key."""
        return self.__class__.__dataclass_fields__[key].default


# ---------------------------------------------------------------------------
# In-process TTL cache
# ---------------------------------------------------------------------------

class _ThresholdCache:
    """Holds one ThresholdConfig and the time it was loaded.

    To swap in a Redis-backed cache: implement the same ``get`` / ``set`` /
    ``invalidate`` interface and replace the ``_cache`` singleton below.
    """

    def __init__(self, ttl_seconds: float = 60.0) -> None:
        self._config: Optional[ThresholdConfig] = None
        self._loaded_at: float = 0.0
        self._ttl = ttl_seconds

    def get(self) -> Optional[ThresholdConfig]:
        if self._config is None:
            return None
        if (time.monotonic() - self._loaded_at) > self._ttl:
            return None
        return self._config

    def set(self, config: ThresholdConfig) -> None:
        self._config = config
        self._loaded_at = time.monotonic()

    def invalidate(self) -> None:
        self._config = None


_cache = _ThresholdCache(ttl_seconds=60.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def get_thresholds(pool: asyncpg.Pool) -> ThresholdConfig:
    """Return the current ThresholdConfig, reloading from DB when the cache is stale."""
    cached = _cache.get()
    if cached is not None:
        return cached
    rows = await pool.fetch("SELECT key, value FROM threshold_config")
    overrides = {r["key"]: float(r["value"]) for r in rows}
    config = ThresholdConfig.from_db_overrides(overrides)
    _cache.set(config)
    return config


def invalidate_threshold_cache() -> None:
    """Force the next ``get_thresholds`` call to reload from the database.

    Call this immediately after any admin write so the current instance picks
    up the change without waiting for the TTL.
    """
    _cache.invalidate()
