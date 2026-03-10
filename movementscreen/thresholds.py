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

try:
    import asyncpg
except ImportError:  # not required outside server context
    asyncpg = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Human-readable descriptions (used by the admin API)
# ---------------------------------------------------------------------------

DESCRIPTIONS: dict[str, str] = {
    # Knee valgus (lower angle = more collapse; lower is worse)
    "knee_valgus_b": "Knee frontal angle (°) ≤ which Grade B (minimal) valgus is flagged (~1° inward)",
    "knee_valgus_c": "Knee frontal angle (°) ≤ which Grade C (mild) valgus is flagged (~3° inward)",
    "knee_valgus_d": "Knee frontal angle (°) ≤ which Grade D (moderate) valgus is flagged (~7° inward)",
    "knee_valgus_e": "Knee frontal angle (°) ≤ which Grade E (significant) valgus is flagged (~10° inward)",
    "knee_valgus_f": "Knee frontal angle (°) ≤ which Grade F (severe) valgus is flagged (~15° inward)",
    # Forward trunk lean
    "trunk_lean_b": "Forward trunk lean (°) ≥ which Grade B (minimal) lean is flagged",
    "trunk_lean_c": "Forward trunk lean (°) ≥ which Grade C (mild) lean is flagged",
    "trunk_lean_d": "Forward trunk lean (°) ≥ which Grade D (moderate) lean is flagged",
    "trunk_lean_e": "Forward trunk lean (°) ≥ which Grade E (significant) lean is flagged",
    "trunk_lean_f": "Forward trunk lean (°) ≥ which Grade F (severe) lean is flagged",
    # Ankle dorsiflexion / heel rise (lower is worse)
    "ankle_df_b": "Ankle dorsiflexion angle (°) ≤ which Grade B restriction is flagged",
    "ankle_df_c": "Ankle dorsiflexion angle (°) ≤ which Grade C restriction is flagged",
    "ankle_df_d": "Ankle dorsiflexion angle (°) ≤ which Grade D restriction is flagged",
    "ankle_df_e": "Ankle dorsiflexion angle (°) ≤ which Grade E restriction is flagged",
    "ankle_df_f": "Ankle dorsiflexion angle (°) ≤ which Grade F restriction is flagged",
    # Lateral trunk shift
    "lateral_shift_b": "Lateral trunk shift (normalised) ≥ which Grade B shift is flagged",
    "lateral_shift_c": "Lateral trunk shift (normalised) ≥ which Grade C shift is flagged",
    "lateral_shift_d": "Lateral trunk shift (normalised) ≥ which Grade D shift is flagged",
    "lateral_shift_e": "Lateral trunk shift (normalised) ≥ which Grade E shift is flagged",
    "lateral_shift_f": "Lateral trunk shift (normalised) ≥ which Grade F shift is flagged",
    # Bilateral asymmetry ratio
    "asymmetry_b": "Asymmetry ratio ≥ which Grade B bilateral asymmetry is flagged",
    "asymmetry_c": "Asymmetry ratio ≥ which Grade C bilateral asymmetry is flagged",
    "asymmetry_d": "Asymmetry ratio ≥ which Grade D bilateral asymmetry is flagged",
    "asymmetry_e": "Asymmetry ratio ≥ which Grade E bilateral asymmetry is flagged",
    "asymmetry_f": "Asymmetry ratio ≥ which Grade F bilateral asymmetry is flagged",
    # Lateral spinal flexion
    "lateral_flexion_b": "Lateral trunk tilt (°) ≥ which Grade B lateral flexion is flagged",
    "lateral_flexion_c": "Lateral trunk tilt (°) ≥ which Grade C lateral flexion is flagged",
    "lateral_flexion_d": "Lateral trunk tilt (°) ≥ which Grade D lateral flexion is flagged",
    "lateral_flexion_e": "Lateral trunk tilt (°) ≥ which Grade E lateral flexion is flagged",
    "lateral_flexion_f": "Lateral trunk tilt (°) ≥ which Grade F lateral flexion is flagged",
    # Spine segmental curvature
    "spine_curve_b": "Spine deviation from 180° (°) ≥ which Grade B curvature is flagged",
    "spine_curve_c": "Spine deviation from 180° (°) ≥ which Grade C curvature is flagged",
    "spine_curve_d": "Spine deviation from 180° (°) ≥ which Grade D curvature is flagged",
    "spine_curve_e": "Spine deviation from 180° (°) ≥ which Grade E curvature is flagged",
    "spine_curve_f": "Spine deviation from 180° (°) ≥ which Grade F curvature is flagged",
    # Upper trunk angle
    "upper_trunk_b": "Upper trunk angle from vertical (°) ≥ which Grade B flexion is flagged",
    "upper_trunk_c": "Upper trunk angle from vertical (°) ≥ which Grade C flexion is flagged",
    "upper_trunk_d": "Upper trunk angle from vertical (°) ≥ which Grade D flexion is flagged",
    "upper_trunk_e": "Upper trunk angle from vertical (°) ≥ which Grade E flexion is flagged",
    "upper_trunk_f": "Upper trunk angle from vertical (°) ≥ which Grade F flexion is flagged",
    # Head forward posture (lateral view)
    "head_forward_b": "Head forward offset (normalised) ≥ which Grade B forward head posture is flagged",
    "head_forward_c": "Head forward offset (normalised) ≥ which Grade C forward head posture is flagged",
    "head_forward_d": "Head forward offset (normalised) ≥ which Grade D forward head posture is flagged",
    "head_forward_e": "Head forward offset (normalised) ≥ which Grade E forward head posture is flagged",
    "head_forward_f": "Head forward offset (normalised) ≥ which Grade F forward head posture is flagged",
    # Squat-specific trunk lean
    "squat_trunk_lean_b": "Squat trunk lean (°) ≥ which Grade B excessive lean is flagged (optimal: 20–40°)",
    "squat_trunk_lean_c": "Squat trunk lean (°) ≥ which Grade C excessive lean is flagged",
    "squat_trunk_lean_d": "Squat trunk lean (°) ≥ which Grade D excessive lean is flagged",
    "squat_trunk_lean_e": "Squat trunk lean (°) ≥ which Grade E excessive lean is flagged",
    "squat_trunk_lean_f": "Squat trunk lean (°) ≥ which Grade F excessive lean is flagged",
    # Tibial angle (lateral DF proxy; lower is worse)
    "tibial_angle_b": "Tibial angle (°) ≤ which Grade B dorsiflexion restriction is flagged (optimal: 30–40°)",
    "tibial_angle_c": "Tibial angle (°) ≤ which Grade C dorsiflexion restriction is flagged",
    "tibial_angle_d": "Tibial angle (°) ≤ which Grade D dorsiflexion restriction is flagged",
    "tibial_angle_e": "Tibial angle (°) ≤ which Grade E dorsiflexion restriction is flagged",
    "tibial_angle_f": "Tibial angle (°) ≤ which Grade F dorsiflexion restriction is flagged",
    # Pelvic tilt (anterior view)
    "pelvic_tilt_b": "Pelvic tilt (°) ≥ which Grade B hip-level asymmetry is flagged",
    "pelvic_tilt_c": "Pelvic tilt (°) ≥ which Grade C hip-level asymmetry is flagged",
    "pelvic_tilt_d": "Pelvic tilt (°) ≥ which Grade D hip-level asymmetry is flagged",
    "pelvic_tilt_e": "Pelvic tilt (°) ≥ which Grade E hip-level asymmetry is flagged",
    "pelvic_tilt_f": "Pelvic tilt (°) ≥ which Grade F hip-level asymmetry is flagged",
}


# ---------------------------------------------------------------------------
# ThresholdConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class ThresholdConfig:
    """All configurable compensation thresholds with their hardcoded defaults."""

    # Knee valgus (frontal angle; lower = more collapsed; lower is worse)
    # 180° = perfectly straight alignment in the frontal plane
    knee_valgus_b: float = 179.0  # ~1° inward — barely detectable
    knee_valgus_c: float = 177.0  # ~3° inward — noticeable
    knee_valgus_d: float = 173.0  # ~7° inward — clinically significant
    knee_valgus_e: float = 170.0  # ~10° inward — marked collapse
    knee_valgus_f: float = 165.0  # ~15° inward — major collapse

    # Forward trunk lean (higher is worse)
    trunk_lean_b: float = 15.0
    trunk_lean_c: float = 20.0
    trunk_lean_d: float = 35.0
    trunk_lean_e: float = 45.0
    trunk_lean_f: float = 50.0

    # Ankle dorsiflexion / heel rise (lower is worse)
    ankle_df_b: float = 105.0
    ankle_df_c: float = 100.0
    ankle_df_d: float = 90.0
    ankle_df_e: float = 80.0
    ankle_df_f: float = 70.0

    # Lateral trunk shift — normalised image coords (higher is worse)
    lateral_shift_b: float = 0.015
    lateral_shift_c: float = 0.02
    lateral_shift_d: float = 0.05
    lateral_shift_e: float = 0.065
    lateral_shift_f: float = 0.08

    # Bilateral asymmetry ratio [0–1] (higher is worse)
    asymmetry_b: float = 0.05
    asymmetry_c: float = 0.10
    asymmetry_d: float = 0.20
    asymmetry_e: float = 0.28
    asymmetry_f: float = 0.35

    # Lateral spinal flexion — degrees (higher is worse)
    lateral_flexion_b: float = 3.0
    lateral_flexion_c: float = 5.0
    lateral_flexion_d: float = 10.0
    lateral_flexion_e: float = 12.5
    lateral_flexion_f: float = 15.0

    # Spine segmental curvature — deviation from 180° (higher is worse)
    spine_curve_b: float = 7.0
    spine_curve_c: float = 10.0
    spine_curve_d: float = 15.0
    spine_curve_e: float = 17.5
    spine_curve_f: float = 20.0

    # Upper trunk / cervicothoracic angle from vertical (higher is worse)
    upper_trunk_b: float = 10.0
    upper_trunk_c: float = 15.0
    upper_trunk_d: float = 25.0
    upper_trunk_e: float = 30.0
    upper_trunk_f: float = 35.0

    # Head forward offset — normalised image coords, lateral view (higher is worse)
    head_forward_b: float = 0.02
    head_forward_c: float = 0.03
    head_forward_d: float = 0.05
    head_forward_e: float = 0.06
    head_forward_f: float = 0.07

    # Squat-specific trunk lean (optimal: 20–40°; flag above 40°+; higher is worse)
    squat_trunk_lean_b: float = 40.0
    squat_trunk_lean_c: float = 45.0
    squat_trunk_lean_d: float = 55.0
    squat_trunk_lean_e: float = 60.0
    squat_trunk_lean_f: float = 65.0

    # Tibial angle — lateral DF proxy; lower = more restricted (lower is worse)
    tibial_angle_b: float = 35.0
    tibial_angle_c: float = 28.0
    tibial_angle_d: float = 23.0
    tibial_angle_e: float = 20.0
    tibial_angle_f: float = 18.0

    # Pelvic tilt from horizontal — anterior view (higher is worse)
    pelvic_tilt_b: float = 2.0
    pelvic_tilt_c: float = 3.0
    pelvic_tilt_d: float = 6.0
    pelvic_tilt_e: float = 8.0
    pelvic_tilt_f: float = 10.0

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

async def get_thresholds(pool: "asyncpg.Pool") -> ThresholdConfig:
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
