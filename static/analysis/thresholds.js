/**
 * thresholds.js
 * Compensation-detection threshold configuration for the browser.
 * Direct port of movementscreen/thresholds.py (ThresholdConfig dataclass defaults
 * and DESCRIPTIONS dict).
 *
 * The server-side TTL+DB cache is replaced with localStorage overrides, which
 * the clinician/admin UI writes via saveThresholdOverrides().
 *
 * localStorage key: 'ms_thresholds'
 * Stored value: JSON object whose keys are a subset of DEFAULT_THRESHOLDS keys
 * and whose values are numbers that override the defaults.
 */

// ---------------------------------------------------------------------------
// Default threshold values — mirrors ThresholdConfig dataclass fields/defaults
// ---------------------------------------------------------------------------

/** @type {Readonly<Record<string, number>>} */
export const DEFAULT_THRESHOLDS = Object.freeze({
  // Knee valgus: medial deviation of knee from hip-ankle line, normalised by hip width.
  // Positive = valgus (knee collapses inward); higher = worse.
  knee_valgus_b: 0.02,   // ~2 % hip-width — barely detectable
  knee_valgus_c: 0.05,   // ~5 % — noticeable inward tracking
  knee_valgus_d: 0.10,   // ~10 % — clinically significant
  knee_valgus_e: 0.15,   // ~15 % — marked collapse
  knee_valgus_f: 0.20,   // ~20 % — major collapse

  // Forward trunk lean (higher is worse)
  trunk_lean_b: 15.0,
  trunk_lean_c: 20.0,
  trunk_lean_d: 35.0,
  trunk_lean_e: 45.0,
  trunk_lean_f: 50.0,

  // Ankle dorsiflexion / heel rise (lower is worse)
  ankle_df_b: 105.0,
  ankle_df_c: 100.0,
  ankle_df_d: 90.0,
  ankle_df_e: 80.0,
  ankle_df_f: 70.0,

  // Lateral trunk shift — normalised image coords (higher is worse)
  lateral_shift_b: 0.015,
  lateral_shift_c: 0.02,
  lateral_shift_d: 0.05,
  lateral_shift_e: 0.065,
  lateral_shift_f: 0.08,

  // Bilateral asymmetry ratio [0–1] (higher is worse)
  asymmetry_b: 0.05,
  asymmetry_c: 0.10,
  asymmetry_d: 0.20,
  asymmetry_e: 0.28,
  asymmetry_f: 0.35,

  // Lateral spinal flexion — degrees (higher is worse)
  lateral_flexion_b: 3.0,
  lateral_flexion_c: 5.0,
  lateral_flexion_d: 10.0,
  lateral_flexion_e: 12.5,
  lateral_flexion_f: 15.0,

  // Spine segmental curvature — deviation from 180° (higher is worse)
  spine_curve_b: 7.0,
  spine_curve_c: 10.0,
  spine_curve_d: 15.0,
  spine_curve_e: 17.5,
  spine_curve_f: 20.0,

  // Upper trunk / cervicothoracic angle from vertical (higher is worse)
  upper_trunk_b: 10.0,
  upper_trunk_c: 15.0,
  upper_trunk_d: 25.0,
  upper_trunk_e: 30.0,
  upper_trunk_f: 35.0,

  // Head forward offset — normalised image coords, lateral view (higher is worse)
  head_forward_b: 0.02,
  head_forward_c: 0.03,
  head_forward_d: 0.05,
  head_forward_e: 0.06,
  head_forward_f: 0.07,

  // Squat-specific trunk lean (optimal: 20–40°; flag above 40°+; higher is worse)
  squat_trunk_lean_b: 40.0,
  squat_trunk_lean_c: 45.0,
  squat_trunk_lean_d: 55.0,
  squat_trunk_lean_e: 60.0,
  squat_trunk_lean_f: 65.0,

  // Tibial angle — lateral DF proxy; lower = more restricted (lower is worse)
  tibial_angle_b: 35.0,
  tibial_angle_c: 28.0,
  tibial_angle_d: 23.0,
  tibial_angle_e: 20.0,
  tibial_angle_f: 18.0,

  // Pelvic tilt from horizontal — anterior view (higher is worse)
  pelvic_tilt_b: 2.0,
  pelvic_tilt_c: 3.0,
  pelvic_tilt_d: 6.0,
  pelvic_tilt_e: 8.0,
  pelvic_tilt_f: 10.0,
});

// ---------------------------------------------------------------------------
// Human-readable descriptions — mirrors DESCRIPTIONS dict in thresholds.py
// ---------------------------------------------------------------------------

/** @type {Readonly<Record<string, string>>} */
export const THRESHOLD_DESCRIPTIONS = Object.freeze({
  // Knee valgus (medial knee deviation normalised by hip width; higher = worse)
  knee_valgus_b: 'Knee medial deviation (normalised) \u2265 which Grade B valgus is flagged (~2\u202f% hip width)',
  knee_valgus_c: 'Knee medial deviation (normalised) \u2265 which Grade C valgus is flagged (~5\u202f% hip width)',
  knee_valgus_d: 'Knee medial deviation (normalised) \u2265 which Grade D valgus is flagged (~10\u202f% hip width)',
  knee_valgus_e: 'Knee medial deviation (normalised) \u2265 which Grade E valgus is flagged (~15\u202f% hip width)',
  knee_valgus_f: 'Knee medial deviation (normalised) \u2265 which Grade F valgus is flagged (~20\u202f% hip width)',
  // Forward trunk lean
  trunk_lean_b: 'Forward trunk lean (\u00b0) \u2265 which Grade B (minimal) lean is flagged',
  trunk_lean_c: 'Forward trunk lean (\u00b0) \u2265 which Grade C (mild) lean is flagged',
  trunk_lean_d: 'Forward trunk lean (\u00b0) \u2265 which Grade D (moderate) lean is flagged',
  trunk_lean_e: 'Forward trunk lean (\u00b0) \u2265 which Grade E (significant) lean is flagged',
  trunk_lean_f: 'Forward trunk lean (\u00b0) \u2265 which Grade F (severe) lean is flagged',
  // Ankle dorsiflexion / heel rise (lower is worse)
  ankle_df_b: 'Ankle dorsiflexion angle (\u00b0) \u2264 which Grade B restriction is flagged',
  ankle_df_c: 'Ankle dorsiflexion angle (\u00b0) \u2264 which Grade C restriction is flagged',
  ankle_df_d: 'Ankle dorsiflexion angle (\u00b0) \u2264 which Grade D restriction is flagged',
  ankle_df_e: 'Ankle dorsiflexion angle (\u00b0) \u2264 which Grade E restriction is flagged',
  ankle_df_f: 'Ankle dorsiflexion angle (\u00b0) \u2264 which Grade F restriction is flagged',
  // Lateral trunk shift
  lateral_shift_b: 'Lateral trunk shift (normalised) \u2265 which Grade B shift is flagged',
  lateral_shift_c: 'Lateral trunk shift (normalised) \u2265 which Grade C shift is flagged',
  lateral_shift_d: 'Lateral trunk shift (normalised) \u2265 which Grade D shift is flagged',
  lateral_shift_e: 'Lateral trunk shift (normalised) \u2265 which Grade E shift is flagged',
  lateral_shift_f: 'Lateral trunk shift (normalised) \u2265 which Grade F shift is flagged',
  // Bilateral asymmetry ratio
  asymmetry_b: 'Asymmetry ratio \u2265 which Grade B bilateral asymmetry is flagged',
  asymmetry_c: 'Asymmetry ratio \u2265 which Grade C bilateral asymmetry is flagged',
  asymmetry_d: 'Asymmetry ratio \u2265 which Grade D bilateral asymmetry is flagged',
  asymmetry_e: 'Asymmetry ratio \u2265 which Grade E bilateral asymmetry is flagged',
  asymmetry_f: 'Asymmetry ratio \u2265 which Grade F bilateral asymmetry is flagged',
  // Lateral spinal flexion
  lateral_flexion_b: 'Lateral trunk tilt (\u00b0) \u2265 which Grade B lateral flexion is flagged',
  lateral_flexion_c: 'Lateral trunk tilt (\u00b0) \u2265 which Grade C lateral flexion is flagged',
  lateral_flexion_d: 'Lateral trunk tilt (\u00b0) \u2265 which Grade D lateral flexion is flagged',
  lateral_flexion_e: 'Lateral trunk tilt (\u00b0) \u2265 which Grade E lateral flexion is flagged',
  lateral_flexion_f: 'Lateral trunk tilt (\u00b0) \u2265 which Grade F lateral flexion is flagged',
  // Spine segmental curvature
  spine_curve_b: 'Spine deviation from 180\u00b0 (\u00b0) \u2265 which Grade B curvature is flagged',
  spine_curve_c: 'Spine deviation from 180\u00b0 (\u00b0) \u2265 which Grade C curvature is flagged',
  spine_curve_d: 'Spine deviation from 180\u00b0 (\u00b0) \u2265 which Grade D curvature is flagged',
  spine_curve_e: 'Spine deviation from 180\u00b0 (\u00b0) \u2265 which Grade E curvature is flagged',
  spine_curve_f: 'Spine deviation from 180\u00b0 (\u00b0) \u2265 which Grade F curvature is flagged',
  // Upper trunk angle
  upper_trunk_b: 'Upper trunk angle from vertical (\u00b0) \u2265 which Grade B flexion is flagged',
  upper_trunk_c: 'Upper trunk angle from vertical (\u00b0) \u2265 which Grade C flexion is flagged',
  upper_trunk_d: 'Upper trunk angle from vertical (\u00b0) \u2265 which Grade D flexion is flagged',
  upper_trunk_e: 'Upper trunk angle from vertical (\u00b0) \u2265 which Grade E flexion is flagged',
  upper_trunk_f: 'Upper trunk angle from vertical (\u00b0) \u2265 which Grade F flexion is flagged',
  // Head forward posture (lateral view)
  head_forward_b: 'Head forward offset (normalised) \u2265 which Grade B forward head posture is flagged',
  head_forward_c: 'Head forward offset (normalised) \u2265 which Grade C forward head posture is flagged',
  head_forward_d: 'Head forward offset (normalised) \u2265 which Grade D forward head posture is flagged',
  head_forward_e: 'Head forward offset (normalised) \u2265 which Grade E forward head posture is flagged',
  head_forward_f: 'Head forward offset (normalised) \u2265 which Grade F forward head posture is flagged',
  // Squat-specific trunk lean
  squat_trunk_lean_b: 'Squat trunk lean (\u00b0) \u2265 which Grade B excessive lean is flagged (optimal: 20\u201340\u00b0)',
  squat_trunk_lean_c: 'Squat trunk lean (\u00b0) \u2265 which Grade C excessive lean is flagged',
  squat_trunk_lean_d: 'Squat trunk lean (\u00b0) \u2265 which Grade D excessive lean is flagged',
  squat_trunk_lean_e: 'Squat trunk lean (\u00b0) \u2265 which Grade E excessive lean is flagged',
  squat_trunk_lean_f: 'Squat trunk lean (\u00b0) \u2265 which Grade F excessive lean is flagged',
  // Tibial angle (lateral DF proxy; lower is worse)
  tibial_angle_b: 'Tibial angle (\u00b0) \u2264 which Grade B dorsiflexion restriction is flagged (optimal: 30\u201340\u00b0)',
  tibial_angle_c: 'Tibial angle (\u00b0) \u2264 which Grade C dorsiflexion restriction is flagged',
  tibial_angle_d: 'Tibial angle (\u00b0) \u2264 which Grade D dorsiflexion restriction is flagged',
  tibial_angle_e: 'Tibial angle (\u00b0) \u2264 which Grade E dorsiflexion restriction is flagged',
  tibial_angle_f: 'Tibial angle (\u00b0) \u2264 which Grade F dorsiflexion restriction is flagged',
  // Pelvic tilt (anterior view)
  pelvic_tilt_b: 'Pelvic tilt (\u00b0) \u2265 which Grade B hip-level asymmetry is flagged',
  pelvic_tilt_c: 'Pelvic tilt (\u00b0) \u2265 which Grade C hip-level asymmetry is flagged',
  pelvic_tilt_d: 'Pelvic tilt (\u00b0) \u2265 which Grade D hip-level asymmetry is flagged',
  pelvic_tilt_e: 'Pelvic tilt (\u00b0) \u2265 which Grade E hip-level asymmetry is flagged',
  pelvic_tilt_f: 'Pelvic tilt (\u00b0) \u2265 which Grade F hip-level asymmetry is flagged',
});

// ---------------------------------------------------------------------------
// localStorage key
// ---------------------------------------------------------------------------

const STORAGE_KEY = 'ms_thresholds';

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Return the current effective thresholds: DEFAULT_THRESHOLDS merged with any
 * numeric overrides stored in localStorage under 'ms_thresholds'.
 *
 * Only keys that exist in DEFAULT_THRESHOLDS and whose stored value parses as
 * a finite number are applied — invalid entries are silently ignored.
 *
 * Mirrors the behaviour of ThresholdConfig.from_db_overrides() but reads from
 * localStorage instead of a database.
 *
 * @returns {Record<string, number>}
 */
export function getThresholds() {
  const result = { ...DEFAULT_THRESHOLDS };

  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw !== null) {
      const overrides = JSON.parse(raw);
      if (overrides !== null && typeof overrides === 'object' && !Array.isArray(overrides)) {
        for (const [key, value] of Object.entries(overrides)) {
          if (Object.prototype.hasOwnProperty.call(DEFAULT_THRESHOLDS, key)) {
            const num = Number(value);
            if (Number.isFinite(num)) {
              result[key] = num;
            }
          }
        }
      }
    }
  } catch {
    // localStorage unavailable or JSON parse error — return defaults
  }

  return result;
}

/**
 * Merge `overrides` into the stored localStorage overrides.
 * Only keys present in DEFAULT_THRESHOLDS with finite numeric values are saved.
 * Existing stored overrides for keys not mentioned in `overrides` are preserved.
 *
 * @param {Record<string, number>} overrides
 */
export function saveThresholdOverrides(overrides) {
  let stored = {};
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw !== null) {
      const parsed = JSON.parse(raw);
      if (parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)) {
        stored = parsed;
      }
    }
  } catch {
    // start fresh if storage is corrupt
  }

  for (const [key, value] of Object.entries(overrides)) {
    if (Object.prototype.hasOwnProperty.call(DEFAULT_THRESHOLDS, key)) {
      const num = Number(value);
      if (Number.isFinite(num)) {
        stored[key] = num;
      }
    }
  }

  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(stored));
  } catch {
    // storage quota exceeded or unavailable — fail silently
  }
}

/**
 * Remove a single threshold override from localStorage, restoring that key to
 * its default value on the next call to getThresholds().
 *
 * @param {string} key - must be a key of DEFAULT_THRESHOLDS
 */
export function clearThresholdOverride(key) {
  let stored = {};
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw !== null) {
      const parsed = JSON.parse(raw);
      if (parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)) {
        stored = parsed;
      }
    }
  } catch {
    return; // nothing to clear
  }

  if (!Object.prototype.hasOwnProperty.call(stored, key)) {
    return; // key not overridden
  }

  delete stored[key];

  try {
    if (Object.keys(stored).length === 0) {
      localStorage.removeItem(STORAGE_KEY);
    } else {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(stored));
    }
  } catch {
    // fail silently
  }
}
