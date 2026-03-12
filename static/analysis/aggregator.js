/**
 * aggregator.js
 * Aggregate per-frame angle data across a full movement screen trial.
 * Direct port of movementscreen/analysis/aggregator.py (TrialAggregator).
 */

import { detectCompensations } from './compensation.js';

// ---------------------------------------------------------------------------
// Tracked fields: camelCase key used in JS angle objects, snake_case field
// name used in output stats (to match server response format), and display name.
// ---------------------------------------------------------------------------

const TRACKED_FIELDS = [
  { key: 'leftKneeFlexion',        field: 'left_knee_flexion',        name: 'Left Knee Flexion' },
  { key: 'rightKneeFlexion',       field: 'right_knee_flexion',       name: 'Right Knee Flexion' },
  { key: 'leftHipFlexion',         field: 'left_hip_flexion',         name: 'Left Hip Flexion' },
  { key: 'rightHipFlexion',        field: 'right_hip_flexion',        name: 'Right Hip Flexion' },
  { key: 'leftAnkleDorsiflexion',  field: 'left_ankle_dorsiflexion',  name: 'Left Ankle Dorsiflexion' },
  { key: 'rightAnkleDorsiflexion', field: 'right_ankle_dorsiflexion', name: 'Right Ankle Dorsiflexion' },
  { key: 'trunkLeanDegrees',       field: 'trunk_lean_degrees',       name: 'Trunk Lean Degrees' },
  { key: 'leftShoulderFlexion',    field: 'left_shoulder_flexion',    name: 'Left Shoulder Flexion' },
  { key: 'rightShoulderFlexion',   field: 'right_shoulder_flexion',   name: 'Right Shoulder Flexion' },
  { key: 'lateralFlexionDegrees',  field: 'lateral_flexion_degrees',  name: 'Lateral Flexion Degrees' },
  { key: 'upperTrunkAngle',        field: 'upper_trunk_angle',        name: 'Upper Trunk Angle' },
  { key: 'spineSegmentalAngle',    field: 'spine_segmental_angle',    name: 'Spine Segmental Angle' },
  { key: 'tibialAngleLeft',        field: 'tibial_angle_left',        name: 'Tibial Angle Left' },
  { key: 'tibialAngleRight',       field: 'tibial_angle_right',       name: 'Tibial Angle Right' },
  { key: 'pelvicTiltDegrees',      field: 'pelvic_tilt_degrees',      name: 'Pelvic Tilt Degrees' },
  { key: 'leftKneeFrontalAngle',   field: 'left_knee_frontal_angle',  name: 'Left Knee Frontal Angle' },
  { key: 'rightKneeFrontalAngle',  field: 'right_knee_frontal_angle', name: 'Right Knee Frontal Angle' },
  { key: 'lateralTrunkShift',      field: 'lateral_trunk_shift',      name: 'Lateral Trunk Shift' },
];

// Fields where LOWER is worse — use 25th percentile
const MIN_IS_WORSE = new Set([
  'leftKneeFlexion', 'rightKneeFlexion',
  'leftHipFlexion',  'rightHipFlexion',
  'leftAnkleDorsiflexion', 'rightAnkleDorsiflexion',
  'tibialAngleLeft', 'tibialAngleRight',
  'spineSegmentalAngle',
]);

// Fields where HIGHER is worse — use 75th percentile
const MAX_IS_WORSE = new Set([
  'trunkLeanDegrees',
  'lateralFlexionDegrees',
  'upperTrunkAngle',
  'leftKneeFrontalAngle', 'rightKneeFrontalAngle',
]);

// Fields handled with signed-abs 75th-percentile strategy
const SIGNED_ABS_75TH = new Set([
  'lateralTrunkShift',
  'pelvicTiltDegrees',
  'headForwardOffset',
]);

// ---------------------------------------------------------------------------
// Statistical helpers
// ---------------------------------------------------------------------------

/**
 * Return the value at fractional position p in a pre-sorted array.
 * Mirrors Python's statistics.median / percentile index logic.
 *
 * @param {number[]} sortedArr - sorted ascending
 * @param {number}   p         - fraction in [0, 1]
 * @returns {number}
 */
function percentile(sortedArr, p) {
  const n = sortedArr.length;
  if (n === 0) return 0;
  const idx = Math.min(n - 1, Math.floor(n * p));
  return sortedArr[idx];
}

/**
 * Median of a pre-sorted array.
 * Matches Python statistics.median behaviour (lower median for even lengths).
 *
 * @param {number[]} sortedArr
 * @returns {number}
 */
function median(sortedArr) {
  const n = sortedArr.length;
  if (n === 0) return 0;
  const mid = Math.floor(n / 2);
  if (n % 2 === 1) {
    return sortedArr[mid];
  }
  return (sortedArr[mid - 1] + sortedArr[mid]) / 2;
}

/**
 * Round to 1 decimal place.
 * @param {number} v
 * @returns {number}
 */
function round1(v) {
  return Math.round(v * 10) / 10;
}

// ---------------------------------------------------------------------------
// Public factory
// ---------------------------------------------------------------------------

/**
 * Create a trial aggregator, mirroring Python's TrialAggregator class.
 *
 * @param {string} screenName
 * @returns {{ addFrame: Function, finalize: Function }}
 */
export function createAggregator(screenName) {
  /** @type {Object[]} */
  const frames = [];

  /**
   * Add a per-frame angles object to the internal buffer.
   * @param {Object} angles - camelCase angle fields
   */
  function addFrame(angles) {
    frames.push(angles);
  }

  /**
   * Finalize the trial and return the result object.
   * Port of: TrialAggregator.finalize() in aggregator.py
   *
   * @param {string}      [cameraAngle='anterior']
   * @param {string}      [screenType='']
   * @param {Object|null} [thresholds=null]
   * @param {string|null} [lateralSide=null]  - 'left'|'right' for lateral squat
   * @returns {Object}
   */
  function finalize(cameraAngle = 'anterior', screenType = '', thresholds = null, lateralSide = null) {

    // -----------------------------------------------------------------------
    // 1. Collect stats for each tracked field
    // -----------------------------------------------------------------------
    const statsOut = [];

    for (const { key, field, name } of TRACKED_FIELDS) {
      const vals = frames
        .map(f => f[key])
        .filter(v => v != null);

      if (vals.length === 0) continue;

      const sum = vals.reduce((a, b) => a + b, 0);
      const mean = sum / vals.length;

      statsOut.push({
        field,
        name,
        min:  round1(Math.min(...vals)),
        max:  round1(Math.max(...vals)),
        mean: round1(mean),
      });
    }

    // -----------------------------------------------------------------------
    // 2. Build representative "worst" angles object for compensation detection
    //    Strategy mirrors Python TrialAggregator.finalize()
    // -----------------------------------------------------------------------
    const worst = {};

    for (const { key } of TRACKED_FIELDS) {
      // Skip keys handled specially below (SIGNED_ABS_75TH)
      if (SIGNED_ABS_75TH.has(key)) continue;

      const vals = frames
        .map(f => f[key])
        .filter(v => v != null)
        .sort((a, b) => a - b);

      if (vals.length === 0) continue;

      if (MIN_IS_WORSE.has(key)) {
        // 25th percentile — worse than median but not a single-frame outlier
        const idx = Math.max(0, Math.floor(vals.length * 0.25));
        worst[key] = vals[idx];
      } else if (MAX_IS_WORSE.has(key)) {
        // 75th percentile
        const idx = Math.min(vals.length - 1, Math.floor(vals.length * 0.75));
        worst[key] = vals[idx];
      } else {
        // Median for everything else
        worst[key] = median(vals);
      }
    }

    // Lateral trunk shift: 75th-percentile absolute deviation (signed)
    {
      const shifts = frames
        .map(f => f.lateralTrunkShift)
        .filter(v => v != null)
        .sort((a, b) => Math.abs(a) - Math.abs(b));
      if (shifts.length > 0) {
        const idx = Math.min(shifts.length - 1, Math.floor(shifts.length * 0.75));
        worst.lateralTrunkShift = shifts[idx];
      }
    }

    // Pelvic tilt: signed; 75th-percentile absolute deviation
    {
      const pelvic = frames
        .map(f => f.pelvicTiltDegrees)
        .filter(v => v != null)
        .sort((a, b) => Math.abs(a) - Math.abs(b));
      if (pelvic.length > 0) {
        const idx = Math.min(pelvic.length - 1, Math.floor(pelvic.length * 0.75));
        worst.pelvicTiltDegrees = pelvic[idx];
      }
    }

    // Head forward offset: 75th-percentile absolute deviation
    {
      const headOffsets = frames
        .map(f => f.headForwardOffset)
        .filter(v => v != null)
        .sort((a, b) => Math.abs(a) - Math.abs(b));
      if (headOffsets.length > 0) {
        const idx = Math.min(headOffsets.length - 1, Math.floor(headOffsets.length * 0.75));
        worst.headForwardOffset = headOffsets[idx];
      }
    }

    // -----------------------------------------------------------------------
    // 3. Run compensation detection on the worst-case angles
    // -----------------------------------------------------------------------
    const { findings: rawFindings, worstSeverity, hasFindings } =
      detectCompensations(worst, cameraAngle, thresholds, screenType, lateralSide);

    // Convert camelCase finding keys to snake_case to match server response format
    const findings = rawFindings.map(f => ({
      name:         f.name,
      severity:     f.severity,
      description:  f.description,
      metric_value: f.metricValue,
      metric_label: f.metricLabel,
    }));

    // -----------------------------------------------------------------------
    // 4. Return result object matching server response format
    // -----------------------------------------------------------------------
    return {
      screen_name:    screenName,
      screen_type:    screenType,
      frame_count:    frames.length,
      camera_angle:   cameraAngle,
      worst_severity: worstSeverity,
      has_findings:   hasFindings,
      findings,
      stats:          statsOut,
    };
  }

  return { addFrame, finalize };
}
