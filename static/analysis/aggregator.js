/**
 * aggregator.js
 * Aggregate per-frame angle data across a full movement screen trial.
 * Direct port of movementscreen/analysis/aggregator.py (TrialAggregator).
 */

import { detectCompensations } from './compensation.js';

// Must match FRONTAL_FULL_DEPTH_FRACTION in screens.js
const FRONTAL_FULL_DEPTH_FRACTION = 0.90;

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
  { key: 'leftFootPronation',      field: 'left_foot_pronation',      name: 'Left Foot Pronation' },
  { key: 'rightFootPronation',     field: 'right_foot_pronation',     name: 'Right Foot Pronation' },
  { key: 'hipLateralShift',        field: 'hip_lateral_shift',        name: 'Hip Lateral Shift' },
  { key: 'shoulderTiltDegrees',    field: 'shoulder_tilt_degrees',    name: 'Shoulder Tilt' },
  { key: 'heelRiseLeft',           field: 'heel_rise_left',           name: 'Heel Rise Left' },
  { key: 'heelRiseRight',          field: 'heel_rise_right',          name: 'Heel Rise Right' },
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
  'leftFootPronation',    'rightFootPronation',
  'heelRiseLeft',         'heelRiseRight',
]);

// Fields handled with signed-abs 75th-percentile strategy
const SIGNED_ABS_75TH = new Set([
  'lateralTrunkShift',
  'pelvicTiltDegrees',
  'headForwardOffset',
  'hipLateralShift',
  'shoulderTiltDegrees',
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
// Frontal valgus is most visible at depth. Only frames at this depth ratio
// or above are used for knee frontal angle (valgus) computation, preventing
// the wide capture gate (0.75) from diluting the peak valgus signal.
const VALGUS_DEPTH_FRACTION = 0.88;

export function createAggregator(screenName) {
  /** @type {Object[]} */
  const frames = [];
  // Subset of frames captured at adequate depth — used for valgus only
  const depthFrames = [];
  let maxDepthRatio = 0;

  /**
   * Add a per-frame angles object to the internal buffer.
   * @param {Object} angles     - camelCase angle fields
   * @param {number} [depthRatio=1] - hip/knee depth ratio from acceptFrameSquat (0–1+)
   */
  function addFrame(angles, depthRatio = 1) {
    frames.push(angles);
    if (depthRatio >= VALGUS_DEPTH_FRACTION) depthFrames.push(angles);
    if (depthRatio > maxDepthRatio) maxDepthRatio = depthRatio;
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

    // Valgus metrics use only deep frames to avoid dilution from descent/ascent
    const VALGUS_KEYS = new Set(['leftKneeFrontalAngle', 'rightKneeFrontalAngle']);

    for (const { key } of TRACKED_FIELDS) {
      // Skip keys handled specially below (SIGNED_ABS_75TH)
      if (SIGNED_ABS_75TH.has(key)) continue;

      // For valgus: prefer depthFrames; fall back to all frames if none at depth
      const sourceFrames = VALGUS_KEYS.has(key) && depthFrames.length > 0 ? depthFrames : frames;

      const vals = sourceFrames
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

    // Knee varus proxies: 25th percentile of ankle-vertical deviation values (most negative
    // = worst varus), negated so positive = varus magnitude for grading.
    // Uses the ankle as a fixed ground reference — immune to lateral hip shift.
    // Uses same deep-frame subset as valgus for the same reason.
    for (const [varusKey, devKey] of [
      ['leftKneeVarus',  'leftKneeAnkleDeviation'],
      ['rightKneeVarus', 'rightKneeAnkleDeviation'],
    ]) {
      const sourceFrames = depthFrames.length > 0 ? depthFrames : frames;
      const vals = sourceFrames.map(f => f[devKey]).filter(v => v != null).sort((a, b) => a - b);
      if (vals.length > 0) {
        // 25th percentile is the most negative value (worst varus end of the distribution)
        const idx = Math.max(0, Math.floor(vals.length * 0.25));
        const p25 = vals[idx];
        // Only set varus proxy if the distribution genuinely leans negative
        if (p25 < -0.02) {
          worst[varusKey] = -p25; // negate so positive = varus magnitude
        }
      }
    }

    // Foot supination proxies: 25th percentile of pronation values (most negative = worst
    // supination), negated so positive = supination magnitude for grading.
    for (const [supinKey, pronKey] of [
      ['leftFootSupination',  'leftFootPronation'],
      ['rightFootSupination', 'rightFootPronation'],
    ]) {
      const vals = frames.map(f => f[pronKey]).filter(v => v != null).sort((a, b) => a - b);
      if (vals.length > 0) {
        const idx = Math.max(0, Math.floor(vals.length * 0.25));
        worst[supinKey] = -vals[idx];
      }
    }

    // -----------------------------------------------------------------------
    // 3. Run compensation detection on the worst-case angles
    // -----------------------------------------------------------------------
    const { findings: rawFindings, worstSeverity: detectedWorst, hasFindings: detectedHasFindings } =
      detectCompensations(worst, cameraAngle, thresholds, screenType, lateralSide);

    // Depth findings for squat — independent of compensation results.
    // Uses maxDepthRatio tracked across all captured frames.
    const GRADE_ORD = { A: 0, B: 1, C: 2, D: 3, E: 4, F: 5 };
    if (screenType === 'squat') {
      if (frames.length === 0) {
        // No squat motion detected at all
        rawFindings.push({
          name: 'Insufficient Squat Depth',
          severity: 'D',
          description:
            'No frames were captured — the movement did not reach a sufficient depth for analysis. ' +
            'Inability to reach depth may indicate ankle dorsiflexion restriction, hip mobility limitation, or pain avoidance.',
          metricValue: 0,
          metricLabel: 'max depth ratio',
        });
      } else if (cameraAngle === 'lateral') {
        // Lateral depth check: grade by minimum knee angle reached.
        // All captured frames are already below the 115° gate; we want to know
        // how far below it they got. Optimal depth ≈ 90° (parallel) or lower.
        const nearKneeKey = lateralSide === 'right' ? 'rightKneeFlexion' : 'leftKneeFlexion';
        const kneeAngles = frames.map(f => f[nearKneeKey]).filter(v => v != null);
        if (kneeAngles.length > 0) {
          const minKnee = Math.min(...kneeAngles);
          let sev = null;
          if      (minKnee >= 108) sev = 'D'; // barely past gate — very shallow
          else if (minKnee >= 100) sev = 'C'; // clearly partial squat
          else if (minKnee >= 92)  sev = 'B'; // near-parallel but short of full depth
          if (sev) {
            rawFindings.push({
              name: 'Reduced Squat Depth',
              severity: sev,
              description:
                `Deepest knee angle reached: ${Math.round(minKnee)}°. ` +
                'Aim for ≤ 90° (thighs parallel or below) for a complete assessment. ' +
                'Limited depth may indicate ankle dorsiflexion restriction or hip mobility limitation.',
              metricValue: Math.round(minKnee),
              metricLabel: 'min knee angle (deg)',
            });
          }
        }
      } else if (maxDepthRatio < FRONTAL_FULL_DEPTH_FRACTION) {
        // Frontal: frames captured but depth was reduced
        rawFindings.push({
          name: 'Reduced Squat Depth',
          severity: 'C',
          description:
            `Squat did not reach near-parallel depth (max depth ratio: ${maxDepthRatio.toFixed(2)}). ` +
            'Reduced depth may reflect ankle dorsiflexion restriction, hip mobility limitation, or deliberate avoidance.',
          metricValue: Math.round(maxDepthRatio * 100) / 100,
          metricLabel: 'max depth ratio',
        });
      }
    }

    let worstSeverity = detectedWorst;
    let hasFindings   = detectedHasFindings;
    for (const f of rawFindings) {
      if (GRADE_ORD[f.severity] > GRADE_ORD[worstSeverity]) worstSeverity = f.severity;
      if (f.severity !== 'A') hasFindings = true;
    }

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

// ---------------------------------------------------------------------------
// Gait aggregator
// ---------------------------------------------------------------------------

/**
 * Create a gait trial aggregator.
 * Collects all in-frame frames (no depth gate) then post-processes them to
 * detect gait cycles via heel-strike detection and extract per-cycle metrics.
 *
 * @param {string} screenName
 * @returns {{ addFrame: Function, finalize: Function }}
 */
export function createGaitAggregator(screenName) {
  const frames = []; // { angles, relAnkleY }

  /** @param {Object} angles  @param {number} relAnkleY */
  function addFrame(angles, relAnkleY) {
    frames.push({ angles, relAnkleY });
  }

  // ±win moving-average smoothing
  function smooth(arr, win) {
    const n = arr.length;
    return arr.map((_, i) => {
      let sum = 0, cnt = 0;
      for (let j = Math.max(0, i - win); j <= Math.min(n - 1, i + win); j++) {
        sum += arr[j]; cnt++;
      }
      return sum / cnt;
    });
  }

  // Detect local maxima in relAnkleY = heel contacts ground (heel strike).
  function detectHeelStrikes(relAnkleYs) {
    const minGap  = 10; // ≥10 frames between strikes (~0.33 s at 30 fps)
    const smoothed = smooth(relAnkleYs, 2);
    const peaks = [];
    for (let i = 1; i < smoothed.length - 1; i++) {
      if (smoothed[i] >= smoothed[i - 1] && smoothed[i] >= smoothed[i + 1]) {
        if (peaks.length === 0 || i - peaks[peaks.length - 1] >= minGap) {
          peaks.push(i);
        } else if (smoothed[i] > smoothed[peaks[peaks.length - 1]]) {
          peaks[peaks.length - 1] = i; // replace if higher within gap
        }
      }
    }
    return peaks;
  }

  function avg(arr) {
    const vals = arr.filter(v => v != null);
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
  }

  /**
   * Finalize gait analysis.
   * @param {string} [lateralSide='left']
   * @param {Object|null} [thresholds=null]
   * @returns {Object} - same shape as createAggregator().finalize()
   */
  function finalize(lateralSide = 'left', thresholds = null) {
    if (frames.length < 10) {
      return {
        screen_name:    screenName,
        screen_type:    'gait',
        frame_count:    frames.length,
        camera_angle:   'lateral',
        step_count:     0,
        worst_severity: 'A',
        has_findings:   false,
        findings:       [],
        stats:          [],
        depth_warning:  true,
      };
    }

    const nearKneeKey    = lateralSide === 'right' ? 'rightKneeFlexion'      : 'leftKneeFlexion';
    const nearTibialKey  = lateralSide === 'right' ? 'tibialAngleRight'       : 'tibialAngleLeft';
    const nearTibialField = lateralSide === 'right' ? 'tibial_angle_right'    : 'tibial_angle_left';
    const nearKneeField   = lateralSide === 'right' ? 'right_knee_flexion'    : 'left_knee_flexion';
    const sideLabel       = lateralSide.charAt(0).toUpperCase() + lateralSide.slice(1);

    // ── Heel-strike detection ──────────────────────────────
    const relAnkleYs  = frames.map(f => f.relAnkleY ?? 0);
    const heelStrikes = detectHeelStrikes(relAnkleYs);
    const stepCount   = Math.max(0, heelStrikes.length - 1);

    // ── Per-cycle metric arrays ────────────────────────────
    const swingKneeMins = [];
    const stanceTrunks  = [];
    const midStanceTibs = [];

    for (let s = 0; s < stepCount; s++) {
      const start    = heelStrikes[s];
      const end      = heelStrikes[s + 1];
      const cycleLen = end - start;
      if (cycleLen < 6) continue;

      // Stance: 0–60 % of cycle; swing: 60–100 %
      const stanceEnd  = Math.floor(start + cycleLen * 0.60);
      // Mid-stance: 20–45 % of cycle (peak DF window)
      const midStSt    = Math.floor(start + cycleLen * 0.20);
      const midStEnd   = Math.floor(start + cycleLen * 0.45);

      // Swing: minimum knee angle = peak flexion (lower = more flexed = better)
      const swingKnees = frames.slice(stanceEnd, end + 1)
        .map(f => f.angles[nearKneeKey]).filter(v => v != null);
      if (swingKnees.length) swingKneeMins.push(Math.min(...swingKnees));

      // Stance: mean trunk lean
      const stanceTrunkVals = frames.slice(start, stanceEnd + 1)
        .map(f => f.angles.trunkLeanDegrees).filter(v => v != null);
      const mTrunk = avg(stanceTrunkVals);
      if (mTrunk != null) stanceTrunks.push(mTrunk);

      // Mid-stance: mean tibial angle (DF proxy)
      const midTibVals = frames.slice(midStSt, midStEnd + 1)
        .map(f => f.angles[nearTibialKey]).filter(v => v != null);
      const mTib = avg(midTibVals);
      if (mTib != null) midStanceTibs.push(mTib);
    }

    // ── Aggregate across cycles (fall back to all-frame stats) ──
    let gaitSwingKneeFlexion = avg(swingKneeMins);
    let gaitTrunkLean        = avg(stanceTrunks);
    let gaitTibialAngle      = avg(midStanceTibs);

    if (gaitSwingKneeFlexion == null) {
      const all = frames.map(f => f.angles[nearKneeKey]).filter(v => v != null);
      if (all.length) gaitSwingKneeFlexion = Math.min(...all);
    }
    if (gaitTrunkLean == null) {
      gaitTrunkLean = avg(frames.map(f => f.angles.trunkLeanDegrees));
    }
    if (gaitTibialAngle == null) {
      gaitTibialAngle = avg(frames.map(f => f.angles[nearTibialKey]));
    }

    // ── Compensation detection ─────────────────────────────
    const gaitAngles = {
      gaitSwingKneeFlexion,
      gaitTrunkLean,
      gaitTibialAngle,
    };

    const { findings: rawFindings, worstSeverity, hasFindings } =
      detectCompensations(gaitAngles, 'lateral', thresholds, 'gait', lateralSide);

    const findings = rawFindings.map(f => ({
      name:         f.name,
      severity:     f.severity,
      description:  f.description,
      metric_value: f.metricValue,
      metric_label: f.metricLabel,
    }));

    // ── Stats ──────────────────────────────────────────────
    const stats = [];

    const allKnees = frames.map(f => f.angles[nearKneeKey]).filter(v => v != null);
    if (allKnees.length) {
      const kMean = allKnees.reduce((a, b) => a + b, 0) / allKnees.length;
      stats.push({ field: nearKneeField, name: `${sideLabel} Knee Flexion`,
        min: round1(Math.min(...allKnees)), max: round1(Math.max(...allKnees)), mean: round1(kMean) });
    }

    const allTrunks = frames.map(f => f.angles.trunkLeanDegrees).filter(v => v != null);
    if (allTrunks.length) {
      const tMean = allTrunks.reduce((a, b) => a + b, 0) / allTrunks.length;
      stats.push({ field: 'trunk_lean_degrees', name: 'Forward Trunk Lean',
        min: round1(Math.min(...allTrunks)), max: round1(Math.max(...allTrunks)), mean: round1(tMean) });
    }

    const allTibs = frames.map(f => f.angles[nearTibialKey]).filter(v => v != null);
    if (allTibs.length) {
      const tibMean = allTibs.reduce((a, b) => a + b, 0) / allTibs.length;
      stats.push({ field: nearTibialField, name: `${sideLabel} Tibial Angle`,
        min: round1(Math.min(...allTibs)), max: round1(Math.max(...allTibs)), mean: round1(tibMean) });
    }

    return {
      screen_name:    screenName,
      screen_type:    'gait',
      frame_count:    frames.length,
      camera_angle:   'lateral',
      step_count:     stepCount,
      worst_severity: worstSeverity,
      has_findings:   hasFindings,
      findings,
      stats,
    };
  }

  return { addFrame, finalize };
}
