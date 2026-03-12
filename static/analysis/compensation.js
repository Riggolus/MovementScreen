/**
 * compensation.js
 * Compensation pattern detection from joint angle data.
 * Direct port of movementscreen/analysis/compensation.py
 *
 * Grades: A (pass) → F (severe).
 */

import { getThresholds } from './thresholds.js';
import { asymmetryRatio } from './geometry.js';

// ---------------------------------------------------------------------------
// Internal grade helper
// ---------------------------------------------------------------------------

/**
 * Return the compensation grade string for `value` given ordered band boundaries.
 *
 * lowerIsWorse=false (higher = worse):
 *   A < b ≤ B < c ≤ C < d ≤ D < e ≤ E < f ≤ F
 *
 * lowerIsWorse=true (lower = worse):
 *   A > b ≥ B > c ≥ C > d ≥ D > e ≥ E > f ≥ F
 *
 * Port of: _grade_from_thresholds() in compensation.py
 *
 * @param {number}       value
 * @param {number}       b
 * @param {number}       c
 * @param {number}       d
 * @param {number|null}  e
 * @param {number|null}  f
 * @param {boolean}      lowerIsWorse
 * @returns {'A'|'B'|'C'|'D'|'E'|'F'}
 */
function gradeFromThresholds(value, b, c, d, e = null, f = null, lowerIsWorse = false) {
  if (lowerIsWorse) {
    if (f !== null && value <= f) return 'F';
    if (e !== null && value <= e) return 'E';
    if (value <= d) return 'D';
    if (value <= c) return 'C';
    if (value <= b) return 'B';
    return 'A';
  } else {
    if (f !== null && value >= f) return 'F';
    if (e !== null && value >= e) return 'E';
    if (value >= d) return 'D';
    if (value >= c) return 'C';
    if (value >= b) return 'B';
    return 'A';
  }
}

// ---------------------------------------------------------------------------
// Bilateral asymmetry helper
// ---------------------------------------------------------------------------

/**
 * Port of: _check_bilateral_asymmetry() in compensation.py
 *
 * @param {Array}  findings - mutable findings array
 * @param {Object} t        - threshold config
 * @param {string} label
 * @param {number|null} left
 * @param {number|null} right
 */
function checkBilateralAsymmetry(findings, t, label, left, right) {
  if (left == null || right == null) return;

  const ratio = asymmetryRatio(left, right);
  const sev = gradeFromThresholds(
    ratio,
    t.asymmetry_b, t.asymmetry_c, t.asymmetry_d,
    t.asymmetry_e, t.asymmetry_f,
    false,
  );
  if (sev === 'A') return;

  findings.push({
    name: `Bilateral ${label} Asymmetry`,
    severity: sev,
    description: `left (${left.toFixed(1)}°) vs right (${right.toFixed(1)}°) ${label.toLowerCase()} differ significantly`,
    metricValue: Math.round(ratio * 1000) / 1000,
    metricLabel: 'asymmetry ratio',
  });
}

// ---------------------------------------------------------------------------
// Grade ordering helper for worstSeverity
// ---------------------------------------------------------------------------

const GRADE_ORDER = { A: 0, B: 1, C: 2, D: 3, E: 4, F: 5 };

// ---------------------------------------------------------------------------
// Main export
// ---------------------------------------------------------------------------

/**
 * Evaluate a joint angles snapshot and return all compensation findings.
 *
 * Port of: detect_compensations() in compensation.py
 *
 * @param {Object}      angles               - camelCase angle fields (nulls allowed)
 * @param {string}      [cameraAngle='anterior']
 * @param {Object|null} [thresholds=null]    - threshold config (from getThresholds())
 * @param {string}      [screenType='']
 * @param {string|null} [lateralSide=null]   - 'left'|'right' — for lateral view, only analyse this side
 * @returns {{ findings: Array, worstSeverity: string, hasFindings: boolean }}
 */
export function detectCompensations(
  angles,
  cameraAngle = 'anterior',
  thresholds = null,
  screenType = '',
  lateralSide = null,
) {
  const t = thresholds ?? getThresholds();
  const findings = [];

  const isFrontal = cameraAngle === 'anterior' || cameraAngle === 'posterior';
  const isLateral = cameraAngle === 'lateral';

  // =========================================================
  // FRONTAL-PLANE CHECKS  (anterior / posterior camera only)
  // =========================================================
  if (isFrontal) {

    // 1. Knee valgus (frontal plane collapse)
    //    Positive deviation = knee medial to hip-ankle line = valgus.
    //    Higher deviation = worse.
    for (const [side, deviation] of [
      ['Left',  angles.leftKneeFrontalAngle],
      ['Right', angles.rightKneeFrontalAngle],
    ]) {
      if (deviation != null) {
        const sev = gradeFromThresholds(
          deviation,
          t.knee_valgus_b, t.knee_valgus_c, t.knee_valgus_d,
          t.knee_valgus_e, t.knee_valgus_f,
          false,
        );
        if (sev !== 'A') {
          findings.push({
            name: `${side} Knee Valgus`,
            severity: sev,
            description: `${side.toLowerCase()} knee collapsing medially toward the midline`,
            metricValue: Math.round(deviation * 1000) / 1000,
            metricLabel: 'knee medial deviation (normalized)',
          });
        }
      }
    }

    // 2. Lateral trunk shift
    //    Only valid from frontal view. Higher abs shift = worse.
    if (angles.lateralTrunkShift != null) {
      const shift = Math.abs(angles.lateralTrunkShift);
      const sev = gradeFromThresholds(
        shift,
        t.lateral_shift_b, t.lateral_shift_c, t.lateral_shift_d,
        t.lateral_shift_e, t.lateral_shift_f,
        false,
      );
      if (sev !== 'A') {
        const direction = angles.lateralTrunkShift > 0 ? 'right' : 'left';
        findings.push({
          name: `Lateral Trunk Shift (${direction})`,
          severity: sev,
          description: `shoulders shifted laterally toward the ${direction} relative to hips`,
          metricValue: Math.round(shift * 1000) / 1000,
          metricLabel: 'shift (normalized)',
        });
      }
    }

    // 3. Pelvic tilt (hip line from horizontal)
    //    Signed: positive = right hip drops lower; higher abs = worse.
    if (angles.pelvicTiltDegrees != null) {
      const tilt = Math.abs(angles.pelvicTiltDegrees);
      const sev = gradeFromThresholds(
        tilt,
        t.pelvic_tilt_b, t.pelvic_tilt_c, t.pelvic_tilt_d,
        t.pelvic_tilt_e, t.pelvic_tilt_f,
        false,
      );
      if (sev !== 'A') {
        const dropSide = angles.pelvicTiltDegrees > 0 ? 'right' : 'left';
        findings.push({
          name: `Pelvic Tilt (${dropSide} drop)`,
          severity: sev,
          description:
            `${dropSide.charAt(0).toUpperCase() + dropSide.slice(1)} hip dropping lower than the opposite side — ` +
            'suggests hip abductor weakness or leg-length difference',
          metricValue: Math.round(tilt * 10) / 10,
          metricLabel: 'pelvic tilt (deg)',
        });
      }
    }

    // 4. Bilateral symmetry (L vs R) — only valid from a frontal camera
    checkBilateralAsymmetry(findings, t, 'Knee Flexion',     angles.leftKneeFlexion,     angles.rightKneeFlexion);
    checkBilateralAsymmetry(findings, t, 'Hip Flexion',      angles.leftHipFlexion,      angles.rightHipFlexion);
    checkBilateralAsymmetry(findings, t, 'Shoulder Flexion', angles.leftShoulderFlexion, angles.rightShoulderFlexion);
  }

  // =========================================================
  // SAGITTAL-PLANE CHECKS  (lateral camera only)
  // =========================================================
  if (isLateral) {

    // When a lateralSide is specified, null out the far side's per-leg data so
    // only the near (visible) leg contributes to tibial angle and DF checks.
    if (lateralSide === 'left') {
      angles = { ...angles, tibialAngleRight: null, rightAnkleDorsiflexion: null };
    } else if (lateralSide === 'right') {
      angles = { ...angles, tibialAngleLeft: null, leftAnkleDorsiflexion: null };
    }

    // 6. Excessive forward trunk lean
    if (angles.trunkLeanDegrees != null) {
      let lb, lc, ld, le, lf, note;
      if (screenType === 'squat') {
        lb = t.squat_trunk_lean_b;
        lc = t.squat_trunk_lean_c;
        ld = t.squat_trunk_lean_d;
        le = t.squat_trunk_lean_e;
        lf = t.squat_trunk_lean_f;
        note = ' (optimal squat range: 20–40°)';
      } else {
        lb = t.trunk_lean_b;
        lc = t.trunk_lean_c;
        ld = t.trunk_lean_d;
        le = t.trunk_lean_e;
        lf = t.trunk_lean_f;
        note = '';
      }
      const sev = gradeFromThresholds(
        angles.trunkLeanDegrees,
        lb, lc, ld, le, lf,
        false,
      );
      if (sev !== 'A') {
        findings.push({
          name: 'Excessive Forward Trunk Lean',
          severity: sev,
          description: `trunk angled excessively forward from vertical${note}`,
          metricValue: Math.round(angles.trunkLeanDegrees * 10) / 10,
          metricLabel: 'trunk lean (deg)',
        });
      }
    }

    // 7. Ankle dorsiflexion — tibial angle (primary lateral proxy)
    //    Tibia angle from vertical at squat depth. Optimal: 30–40°. Lower = worse.
    for (const [side, angle] of [
      ['Left',  angles.tibialAngleLeft],
      ['Right', angles.tibialAngleRight],
    ]) {
      if (angle != null) {
        const sev = gradeFromThresholds(
          angle,
          t.tibial_angle_b, t.tibial_angle_c, t.tibial_angle_d,
          t.tibial_angle_e, t.tibial_angle_f,
          true,
        );
        if (sev !== 'A') {
          findings.push({
            name: `${side} Restricted Dorsiflexion`,
            severity: sev,
            description:
              `${side.toLowerCase()} tibia inclination suggests limited ankle dorsiflexion ` +
              `(optimal at squat depth: 30–40°)`,
            metricValue: Math.round(angle * 10) / 10,
            metricLabel: 'tibial angle (deg)',
          });
        }
      }
    }

    // 8. Ankle dorsiflexion — knee-ankle-foot proxy (secondary)
    //    Only flag if tibial angle didn't already fire for the same side.
    const tibialFlagged = new Set(
      findings
        .filter(f => f.name.includes('Restricted Dorsiflexion'))
        .map(f => f.name.startsWith('Left') ? 'Left' : 'Right'),
    );

    for (const [side, df] of [
      ['Left',  angles.leftAnkleDorsiflexion],
      ['Right', angles.rightAnkleDorsiflexion],
    ]) {
      if (df != null && !tibialFlagged.has(side)) {
        const sev = gradeFromThresholds(
          df,
          t.ankle_df_b, t.ankle_df_c, t.ankle_df_d,
          t.ankle_df_e, t.ankle_df_f,
          true,
        );
        if (sev !== 'A') {
          findings.push({
            name: `${side} Heel Rise / Limited Dorsiflexion`,
            severity: sev,
            description: `restricted ankle dorsiflexion on ${side.toLowerCase()} side`,
            metricValue: Math.round(df * 10) / 10,
            metricLabel: 'ankle angle (deg)',
          });
        }
      }
    }

    // 9. Head forward posture
    if (angles.headForwardOffset != null) {
      const offset = Math.abs(angles.headForwardOffset);
      const sev = gradeFromThresholds(
        offset,
        t.head_forward_b, t.head_forward_c, t.head_forward_d,
        t.head_forward_e, t.head_forward_f,
        false,
      );
      if (sev !== 'A') {
        findings.push({
          name: 'Head Forward Posture',
          severity: sev,
          description: 'ear positioned significantly ahead of shoulder in the sagittal plane',
          metricValue: Math.round(offset * 1000) / 1000,
          metricLabel: 'head offset (normalized)',
        });
      }
    }

    // 10. Upper trunk flexion
    if (angles.upperTrunkAngle != null) {
      const sev = gradeFromThresholds(
        angles.upperTrunkAngle,
        t.upper_trunk_b, t.upper_trunk_c, t.upper_trunk_d,
        t.upper_trunk_e, t.upper_trunk_f,
        false,
      );
      if (sev !== 'A') {
        findings.push({
          name: 'Upper Trunk Flexion',
          severity: sev,
          description:
            'head and upper thoracic spine forward from vertical; ' +
            'may indicate thoracic kyphosis or cervical hyperlordosis',
          metricValue: Math.round(angles.upperTrunkAngle * 10) / 10,
          metricLabel: 'upper trunk angle (deg)',
        });
      }
    }

    // 11. Spinal segmental curvature
    //     Ear-shoulder-hip angle: 180° = straight; deviations = curvature.
    if (angles.spineSegmentalAngle != null) {
      const deviation = 180.0 - angles.spineSegmentalAngle;
      if (deviation > 0) {
        const sev = gradeFromThresholds(
          deviation,
          t.spine_curve_b, t.spine_curve_c, t.spine_curve_d,
          t.spine_curve_e, t.spine_curve_f,
          false,
        );
        if (sev !== 'A') {
          findings.push({
            name: 'Spinal Segmental Curvature',
            severity: sev,
            description:
              'segmental bend between upper and lower spine; ' +
              'may indicate thoracic kyphosis or lumbar lordosis',
            metricValue: Math.round(deviation * 10) / 10,
            metricLabel: 'segmental deviation (deg)',
          });
        }
      }
    }

    // 12. Tibial bilateral asymmetry (lateral) — skip when a single side is selected
    if (!lateralSide) {
      checkBilateralAsymmetry(findings, t, 'Tibial Inclination', angles.tibialAngleLeft, angles.tibialAngleRight);
    }
  }

  // =========================================================
  // GAIT  (screenType === 'gait')
  // =========================================================
  if (screenType === 'gait') {

    // 1. Stiff-legged gait: swing-phase knee doesn't flex enough.
    //    peakSwingKneeFlexion = minimum near-knee angle during swing.
    //    Higher value = knee barely bends = worse.
    if (angles.gaitSwingKneeFlexion != null) {
      const sev = gradeFromThresholds(
        angles.gaitSwingKneeFlexion,
        t.gait_swing_knee_b, t.gait_swing_knee_c, t.gait_swing_knee_d,
        t.gait_swing_knee_e, t.gait_swing_knee_f,
        false,
      );
      if (sev !== 'A') {
        findings.push({
          name: 'Reduced Swing Phase Knee Flexion',
          severity: sev,
          description:
            'Knee does not flex sufficiently during the swing phase — suggests a stiff-legged or antalgic gait pattern',
          metricValue: Math.round(angles.gaitSwingKneeFlexion * 10) / 10,
          metricLabel: 'min swing knee angle (deg)',
        });
      }
    }

    // 2. Excessive forward trunk lean during gait.
    //    Normal walking lean: ~5–8°. Higher = worse.
    if (angles.gaitTrunkLean != null) {
      const sev = gradeFromThresholds(
        angles.gaitTrunkLean,
        t.gait_trunk_lean_b, t.gait_trunk_lean_c, t.gait_trunk_lean_d,
        t.gait_trunk_lean_e, t.gait_trunk_lean_f,
        false,
      );
      if (sev !== 'A') {
        findings.push({
          name: 'Excessive Forward Trunk Lean (Gait)',
          severity: sev,
          description:
            'Trunk angled excessively forward during the stance phase of gait — may indicate hip extensor weakness or pain avoidance',
          metricValue: Math.round(angles.gaitTrunkLean * 10) / 10,
          metricLabel: 'mean stance trunk lean (deg)',
        });
      }
    }

    // 3. Restricted ankle dorsiflexion / limited push-off.
    //    Tibial angle at mid-stance: lower = tibia more vertical = restricted DF. Lower is worse.
    if (angles.gaitTibialAngle != null) {
      const sev = gradeFromThresholds(
        angles.gaitTibialAngle,
        t.gait_tibial_b, t.gait_tibial_c, t.gait_tibial_d,
        t.gait_tibial_e, t.gait_tibial_f,
        true,
      );
      if (sev !== 'A') {
        findings.push({
          name: 'Reduced Ankle Dorsiflexion (Gait)',
          severity: sev,
          description:
            'Tibia remains too upright at mid-stance — suggests restricted ankle dorsiflexion limiting forward progression',
          metricValue: Math.round(angles.gaitTibialAngle * 10) / 10,
          metricLabel: 'tibial angle at mid-stance (deg)',
        });
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Build result
  // ---------------------------------------------------------------------------
  let worstSeverity = 'A';
  for (const f of findings) {
    if (GRADE_ORDER[f.severity] > GRADE_ORDER[worstSeverity]) {
      worstSeverity = f.severity;
    }
  }

  return {
    findings,
    worstSeverity,
    hasFindings: findings.length > 0,
  };
}
