/**
 * screens.js
 * Depth-gate (accept_frame) logic for each movement screen type.
 * Direct port of:
 *   movementscreen/screens/squat.py     → acceptFrameSquat()
 *   movementscreen/screens/lunge.py     → acceptFrameLunge()
 *   movementscreen/screens/overhead_reach.py → acceptFrameOverhead()
 */

import { angleBetween } from './geometry.js';
import { LM } from './joint_angles.js';

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Returns true if the landmark at `idx` has visibility > 0.5.
 * Mirrors: Landmark.visible in landmarks.py
 *
 * @param {Array} landmarks - MediaPipe landmark array
 * @param {number} idx
 * @returns {boolean}
 */
function vis(landmarks, idx) {
  return landmarks[idx].visibility > 0.5;
}

/**
 * Returns the [x, y] position of the landmark at `idx`.
 * Mirrors: Landmark.as_array() in landmarks.py
 *
 * @param {Array} landmarks
 * @param {number} idx
 * @returns {number[]}
 */
function xy(landmarks, idx) {
  return [landmarks[idx].x, landmarks[idx].y];
}

// ---------------------------------------------------------------------------
// Constants (mirror Python module-level constants)
// ---------------------------------------------------------------------------

/** Lateral-camera knee angle below which a squat is considered at-depth. */
const SQUAT_DEPTH_THRESHOLD_DEGREES = 115.0;

/**
 * Frontal-camera hip-to-knee normalised height ratio threshold.
 * 0.95 = hip must be within ~5% of knee height (near-parallel).
 * Was 0.85 which fired too early (above-parallel quarter squats passing as "at depth").
 */
const FRONTAL_DEPTH_HIP_FRACTION = 0.95;

/** Lead-knee angle below which a lunge is considered at-depth. */
const LUNGE_DEPTH_THRESHOLD_DEGREES = 105.0;

// ---------------------------------------------------------------------------
// Squat depth gate
// ---------------------------------------------------------------------------

/**
 * Determine whether a squat frame is at depth.
 * Port of: SquatScreen.accept_frame() in squat.py
 *
 * Frontal (anterior/posterior):
 *   Uses normalised hip-to-knee vertical proximity as a depth proxy because
 *   the 2-D knee angle does not capture sagittal depth from this view.
 *
 * Lateral:
 *   Uses the 2-D hip-knee-ankle angle, which correctly captures sagittal flexion.
 *
 * @param {Array}  landmarks    - MediaPipe 33-landmark array
 * @param {string} cameraAngle  - 'anterior' | 'posterior' | 'lateral'
 * @param {string} [lateralSide='left'] - 'left' | 'right' — only used for lateral view
 * @returns {boolean}
 */
export function acceptFrameSquat(landmarks, cameraAngle, lateralSide = 'left') {
  if (cameraAngle === 'anterior' || cameraAngle === 'posterior') {
    // Collect visible shoulder and ankle y-coordinates for body height normalisation
    const shoulderYs = [LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER]
      .filter(idx => vis(landmarks, idx))
      .map(idx => landmarks[idx].y);

    const ankleYs = [LM.LEFT_ANKLE, LM.RIGHT_ANKLE]
      .filter(idx => vis(landmarks, idx))
      .map(idx => landmarks[idx].y);

    if (shoulderYs.length === 0 || ankleYs.length === 0) return false;

    const topY    = Math.min(...shoulderYs);
    const bottomY = Math.max(...ankleYs);
    const bodyH   = bottomY - topY;
    if (bodyH < 0.01) return false;

    for (const [hipIdx, kneeIdx] of [
      [LM.LEFT_HIP,  LM.LEFT_KNEE],
      [LM.RIGHT_HIP, LM.RIGHT_KNEE],
    ]) {
      if (vis(landmarks, hipIdx) && vis(landmarks, kneeIdx)) {
        const hipNorm  = (landmarks[hipIdx].y  - topY) / bodyH;
        const kneeNorm = (landmarks[kneeIdx].y - topY) / bodyH;
        if (kneeNorm > 0 && hipNorm >= kneeNorm * FRONTAL_DEPTH_HIP_FRACTION) {
          return true;
        }
      }
    }
    return false;

  } else {
    // Lateral camera: only check the selected side (the near/visible leg).
    // The far leg's landmarks are unreliable from a side-on view.
    const [hipIdx, kneeIdx, ankleIdx] = lateralSide === 'right'
      ? [LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE]
      : [LM.LEFT_HIP,  LM.LEFT_KNEE,  LM.LEFT_ANKLE];

    if (vis(landmarks, hipIdx) && vis(landmarks, kneeIdx) && vis(landmarks, ankleIdx)) {
      const kneeAngle = angleBetween(
        xy(landmarks, hipIdx),
        xy(landmarks, kneeIdx),
        xy(landmarks, ankleIdx),
      );
      return kneeAngle < SQUAT_DEPTH_THRESHOLD_DEGREES;
    }
    return false;
  }
}

// ---------------------------------------------------------------------------
// Lunge depth gate
// ---------------------------------------------------------------------------

/**
 * Determine whether a lunge frame is at depth.
 * Port of: LungeScreen.accept_frame() in lunge.py
 *
 * Uses the lead-knee 2-D angle, which works from most camera angles because
 * lunge knee flexion is large enough to be visible in both frontal and lateral views.
 *
 * @param {Array}  landmarks   - MediaPipe 33-landmark array
 * @param {string} cameraAngle - 'anterior' | 'posterior' | 'lateral'
 * @param {string} leadSide    - 'left' | 'right'
 * @returns {boolean}
 */
export function acceptFrameLunge(landmarks, cameraAngle, leadSide) {
  const [hipIdx, kneeIdx, ankleIdx] = leadSide === 'left'
    ? [LM.LEFT_HIP,  LM.LEFT_KNEE,  LM.LEFT_ANKLE]
    : [LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE];

  if (vis(landmarks, hipIdx) && vis(landmarks, kneeIdx) && vis(landmarks, ankleIdx)) {
    const kneeAngle = angleBetween(
      xy(landmarks, hipIdx),
      xy(landmarks, kneeIdx),
      xy(landmarks, ankleIdx),
    );
    return kneeAngle < LUNGE_DEPTH_THRESHOLD_DEGREES;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Overhead reach depth gate
// ---------------------------------------------------------------------------

/**
 * Determine whether an overhead reach frame shows hands overhead.
 * Port of: OverheadReachScreen.accept_frame() in overhead_reach.py
 *
 * Accepts frames where at least one visible wrist is above the visible nose
 * (smaller y = higher in image because y increases downward).
 *
 * @param {Array} landmarks - MediaPipe 33-landmark array
 * @returns {boolean}
 */
export function acceptFrameOverhead(landmarks) {
  for (const wristIdx of [LM.LEFT_WRIST, LM.RIGHT_WRIST]) {
    if (vis(landmarks, wristIdx) && vis(landmarks, LM.NOSE)) {
      if (landmarks[wristIdx].y < landmarks[LM.NOSE].y) {
        return true;
      }
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// Gait (walk-past) frame gate
// ---------------------------------------------------------------------------

/**
 * Determine whether landmarks are usable for gait analysis.
 * Unlike squat/lunge there is no depth gate — all in-frame frames are collected.
 * Returns a relAnkleY value (heel Y minus hip Y) used for heel-strike detection.
 *
 * @param {Array}  landmarks   - MediaPipe 33-landmark array
 * @param {string} [lateralSide='left'] - 'left' | 'right' — near leg (closest to camera)
 * @returns {{ inFrame: boolean, relAnkleY: number|null }}
 */
export function acceptFrameGait(landmarks, lateralSide = 'left') {
  const [hipIdx, kneeIdx, ankleIdx, heelIdx] = lateralSide === 'right'
    ? [LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE, LM.RIGHT_HEEL]
    : [LM.LEFT_HIP,  LM.LEFT_KNEE,  LM.LEFT_ANKLE,  LM.LEFT_HEEL];

  if (!vis(landmarks, hipIdx) || !vis(landmarks, kneeIdx) ||
      !vis(landmarks, ankleIdx) || !vis(landmarks, heelIdx)) {
    return { inFrame: false, relAnkleY: null };
  }

  // Heel Y relative to hip Y (same side). Larger = foot near ground (stance); smaller = foot lifted (swing).
  const relAnkleY = landmarks[heelIdx].y - landmarks[hipIdx].y;
  return { inFrame: true, relAnkleY };
}
