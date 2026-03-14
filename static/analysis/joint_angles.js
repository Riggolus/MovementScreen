/**
 * joint_angles.js
 * Compute named joint angles from a MediaPipe landmark array.
 * Direct port of movementscreen/analysis/joint_angles.py
 *
 * The `landmarks` parameter is the array of 33 objects produced by the
 * MediaPipe JS Pose solution, each with {x, y, z, visibility, presence}.
 * x/y are normalised [0,1] image coordinates; y increases downward.
 * Visibility threshold: 0.5 (mirrors the Python `visible` property).
 */

import { angleBetween, midpoint, verticalAngle, subtract } from './geometry.js';

// ---------------------------------------------------------------------------
// Landmark index constants — mirrors LM (IntEnum) in landmarks.py
// ---------------------------------------------------------------------------
export const LM = Object.freeze({
  NOSE: 0,
  LEFT_EYE_INNER: 1,
  LEFT_EYE: 2,
  LEFT_EYE_OUTER: 3,
  RIGHT_EYE_INNER: 4,
  RIGHT_EYE: 5,
  RIGHT_EYE_OUTER: 6,
  LEFT_EAR: 7,
  RIGHT_EAR: 8,
  MOUTH_LEFT: 9,
  MOUTH_RIGHT: 10,
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
  LEFT_PINKY: 17,
  LEFT_INDEX: 19,
  RIGHT_INDEX: 20,
  LEFT_THUMB: 21,
  RIGHT_THUMB: 22,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  LEFT_KNEE: 25,
  RIGHT_KNEE: 26,
  LEFT_ANKLE: 27,
  RIGHT_ANKLE: 28,
  LEFT_HEEL: 29,
  RIGHT_HEEL: 30,
  LEFT_FOOT_INDEX: 31,
  RIGHT_FOOT_INDEX: 32,
});

// ---------------------------------------------------------------------------
// Internal helpers (mirror Landmark.visible and Landmark.as_array)
// ---------------------------------------------------------------------------

/** @param {{visibility: number}} lm */
function visible(lm) {
  return lm.visibility > 0.5;
}

/**
 * Extract the 2-D [x, y] array from a landmark object.
 * Mirrors Landmark.as_array(dims=2).
 * @param {{x: number, y: number}} lm
 * @returns {number[]}
 */
function asArray(lm) {
  return [lm.x, lm.y];
}

/**
 * Return true if all listed landmark indices have visibility > 0.5.
 * Mirrors: all(g(lm).visible for lm in (...))
 * @param {object[]} landmarks
 * @param {number[]} indices
 * @returns {boolean}
 */
function allVisible(landmarks, indices) {
  return indices.every((i) => visible(landmarks[i]));
}

/**
 * Return true if both landmarks are visible.
 * Mirrors: PoseFrame.bilateral_visible(left, right)
 * @param {object[]} landmarks
 * @param {number} leftIdx
 * @param {number} rightIdx
 * @returns {boolean}
 */
function bilateralVisible(landmarks, leftIdx, rightIdx) {
  return visible(landmarks[leftIdx]) && visible(landmarks[rightIdx]);
}

// ---------------------------------------------------------------------------
// Main export
// ---------------------------------------------------------------------------

/**
 * Compute all joint angles from a single MediaPipe landmark array.
 * Direct port of compute_joint_angles(frame) in joint_angles.py.
 *
 * @param {Array<{x:number, y:number, z:number, visibility:number, presence:number}>} landmarks
 *   The 33-element array from MediaPipe BlazePose.
 * @returns {{
 *   leftKneeFlexion: number|null,
 *   rightKneeFlexion: number|null,
 *   leftHipFlexion: number|null,
 *   rightHipFlexion: number|null,
 *   leftAnkleDorsiflexion: number|null,
 *   rightAnkleDorsiflexion: number|null,
 *   trunkLeanDegrees: number|null,
 *   lateralTrunkShift: number|null,
 *   leftShoulderFlexion: number|null,
 *   rightShoulderFlexion: number|null,
 *   leftElbowAngle: number|null,
 *   rightElbowAngle: number|null,
 *   leftKneeFrontalAngle: number|null,
 *   rightKneeFrontalAngle: number|null,
 *   tibialAngleLeft: number|null,
 *   tibialAngleRight: number|null,
 *   pelvicTiltDegrees: number|null,
 *   lateralFlexionDegrees: number|null,
 *   upperTrunkAngle: number|null,
 *   spineSegmentalAngle: number|null,
 *   headForwardOffset: number|null,
 * }}
 */
export function computeJointAngles(landmarks) {
  const angles = {
    leftKneeFlexion: null,
    rightKneeFlexion: null,
    leftHipFlexion: null,
    rightHipFlexion: null,
    leftAnkleDorsiflexion: null,
    rightAnkleDorsiflexion: null,
    trunkLeanDegrees: null,
    lateralTrunkShift: null,
    leftShoulderFlexion: null,
    rightShoulderFlexion: null,
    leftElbowAngle: null,
    rightElbowAngle: null,
    leftKneeFrontalAngle: null,
    rightKneeFrontalAngle: null,
    tibialAngleLeft: null,
    tibialAngleRight: null,
    pelvicTiltDegrees: null,
    lateralFlexionDegrees: null,
    upperTrunkAngle: null,
    spineSegmentalAngle: null,
    headForwardOffset: null,
    leftFootPronation:  null,
    rightFootPronation: null,
    hipLateralShift:    null,
    shoulderTiltDegrees: null,
    heelRiseLeft:  null,
    heelRiseRight: null,
  };

  // --- Knee flexion (hip-knee-ankle) ---
  if (allVisible(landmarks, [LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE])) {
    angles.leftKneeFlexion = angleBetween(
      asArray(landmarks[LM.LEFT_HIP]),
      asArray(landmarks[LM.LEFT_KNEE]),
      asArray(landmarks[LM.LEFT_ANKLE]),
    );
  }
  if (allVisible(landmarks, [LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE])) {
    angles.rightKneeFlexion = angleBetween(
      asArray(landmarks[LM.RIGHT_HIP]),
      asArray(landmarks[LM.RIGHT_KNEE]),
      asArray(landmarks[LM.RIGHT_ANKLE]),
    );
  }

  // --- Hip flexion (shoulder-hip-knee) ---
  if (allVisible(landmarks, [LM.LEFT_SHOULDER, LM.LEFT_HIP, LM.LEFT_KNEE])) {
    angles.leftHipFlexion = angleBetween(
      asArray(landmarks[LM.LEFT_SHOULDER]),
      asArray(landmarks[LM.LEFT_HIP]),
      asArray(landmarks[LM.LEFT_KNEE]),
    );
  }
  if (allVisible(landmarks, [LM.RIGHT_SHOULDER, LM.RIGHT_HIP, LM.RIGHT_KNEE])) {
    angles.rightHipFlexion = angleBetween(
      asArray(landmarks[LM.RIGHT_SHOULDER]),
      asArray(landmarks[LM.RIGHT_HIP]),
      asArray(landmarks[LM.RIGHT_KNEE]),
    );
  }

  // --- Ankle dorsiflexion (knee-ankle-foot_index) ---
  if (allVisible(landmarks, [LM.LEFT_KNEE, LM.LEFT_ANKLE, LM.LEFT_FOOT_INDEX])) {
    angles.leftAnkleDorsiflexion = angleBetween(
      asArray(landmarks[LM.LEFT_KNEE]),
      asArray(landmarks[LM.LEFT_ANKLE]),
      asArray(landmarks[LM.LEFT_FOOT_INDEX]),
    );
  }
  if (allVisible(landmarks, [LM.RIGHT_KNEE, LM.RIGHT_ANKLE, LM.RIGHT_FOOT_INDEX])) {
    angles.rightAnkleDorsiflexion = angleBetween(
      asArray(landmarks[LM.RIGHT_KNEE]),
      asArray(landmarks[LM.RIGHT_ANKLE]),
      asArray(landmarks[LM.RIGHT_FOOT_INDEX]),
    );
  }

  // --- Trunk metrics (lean, lateral shift, lateral flexion, spine segmentation) ---
  if (
    allVisible(landmarks, [
      LM.LEFT_SHOULDER,
      LM.RIGHT_SHOULDER,
      LM.LEFT_HIP,
      LM.RIGHT_HIP,
    ])
  ) {
    const midShoulder = midpoint(
      asArray(landmarks[LM.LEFT_SHOULDER]),
      asArray(landmarks[LM.RIGHT_SHOULDER]),
    );
    const midHip = midpoint(
      asArray(landmarks[LM.LEFT_HIP]),
      asArray(landmarks[LM.RIGHT_HIP]),
    );

    // trunk_vec = mid_shoulder - mid_hip (pointing upward in image coords)
    const trunkVec = subtract(midShoulder, midHip);

    // negate because y is down: vertical_angle(-trunk_vec)
    angles.trunkLeanDegrees = verticalAngle([-trunkVec[0], -trunkVec[1]]);

    // Lateral shift: signed horizontal distance (shoulder vs hip)
    angles.lateralTrunkShift = midShoulder[0] - midHip[0];

    // Lateral flexion angle: convert horizontal offset to degrees
    const vertDist = midHip[1] - midShoulder[1]; // positive (hip is below shoulder in image coords)
    if (vertDist > 0) {
      angles.lateralFlexionDegrees =
        (Math.atan2(Math.abs(angles.lateralTrunkShift), vertDist) * 180) / Math.PI;
    }

    // --- Spine segmentation using ears ---
    if (bilateralVisible(landmarks, LM.LEFT_EAR, LM.RIGHT_EAR)) {
      const midEar = midpoint(
        asArray(landmarks[LM.LEFT_EAR]),
        asArray(landmarks[LM.RIGHT_EAR]),
      );

      // Upper trunk angle: ear→shoulder segment deviation from vertical
      // upper_vec = mid_shoulder - mid_ear  (pointing downward: ear to shoulder)
      const upperVec = subtract(midShoulder, midEar);
      angles.upperTrunkAngle = verticalAngle(upperVec);

      // Spine segmental angle: angle at mid-shoulder between ear–shoulder–hip
      angles.spineSegmentalAngle = angleBetween(midEar, midShoulder, midHip);

      // Head forward offset: horizontal ear position relative to shoulder
      angles.headForwardOffset = midEar[0] - midShoulder[0];
    }
  }

  // --- Shoulder flexion (hip-shoulder-elbow) ---
  if (allVisible(landmarks, [LM.LEFT_HIP, LM.LEFT_SHOULDER, LM.LEFT_ELBOW])) {
    angles.leftShoulderFlexion = angleBetween(
      asArray(landmarks[LM.LEFT_HIP]),
      asArray(landmarks[LM.LEFT_SHOULDER]),
      asArray(landmarks[LM.LEFT_ELBOW]),
    );
  }
  if (allVisible(landmarks, [LM.RIGHT_HIP, LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW])) {
    angles.rightShoulderFlexion = angleBetween(
      asArray(landmarks[LM.RIGHT_HIP]),
      asArray(landmarks[LM.RIGHT_SHOULDER]),
      asArray(landmarks[LM.RIGHT_ELBOW]),
    );
  }

  // --- Elbow angle (shoulder-elbow-wrist) ---
  if (allVisible(landmarks, [LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST])) {
    angles.leftElbowAngle = angleBetween(
      asArray(landmarks[LM.LEFT_SHOULDER]),
      asArray(landmarks[LM.LEFT_ELBOW]),
      asArray(landmarks[LM.LEFT_WRIST]),
    );
  }
  if (allVisible(landmarks, [LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST])) {
    angles.rightElbowAngle = angleBetween(
      asArray(landmarks[LM.RIGHT_SHOULDER]),
      asArray(landmarks[LM.RIGHT_ELBOW]),
      asArray(landmarks[LM.RIGHT_WRIST]),
    );
  }

  // --- Knee frontal deviation (valgus proxy, frontal camera) ---
  // Measures how far the knee deviates medially from the hip-ankle alignment line.
  // Positive = medial collapse (valgus); negative = lateral bow (varus).
  // Normalised by hip width so the metric is body-size independent.
  if (bilateralVisible(landmarks, LM.LEFT_HIP, LM.RIGHT_HIP)) {
    const hipWidth = Math.abs(
      landmarks[LM.LEFT_HIP].x - landmarks[LM.RIGHT_HIP].x,
    );
    if (hipWidth > 0.01) {
      const sides = [
        { side: 'left',  hipIdx: LM.LEFT_HIP,  kneeIdx: LM.LEFT_KNEE,  ankleIdx: LM.LEFT_ANKLE  },
        { side: 'right', hipIdx: LM.RIGHT_HIP, kneeIdx: LM.RIGHT_KNEE, ankleIdx: LM.RIGHT_ANKLE },
      ];
      for (const { side, hipIdx, kneeIdx, ankleIdx } of sides) {
        if (allVisible(landmarks, [hipIdx, kneeIdx, ankleIdx])) {
          const hip   = asArray(landmarks[hipIdx]);
          const knee  = asArray(landmarks[kneeIdx]);
          const ankle = asArray(landmarks[ankleIdx]);
          const dy = ankle[1] - hip[1];
          if (Math.abs(dy) > 0.01) {
            const t = (knee[1] - hip[1]) / dy;
            const expectedX = hip[0] + t * (ankle[0] - hip[0]);
            let deviation;
            if (side === 'left') {
              // Left knee is on the RIGHT side of the image (large x).
              // Valgus = knee moves toward centre = x decreases = expectedX - knee[0] > 0.
              deviation = (expectedX - knee[0]) / hipWidth;
            } else {
              // Right knee is on the LEFT side of the image (small x).
              // Valgus = knee moves toward centre = x increases = knee[0] - expectedX > 0.
              deviation = (knee[0] - expectedX) / hipWidth;
            }
            if (side === 'left') {
              angles.leftKneeFrontalAngle = deviation;
            } else {
              angles.rightKneeFrontalAngle = deviation;
            }
          }
        }
      }
    }
  }

  // --- Foot frontal deviation (pronation / supination proxy, anterior view) ---
  // Measures the horizontal deviation of the heel relative to the ankle.
  // Positive = heel medial to ankle = pronation (arch collapse / eversion).
  // Negative = heel lateral to ankle = supination (inversion).
  // Normalised by hip width for consistency with the knee valgus metric.
  if (bilateralVisible(landmarks, LM.LEFT_HIP, LM.RIGHT_HIP)) {
    const hipWidthFoot = Math.abs(
      landmarks[LM.LEFT_HIP].x - landmarks[LM.RIGHT_HIP].x,
    );
    if (hipWidthFoot > 0.01) {
      for (const { side, ankleIdx, heelIdx } of [
        { side: 'left',  ankleIdx: LM.LEFT_ANKLE,  heelIdx: LM.LEFT_HEEL  },
        { side: 'right', ankleIdx: LM.RIGHT_ANKLE, heelIdx: LM.RIGHT_HEEL },
      ]) {
        if (allVisible(landmarks, [ankleIdx, heelIdx])) {
          const ankle = asArray(landmarks[ankleIdx]);
          const heel  = asArray(landmarks[heelIdx]);
          // Left leg sits on the right side of the image (large x).
          //   Pronation = heel rolls inward = heel.x decreases = ankle.x > heel.x > 0.
          // Right leg sits on the left side (small x).
          //   Pronation = heel rolls inward = heel.x increases = heel.x > ankle.x > 0.
          const deviation = side === 'left'
            ? (ankle[0] - heel[0]) / hipWidthFoot
            : (heel[0] - ankle[0]) / hipWidthFoot;
          if (side === 'left') angles.leftFootPronation  = deviation;
          else                 angles.rightFootPronation = deviation;
        }
      }
    }
  }

  // --- Tibial angle (tibia from vertical) ---
  // Measures forward inclination of the shin — proxy for ankle dorsiflexion from lateral view.
  if (allVisible(landmarks, [LM.LEFT_KNEE, LM.LEFT_ANKLE])) {
    const tibiaLeft = subtract(
      asArray(landmarks[LM.LEFT_ANKLE]),
      asArray(landmarks[LM.LEFT_KNEE]),
    );
    angles.tibialAngleLeft = verticalAngle(tibiaLeft);
  }
  if (allVisible(landmarks, [LM.RIGHT_KNEE, LM.RIGHT_ANKLE])) {
    const tibiaRight = subtract(
      asArray(landmarks[LM.RIGHT_ANKLE]),
      asArray(landmarks[LM.RIGHT_KNEE]),
    );
    angles.tibialAngleRight = verticalAngle(tibiaRight);
  }

  // --- Hip lateral shift over base of support ---
  // Horizontal offset of hip midpoint relative to ankle midpoint, normalised by hip width.
  // Positive = hips shifted right; negative = shifted left.
  // Distinct from lateralTrunkShift (shoulder–hip) and pelvicTiltDegrees (rotation).
  if (
    bilateralVisible(landmarks, LM.LEFT_HIP,   LM.RIGHT_HIP) &&
    bilateralVisible(landmarks, LM.LEFT_ANKLE, LM.RIGHT_ANKLE)
  ) {
    const lHip = asArray(landmarks[LM.LEFT_HIP]);
    const rHip = asArray(landmarks[LM.RIGHT_HIP]);
    const lAnk = asArray(landmarks[LM.LEFT_ANKLE]);
    const rAnk = asArray(landmarks[LM.RIGHT_ANKLE]);
    const hw   = Math.abs(lHip[0] - rHip[0]);
    if (hw > 0.01) {
      const mHip   = midpoint(lHip, rHip);
      const mAnkle = midpoint(lAnk, rAnk);
      angles.hipLateralShift = (mHip[0] - mAnkle[0]) / hw;
    }
  }

  // --- Shoulder tilt (shoulder line from horizontal, anterior view) ---
  // Angle of the shoulder girdle from horizontal — distinct from lateral trunk flexion.
  // Positive = right shoulder lower; negative = left shoulder lower.
  if (bilateralVisible(landmarks, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER)) {
    const lSh  = asArray(landmarks[LM.LEFT_SHOULDER]);
    const rSh  = asArray(landmarks[LM.RIGHT_SHOULDER]);
    const horiz = Math.abs(rSh[0] - lSh[0]);
    if (horiz > 0.01) {
      const vert = rSh[1] - lSh[1]; // positive = right shoulder lower (y down)
      angles.shoulderTiltDegrees = (Math.atan2(vert, horiz) * 180) / Math.PI;
    }
  }

  // --- Heel rise (heel elevation relative to ankle, normalised by tibia length, lateral view) ---
  // Positive = heel below ankle (normal); near-zero or negative = heel rising off the floor.
  // NOTE: heel visibility is deliberately checked at a lower threshold (0.25) because from a
  // lateral view at squat depth the heel sits behind the ankle and MediaPipe confidence drops
  // below 0.5 even when the landmark position is still reliable.
  for (const { heelRiseKey, kneeIdx, ankleIdx, heelIdx } of [
    { heelRiseKey: 'heelRiseLeft',  kneeIdx: LM.LEFT_KNEE,  ankleIdx: LM.LEFT_ANKLE,  heelIdx: LM.LEFT_HEEL  },
    { heelRiseKey: 'heelRiseRight', kneeIdx: LM.RIGHT_KNEE, ankleIdx: LM.RIGHT_ANKLE, heelIdx: LM.RIGHT_HEEL },
  ]) {
    if (allVisible(landmarks, [kneeIdx, ankleIdx]) && landmarks[heelIdx].visibility > 0.25) {
      const tibiaLen = landmarks[ankleIdx].y - landmarks[kneeIdx].y; // positive (ankle is below knee in image)
      if (tibiaLen > 0.01) {
        angles[heelRiseKey] = (landmarks[heelIdx].y - landmarks[ankleIdx].y) / tibiaLen;
      }
    }
  }

  // --- Pelvic tilt (hip line from horizontal, anterior view) ---
  // Signed: positive = right hip lower; negative = left hip lower.
  if (bilateralVisible(landmarks, LM.LEFT_HIP, LM.RIGHT_HIP)) {
    const leftHip  = asArray(landmarks[LM.LEFT_HIP]);
    const rightHip = asArray(landmarks[LM.RIGHT_HIP]);
    const horiz = Math.abs(rightHip[0] - leftHip[0]);
    if (horiz > 0.01) {
      const vert = rightHip[1] - leftHip[1]; // positive = right hip lower (y down)
      angles.pelvicTiltDegrees = (Math.atan2(vert, horiz) * 180) / Math.PI;
    }
  }

  return angles;
}
