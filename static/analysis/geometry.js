/**
 * geometry.js
 * Geometric helpers for joint-angle and positional calculations.
 * Direct port of movementscreen/utils/geometry.py
 *
 * All functions operate on plain 2-element arrays [x, y].
 */

/**
 * Element-wise subtraction: a - b.
 * @param {number[]} a - [x, y]
 * @param {number[]} b - [x, y]
 * @returns {number[]}
 */
export function subtract(a, b) {
  return [a[0] - b[0], a[1] - b[1]];
}

/**
 * Negate a 2-element array.
 * @param {number[]} v - [x, y]
 * @returns {number[]}
 */
export function negate(v) {
  return [-v[0], -v[1]];
}

/**
 * Return the angle (degrees) at `vertex` formed by vectors vertex→a and vertex→b.
 * Returns 0.0 if either vector has zero length.
 *
 * Port of: angle_between(a, vertex, b) in geometry.py
 *
 * @param {number[]} a      - [x, y]
 * @param {number[]} vertex - [x, y]
 * @param {number[]} b      - [x, y]
 * @returns {number} angle in degrees
 */
export function angleBetween(a, vertex, b) {
  const va = subtract(a, vertex);
  const vb = subtract(b, vertex);

  const normA = Math.sqrt(va[0] * va[0] + va[1] * va[1]);
  const normB = Math.sqrt(vb[0] * vb[0] + vb[1] * vb[1]);

  if (normA === 0 || normB === 0) {
    return 0.0;
  }

  const dot = va[0] * vb[0] + va[1] * vb[1];
  const cosTheta = Math.min(1.0, Math.max(-1.0, dot / (normA * normB)));
  return (Math.acos(cosTheta) * 180) / Math.PI;
}

/**
 * Angle (degrees) between vector v and the downward vertical [0, 1] in 2-D.
 * Returns 0.0 if v is a zero vector.
 *
 * Port of: vertical_angle(v) in geometry.py
 * The Python implementation computes the angle against [0, 1] via dot product,
 * NOT arctan2(|x|, |y|).
 *
 * @param {number[]} v - [x, y]
 * @returns {number} angle in degrees
 */
export function verticalAngle(v) {
  // down = [0, 1]
  const norm = Math.sqrt(v[0] * v[0] + v[1] * v[1]);
  if (norm === 0) {
    return 0.0;
  }
  // dot(v, [0,1]) == v[1]
  const cosTheta = Math.min(1.0, Math.max(-1.0, v[1] / norm));
  return (Math.acos(cosTheta) * 180) / Math.PI;
}

/**
 * Return the midpoint (centroid) of two 2-element arrays.
 *
 * Port of: midpoint(*points) in geometry.py (two-point form).
 *
 * @param {number[]} a - [x, y]
 * @param {number[]} b - [x, y]
 * @returns {number[]}
 */
export function midpoint(a, b) {
  return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
}

/**
 * Asymmetry ratio in [0, 1] where 0 is perfect symmetry.
 * Formula: |left - right| / max(|left|, |right|, 1e-6)
 *
 * Port of: asymmetry_ratio(left, right) in geometry.py
 * NOTE: the Python source uses max(|left|, |right|, 1e-6), not the average.
 *
 * @param {number} left
 * @param {number} right
 * @returns {number}
 */
export function asymmetryRatio(left, right) {
  const denom = Math.max(Math.abs(left), Math.abs(right), 1e-6);
  return Math.abs(left - right) / denom;
}
