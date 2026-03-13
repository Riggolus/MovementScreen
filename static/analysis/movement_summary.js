/**
 * movement_summary.js
 *
 * Cross-assessment aggregation engine.
 * Takes a list of stored assessment objects and produces a GlobalSummary —
 * grouping individual findings into movement chains, computing trend
 * direction, and surfacing priority patterns with corrective guidance.
 */

// ---------------------------------------------------------------------------
// Grade helpers
// ---------------------------------------------------------------------------

const GRADE_ORDER = { A: 0, B: 1, C: 2, D: 3, E: 4, F: 5 };
const GRADE_LABEL = { A: 'Pass', B: 'Minimal', C: 'Mild', D: 'Moderate', E: 'Significant', F: 'Severe' };

function worseGrade(a, b) {
  return GRADE_ORDER[a] >= GRADE_ORDER[b] ? a : b;
}

function avgGradeNum(findings) {
  if (!findings.length) return 0;
  return findings.reduce((sum, f) => sum + GRADE_ORDER[f.severity], 0) / findings.length;
}

// ---------------------------------------------------------------------------
// Movement chain definitions
// ---------------------------------------------------------------------------
// Each chain maps a set of finding name substrings to an underlying dysfunction.
// A finding matches a chain if its name contains ANY of the chain's patterns.

export const MOVEMENT_CHAINS = [
  {
    id: 'ankle_foot',
    label: 'Ankle & Foot',
    icon: '🦶',
    description:
      'Ankle dorsiflexion restriction limits squat depth, disrupts gait push-off, ' +
      'and forces compensations upstream into the knee and hip.',
    patterns: ['Dorsiflexion', 'Heel Rise', 'Tibial Inclination'],
    corrections: [
      'Knee-to-wall ankle mobilisation — aim for > 10 cm from wall',
      'Eccentric heel-drop protocol off a step, 3 × 15 reps daily',
      'Gastrocnemius and soleus stretching (straight and bent knee)',
    ],
  },
  {
    id: 'knee_hip_stability',
    label: 'Knee & Hip Stability',
    icon: '🦵',
    description:
      'Medial knee collapse and pelvic drop during loading tasks signal hip ' +
      'abductor and glute weakness, raising injury risk at the knee and lower back.',
    patterns: ['Valgus', 'Pelvic Tilt', 'Swing Phase Knee'],
    corrections: [
      'Glute medius activation: clamshells, lateral band walks, side-lying abduction',
      'Single-leg stability progressions: step-ups, single-leg RDL, skater squat',
      'Squat cueing with resistance band above knees to reinforce knee tracking',
    ],
  },
  {
    id: 'hip_mobility',
    label: 'Hip Mobility',
    icon: '🔄',
    description:
      'Restricted hip flexion and bilateral asymmetry suggest hip flexor tightness ' +
      'or joint restriction, often manifesting as excessive trunk lean to compensate.',
    patterns: ['Hip Flexion', 'Forward Trunk Lean'],
    corrections: [
      'Hip flexor stretching: couch stretch, 90/90 hip mobility',
      'Hip hinge pattern: Romanian deadlift, good morning to lengthen posterior chain',
      'Thoracic extension mobility to decouple trunk lean from hip restriction',
    ],
  },
  {
    id: 'trunk_control',
    label: 'Trunk Control',
    icon: '⚖️',
    description:
      'Lateral trunk shift and rotational asymmetry reflect core stability deficits ' +
      'that compromise load transfer between the lower and upper body.',
    patterns: ['Lateral Trunk Shift', 'Lateral Flexion', 'Spinal Segmental'],
    corrections: [
      'Anti-lateral flexion work: suitcase carry, Pallof press, side plank progressions',
      'Rotary stability: bird-dog, dead-bug, contralateral reach patterns',
      'Address underlying hip restriction driving the compensatory shift',
    ],
  },
  {
    id: 'shoulder_thoracic',
    label: 'Shoulder & Thoracic',
    icon: '🙌',
    description:
      'Overhead reach limitation and shoulder asymmetry reflect thoracic stiffness ' +
      'or shoulder mobility restriction — critical for overhead pressing and throwing patterns.',
    patterns: ['Shoulder Flexion', 'Head Forward', 'Upper Trunk', 'Elbow'],
    corrections: [
      'Thoracic extension over foam roller: 60 s hold, 3 segments',
      'Lat and posterior capsule stretching: doorway pec stretch, sleeper stretch',
      'Overhead reach progressions: wall slide, lat pull-down, bar hang 30 s',
    ],
  },
  {
    id: 'gait_pattern',
    label: 'Gait Pattern',
    icon: '🚶',
    description:
      'Gait compensations reveal how upstream mobility and stability restrictions ' +
      'manifest during the demands of real-world locomotion.',
    patterns: ['Swing Phase', 'Trunk Lean (Gait)', 'Ankle Dorsiflexion (Gait)'],
    corrections: [
      'Resolve upstream restrictions flagged in squat and lunge screens first',
      'Gait re-education drills: high-knee march, A-skip, heel-toe walking',
      'Single-leg stance training to improve stance-phase stability',
    ],
  },
];

// ---------------------------------------------------------------------------
// Trend computation
// ---------------------------------------------------------------------------

/**
 * Given findings sorted oldest→newest, compare the first half vs second half
 * average grade number. Returns 'improving', 'declining', or 'stable'.
 * Returns null when there are fewer than 4 data points.
 */
function computeTrend(sortedFindings) {
  if (sortedFindings.length < 4) return null;
  const mid = Math.floor(sortedFindings.length / 2);
  const older = sortedFindings.slice(0, mid);
  const newer = sortedFindings.slice(mid);
  const diff = avgGradeNum(newer) - avgGradeNum(older);
  if (diff < -0.4) return 'improving';
  if (diff >  0.4) return 'declining';
  return 'stable';
}

// ---------------------------------------------------------------------------
// Screen-type display labels
// ---------------------------------------------------------------------------
const SCREEN_LABEL = { squat: 'Squat', lunge: 'Lunge', overhead: 'Overhead', gait: 'Gait' };

// ---------------------------------------------------------------------------
// Main export
// ---------------------------------------------------------------------------

/**
 * Compute a GlobalSummary from an array of stored assessment objects.
 *
 * @param {object[]} assessments  - newest-first from IndexedDB
 * @returns {object|null}
 */
export function computeGlobalSummary(assessments) {
  if (!assessments || assessments.length === 0) return null;

  // Flatten all findings, enriched with assessment metadata
  const allFindings = [];
  for (const a of assessments) {
    for (const f of (a.findings || [])) {
      allFindings.push({
        name:        f.name,
        severity:    f.severity,
        screen_type: a.screen_type,
        camera_angle: a.camera_angle,
        recorded_at: a.recorded_at,
      });
    }
  }

  // Sort oldest→newest for trend calc
  const byDate = [...allFindings].sort(
    (a, b) => new Date(a.recorded_at) - new Date(b.recorded_at),
  );

  // Evaluate each chain
  const chains = MOVEMENT_CHAINS.map(chain => {
    const matched = allFindings.filter(f =>
      chain.patterns.some(p => f.name.includes(p)),
    );

    if (matched.length === 0) {
      return { ...chain, grade: 'A', confirmedIn: [], trend: null, priority: 0 };
    }

    // Worst grade observed across all matched findings
    const grade = matched.reduce(
      (worst, f) => worseGrade(worst, f.severity), 'A',
    );

    // Unique screen types that confirmed this chain
    const confirmedIn = [...new Set(matched.map(f => SCREEN_LABEL[f.screen_type] || f.screen_type))];

    // Trend: use time-ordered subset
    const trend = computeTrend(
      byDate.filter(f => chain.patterns.some(p => f.name.includes(p))),
    );

    // Priority: grade severity × confirmation breadth (cross-assessment credibility)
    const priority = GRADE_ORDER[grade] * 10 + confirmedIn.length;

    return { ...chain, grade, confirmedIn, trend, priority };
  });

  // Sort by priority (worst + most confirmed first); passing chains go last
  chains.sort((a, b) => {
    if (a.grade === 'A' && b.grade !== 'A') return 1;
    if (b.grade === 'A' && a.grade !== 'A') return -1;
    return b.priority - a.priority;
  });

  // Overall grade = worst active chain
  const activeChains = chains.filter(c => c.grade !== 'A');
  const overallGrade = activeChains.length
    ? activeChains.reduce((w, c) => worseGrade(w, c.grade), 'A')
    : 'A';

  // Overall trend
  const trends = chains.map(c => c.trend).filter(Boolean);
  const nImproving = trends.filter(t => t === 'improving').length;
  const nDeclining = trends.filter(t => t === 'declining').length;
  const overallTrend =
    nImproving > nDeclining ? 'improving' :
    nDeclining > nImproving ? 'declining' : 'stable';

  // Date range
  const dates = assessments.map(a => new Date(a.recorded_at));
  const earliest = new Date(Math.min(...dates));
  const latest   = new Date(Math.max(...dates));

  return {
    overallGrade,
    overallTrend,
    assessmentCount: assessments.length,
    activeChainCount: activeChains.length,
    dateRange: { from: earliest, to: latest },
    chains,
  };
}
