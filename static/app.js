'use strict';

import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs';

import { computeJointAngles, LM } from './analysis/joint_angles.js';
import { createAggregator }   from './analysis/aggregator.js';
import {
  acceptFrameSquat,
  acceptFrameLunge,
  acceptFrameOverhead,
} from './analysis/screens.js';
import {
  getThresholds,
  saveThresholdOverrides,
  clearThresholdOverride,
  DEFAULT_THRESHOLDS,
  THRESHOLD_DESCRIPTIONS,
} from './analysis/thresholds.js';
import {
  saveAssessment,
  getAssessments,
  getAssessment,
} from './db/local_db.js';

// ── MediaPipe ────────────────────────────────────────────
let poseLandmarker = null;
let drawingUtils   = null;
let animFrameId    = null;

async function initPoseLandmarker() {
  if (poseLandmarker) return;
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
  );
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numPoses: 1,
  });
}
initPoseLandmarker().catch(() => {});

// ── Auth state ───────────────────────────────────────────

// ── App state ─────────────────────────────────────────────
let currentScreen      = 'squat';
let currentSide        = 'left';
let currentAngle       = 'anterior';
let currentLateralSide = 'left'; // which leg is closest to the camera in lateral view
let facingMode     = 'environment';
let mediaStream    = null;
let isRecording        = false;
let isPositioning      = false;
let positionGoodFrames = 0;
const POSITION_HOLD_FRAMES = 45; // ~1.5 s at 30 fps

// Auto-stop: if body fills >92% of frame for 20 consecutive frames after
// at least 60 recording frames (~2 s), the patient walked toward the camera.
let recordingFrameCount = 0;
let walkTowardFrames    = 0;
const WALK_TOWARD_THRESHOLD = 0.92;
const WALK_TOWARD_FRAMES    = 20;
const MIN_FRAMES_BEFORE_AUTOSTOP = 60;

let aggregator     = null;
let timerInterval  = null;
let secondsElapsed = 0;

// ── DOM refs ─────────────────────────────────────────────
const views = {
  auth:       document.getElementById('view-auth'),
  setup:      document.getElementById('view-setup'),
  recording:  document.getElementById('view-recording'),
  processing: document.getElementById('view-processing'),
  results:    document.getElementById('view-results'),
  history:    document.getElementById('view-history'),
  admin:      document.getElementById('view-admin'),
  report:     document.getElementById('view-report'),
  error:      document.getElementById('view-error'),
};
const preview        = document.getElementById('preview');
const skeletonCanvas = document.getElementById('skeleton-canvas');
const skeletonCtx    = skeletonCanvas.getContext('2d');
const timerEl        = document.getElementById('timer');
const lungeOptions        = document.getElementById('lunge-options');
const lateralSideOptions  = document.getElementById('lateral-side-options');
const poseStatus     = document.getElementById('pose-status');
const headerNav      = document.getElementById('header-nav');

// ── View helpers ─────────────────────────────────────────
function showView(name) {
  Object.values(views).forEach(v => v.classList.remove('active'));
  views[name].classList.add('active');
}

function updateHeader() {
  headerNav.innerHTML = `
    <button class="nav-btn" id="nav-home-btn">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>
      <span>Home</span>
    </button>
    <button class="nav-btn" id="nav-history-btn">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
      <span>History</span>
    </button>
    <button class="nav-btn" id="nav-admin-btn">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M4.93 4.93a10 10 0 0 0 0 14.14"/></svg>
      <span>Thresholds</span>
    </button>
  `;
  document.getElementById('nav-home-btn').addEventListener('click', () => showView('setup'));
  document.getElementById('nav-history-btn').addEventListener('click', loadHistory);
  document.getElementById('nav-admin-btn').addEventListener('click', loadAdminPage);
}

// ── Assessment instructions ───────────────────────────────
const SCREEN_INSTRUCTIONS = {
  squat: {
    title: 'Bodyweight Squat',
    desc: 'Stand with feet shoulder-width apart, toes slightly turned out. Push your hips back and bend your knees to lower as far as comfortable, keeping your heels flat throughout.',
    cues: [
      'Feet shoulder-width apart, toes 15–30° outward',
      'Keep heels flat — do not let them rise',
      'Drive knees out in line with your toes',
      'Lower until thighs are parallel (or as deep as comfortable)',
      'Perform 3–5 slow, controlled reps',
    ],
    camera: 'Anterior view: best for knee valgus and trunk shift. Lateral view: best for ankle mobility and trunk lean.',
  },
  lunge: {
    title: 'Forward Lunge',
    desc: (side) => `Step forward with your <strong>${side}</strong> foot into a lunge. Lower your rear knee toward the ground, keeping your front shin as vertical as possible and your torso upright.`,
    cues: (side) => [
      `Step forward with your ${side} foot`,
      'Keep your front shin vertical — knee over ankle',
      'Lower your rear knee toward (not onto) the floor',
      'Keep your torso upright — avoid leaning forward',
      'Push back to start and perform 3–5 controlled reps',
    ],
    camera: 'Anterior view recommended for knee tracking and trunk shift.',
  },
  overhead: {
    title: 'Overhead Reach',
    desc: 'Stand tall and raise both arms directly overhead as high as possible. Keep your core braced and avoid arching your lower back. Lower with control and repeat.',
    cues: [
      'Stand with feet hip-width apart',
      'Raise both arms simultaneously, reaching as high as possible',
      'Keep your lower back neutral — do not arch or flare your ribs',
      'Aim to get arms fully vertical beside your ears',
      'Perform 3–5 slow, controlled reps',
    ],
    camera: 'Anterior view: captures shoulder asymmetry. Lateral view: captures trunk extension compensation.',
  },
};

function renderInstructions(screen, side) {
  const card  = document.getElementById('instructions-card');
  const instr = SCREEN_INSTRUCTIONS[screen];
  if (!instr) { card.innerHTML = ''; return; }

  const desc = typeof instr.desc === 'function' ? instr.desc(side) : instr.desc;
  const cues = typeof instr.cues === 'function' ? instr.cues(side) : instr.cues;

  card.innerHTML = `
    <h3 class="instructions-title">${instr.title}</h3>
    <p class="instructions-desc">${desc}</p>
    <ul class="instructions-cues">${cues.map(c => `<li>${c}</li>`).join('')}</ul>
    <p class="instructions-camera">📐 ${instr.camera}</p>
  `;
}

// ── Screen / angle / side selectors ──────────────────────
document.querySelectorAll('.screen-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.screen-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentScreen = btn.dataset.screen;
    lungeOptions.classList.toggle('hidden', currentScreen !== 'lunge');
    renderInstructions(currentScreen, currentSide);
  });
});

document.querySelectorAll('.angle-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.angle-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentAngle = btn.dataset.angle;
    lateralSideOptions.classList.toggle('hidden', currentAngle !== 'lateral');
  });
});

document.querySelectorAll('.lunge-options .toggle-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.lunge-options .toggle-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentSide = btn.dataset.side;
    renderInstructions(currentScreen, currentSide);
  });
});

document.querySelectorAll('.lateral-side-options .toggle-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.lateral-side-options .toggle-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentLateralSide = btn.dataset.side;
  });
});

// Initialise instructions on load
renderInstructions('squat', 'left');

// ── Position check ────────────────────────────────────────
// Key landmark indices needed per camera angle
const FRONTAL_KEY_LMS = [
  LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER,
  LM.LEFT_HIP,      LM.RIGHT_HIP,
  LM.LEFT_KNEE,     LM.RIGHT_KNEE,
  LM.LEFT_ANKLE,    LM.RIGHT_ANKLE,
];

/**
 * Check whether the person is optimally positioned in frame.
 * Returns null when position is good, or a guidance string otherwise.
 */
function checkPosition(lms) {
  const isLateral = currentAngle === 'lateral';

  // Determine which landmarks must be visible
  const required = isLateral
    ? (currentLateralSide === 'right'
        ? [LM.RIGHT_SHOULDER, LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE]
        : [LM.LEFT_SHOULDER,  LM.LEFT_HIP,  LM.LEFT_KNEE,  LM.LEFT_ANKLE])
    : FRONTAL_KEY_LMS;

  if (!required.every(i => lms[i].visibility > 0.5)) return 'Step fully into frame';

  // Body height as a fraction of frame height (shoulder top → ankle bottom)
  const shoulderYs = [LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER]
    .filter(i => lms[i].visibility > 0.5).map(i => lms[i].y);
  const ankleYs = [LM.LEFT_ANKLE, LM.RIGHT_ANKLE]
    .filter(i => lms[i].visibility > 0.5).map(i => lms[i].y);
  if (!shoulderYs.length || !ankleYs.length) return 'Step fully into frame';

  const bodyH = Math.max(...ankleYs) - Math.min(...shoulderYs);
  if (bodyH < 0.50) return 'Move closer';
  if (bodyH > 0.88) return 'Move further away';

  // Horizontal centering (frontal views only)
  if (!isLateral) {
    const visShoulders = [LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER].filter(i => lms[i].visibility > 0.5);
    const midX = visShoulders.reduce((s, i) => s + lms[i].x, 0) / visShoulders.length;
    if (midX < 0.35) return 'Step to the right';
    if (midX > 0.65) return 'Step to the left';
  }

  return null; // good
}

// ── Recording ─────────────────────────────────────────────
document.getElementById('start-btn').addEventListener('click', startRecording);

async function startRecording() {
  await initPoseLandmarker().catch(() => {});
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
  } catch {
    try { mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false }); }
    catch { showError('Camera access denied. Please allow camera permission and reload.'); return; }
  }

  preview.srcObject = mediaStream;
  showView('recording');

  const mirrored = facingMode === 'user';
  preview.style.transform        = mirrored ? 'scaleX(-1)' : '';
  skeletonCanvas.style.transform = mirrored ? 'scaleX(-1)' : '';

  const SCREEN_NAMES = { squat: 'Bodyweight Squat', lunge: 'Forward Lunge', overhead: 'Overhead Reach' };
  aggregator = createAggregator(SCREEN_NAMES[currentScreen] || currentScreen);

  isPositioning = true;
  positionGoodFrames = 0;
  setPositionOverlay('Waiting for pose…', 0);

  if (poseLandmarker) {
    drawingUtils = new DrawingUtils(skeletonCtx);
    startSkeletonLoop();
  }
}

function getBodyH(lms) {
  const shoulderYs = [LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER]
    .filter(i => lms[i].visibility > 0.5).map(i => lms[i].y);
  const ankleYs = [LM.LEFT_ANKLE, LM.RIGHT_ANKLE]
    .filter(i => lms[i].visibility > 0.5).map(i => lms[i].y);
  if (!shoulderYs.length || !ankleYs.length) return 0;
  return Math.max(...ankleYs) - Math.min(...shoulderYs);
}

function beginRecording() {
  isPositioning = false;
  recordingFrameCount = 0;
  walkTowardFrames = 0;
  document.getElementById('position-overlay').classList.add('hidden');
  isRecording = true;
  startTimer();
}

document.getElementById('position-skip-btn').addEventListener('click', beginRecording);

function setPositionOverlay(msg, progress) {
  document.getElementById('position-overlay').classList.remove('hidden');
  document.getElementById('position-message').textContent = msg;
  document.getElementById('position-bar').style.width = `${Math.round(progress * 100)}%`;
  document.getElementById('position-hint').style.opacity = progress > 0 ? '1' : '0';
}

// ── Skeleton loop ─────────────────────────────────────────
function startSkeletonLoop() {
  let lastTs = -1;
  function loop() {
    if (!poseLandmarker) return;
    const { offsetWidth: w, offsetHeight: h } = skeletonCanvas;
    if (skeletonCanvas.width !== w || skeletonCanvas.height !== h) {
      skeletonCanvas.width = w; skeletonCanvas.height = h;
    }
    skeletonCtx.clearRect(0, 0, w, h);
    const now = performance.now();
    if (now !== lastTs && preview.readyState >= 2) {
      lastTs = now;
      try {
        const result = poseLandmarker.detectForVideo(preview, now);
        if (result.landmarks?.length > 0) {
          const lms = result.landmarks[0];

          // Skeleton colour: green when position is good, indigo otherwise
          const skelColor = (isPositioning && positionGoodFrames > 0)
            ? 'rgba(16,185,129,.85)' : 'rgba(99,102,241,.85)';
          drawingUtils.drawConnectors(lms, PoseLandmarker.POSE_CONNECTIONS, {
            color: skelColor, lineWidth: 2.5,
          });
          drawingUtils.drawLandmarks(lms, {
            color: '#ffffff', fillColor: isPositioning && positionGoodFrames > 0
              ? 'rgba(16,185,129,.7)' : 'rgba(99,102,241,.7)',
            lineWidth: 1, radius: 4,
          });

          poseStatus.textContent = 'Pose detected';
          poseStatus.classList.add('detected');

          if (isPositioning) {
            const guidance = checkPosition(lms);
            if (guidance === null) {
              positionGoodFrames++;
              const pct = positionGoodFrames / POSITION_HOLD_FRAMES;
              setPositionOverlay('Perfect — hold still', Math.min(pct, 1));
              if (positionGoodFrames >= POSITION_HOLD_FRAMES) beginRecording();
            } else {
              positionGoodFrames = 0;
              setPositionOverlay(guidance, 0);
            }
          } else if (isRecording && aggregator) {
            recordingFrameCount++;

            // Auto-stop: patient walked toward camera after exercise
            const bodyH = getBodyH(lms);
            if (recordingFrameCount > MIN_FRAMES_BEFORE_AUTOSTOP && bodyH > WALK_TOWARD_THRESHOLD) {
              walkTowardFrames++;
              if (walkTowardFrames >= WALK_TOWARD_FRAMES) {
                stopRecording();
                return;
              }
            } else {
              walkTowardFrames = 0;
            }

            let atDepth = false;
            if (currentScreen === 'squat')         atDepth = acceptFrameSquat(lms, currentAngle, currentLateralSide);
            else if (currentScreen === 'lunge')    atDepth = acceptFrameLunge(lms, currentAngle, currentSide);
            else if (currentScreen === 'overhead') atDepth = acceptFrameOverhead(lms);
            if (atDepth) aggregator.addFrame(computeJointAngles(lms));
          }
        } else {
          poseStatus.textContent = 'Waiting for pose…';
          poseStatus.classList.remove('detected');
          if (isPositioning) {
            positionGoodFrames = 0;
            setPositionOverlay('Step fully into frame', 0);
          }
        }
      } catch { /* skip frame */ }
    }
    animFrameId = requestAnimationFrame(loop);
  }
  animFrameId = requestAnimationFrame(loop);
}

function stopSkeletonLoop() {
  if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }
  skeletonCtx.clearRect(0, 0, skeletonCanvas.width, skeletonCanvas.height);
}

document.getElementById('stop-btn').addEventListener('click', stopRecording);

function stopRecording() {
  stopTimer(); stopSkeletonLoop();
  isRecording = false;
  mediaStream.getTracks().forEach(t => t.stop());
  showView('processing');
  analyseLocally();
}

document.getElementById('cancel-btn').addEventListener('click', () => {
  isPositioning = false; positionGoodFrames = 0;
  recordingFrameCount = 0; walkTowardFrames = 0;
  stopTimer(); stopSkeletonLoop();
  isRecording = false;
  mediaStream?.getTracks().forEach(t => t.stop());
  showView('setup');
});

document.getElementById('flip-btn').addEventListener('click', async () => {
  facingMode = facingMode === 'environment' ? 'user' : 'environment';
  stopSkeletonLoop();
  mediaStream?.getTracks().forEach(t => t.stop());
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false,
    });
    preview.srcObject = mediaStream;
    const mirrored = facingMode === 'user';
    preview.style.transform        = mirrored ? 'scaleX(-1)' : '';
    skeletonCanvas.style.transform = mirrored ? 'scaleX(-1)' : '';
    if (poseLandmarker) startSkeletonLoop();
  } catch { /* keep current */ }
});

// ── Timer ─────────────────────────────────────────────────
function startTimer() {
  secondsElapsed = 0; updateTimerDisplay();
  timerInterval = setInterval(() => { secondsElapsed++; updateTimerDisplay(); }, 1000);
}
function stopTimer() { clearInterval(timerInterval); timerInterval = null; }
function updateTimerDisplay() {
  const m = Math.floor(secondsElapsed / 60), s = secondsElapsed % 60;
  timerEl.textContent = `${m}:${String(s).padStart(2, '0')}`;
}

// ── Local analysis + sync ─────────────────────────────────
async function analyseLocally() {
  if (!aggregator) { showError('No recording data found. Please try again.'); return; }
  try {
    const lateralSide = (currentAngle === 'lateral' && currentScreen === 'squat') ? currentLateralSide : null;
    const result = aggregator.finalize(currentAngle, currentScreen, getThresholds(), lateralSide);

    // Warn if very few depth frames were captured (likely didn't reach depth)
    if (result.frame_count < 5) {
      result.depth_warning = true;
    }

    const record = {
      ...result,
      lead_side:   currentScreen === 'lunge' ? currentSide : null,
      recorded_at: new Date().toISOString(),
      synced:      false,
      server_id:   null,
    };

    const localId = await saveAssessment(record);
    result.recorded_at = record.recorded_at;
    result.saved = true;

    renderResults(result);
    showView('results');

  } catch (err) {
    showError(err.message);
  }
}

// ── Results rendering ─────────────────────────────────────
const SEV_COLOR  = {
  A: 'var(--grade-a)', B: 'var(--grade-b)', C: 'var(--grade-c)',
  D: 'var(--grade-d)', E: 'var(--grade-e)', F: 'var(--grade-f)',
  // legacy keys kept for cached/old API responses
  none: 'var(--grade-a)', mild: 'var(--grade-c)', moderate: 'var(--grade-d)', severe: 'var(--grade-f)',
};
const SEV_LABEL  = {
  A: 'Pass', B: 'Minimal', C: 'Mild', D: 'Moderate', E: 'Significant', F: 'Severe',
  none: 'Pass', mild: 'Mild', moderate: 'Moderate', severe: 'Severe',
};
const SEV_EMOJI  = {
  A: 'A', B: 'B', C: 'C', D: 'D', E: 'E', F: 'F',
  none: 'A', mild: 'C', moderate: 'D', severe: 'F',
};
const ANGLE_LABEL = { anterior: 'Anterior', lateral: 'Lateral', posterior: 'Posterior' };

function renderResults(data) {
  const sev = data.worst_severity, color = SEV_COLOR[sev];
  const summary = generateSummary(data);

  // ── Header ──────────────────────────────────────────────
  let html = `
    <div class="results-header">
      <div class="results-grade-ring" style="border-color:${color};color:${color}">${SEV_EMOJI[sev]}</div>
      <h1>${data.screen_name}</h1>
      <p class="results-meta">${data.frame_count} frames · ${ANGLE_LABEL[data.camera_angle] ?? ''} view</p>
      <span class="overall-pill" style="background:${color}">${SEV_LABEL[sev]}</span>
      ${data.saved ? `<p class="saved-badge">✓ Saved to your history</p>` : ''}
    </div>
    <div class="results-body">
  `;

  // ── Depth warning ─────────────────────────────────────────
  if (data.depth_warning) {
    html += `
      <div class="depth-warning">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
        <span>Too few frames captured at depth — try squatting lower or recording a longer set for accurate results.</span>
      </div>
    `;
  }

  // ── Summary ──────────────────────────────────────────────
  html += `
    <div class="results-summary-card">
      <p class="results-summary-text">${summary}</p>
    </div>
  `;

  // ── Findings ─────────────────────────────────────────────
  if (data.findings.length === 0) {
    html += `<div class="no-findings"><span class="no-findings-icon">✓</span><p>No compensations detected — great movement quality!</p></div>`;
  } else {
    html += `<h2 class="section-title">Findings &amp; Corrections</h2>`;
    for (const f of data.findings) {
      const c = SEV_COLOR[f.severity];
      const rec = getRecommendationInfo(f.name);
      const tipsHtml = rec
        ? `<ul class="finding-tips">${rec.tips.map(t => `<li>${t}</li>`).join('')}</ul>`
        : '';
      const whatHtml = rec
        ? `<p class="finding-what"><strong>What it means:</strong> ${rec.means}</p>`
        : '';
      html += `
        <div class="finding-card" style="--border-color:${c}">
          <div class="finding-header">
            <span class="severity-badge" style="background:${c}">${SEV_LABEL[f.severity]}</span>
            <span class="finding-name">${f.name}</span>
          </div>
          <p class="finding-desc">${f.description}</p>
          ${whatHtml}
          ${tipsHtml}
        </div>
      `;
    }
  }

  // ── Biomechanical Measurements (collapsible) ──────────────
  const NORMALIZED_FIELDS = new Set(['left_knee_frontal_angle', 'right_knee_frontal_angle', 'lateral_trunk_shift', 'head_forward_offset']);
  let statsHtml = '';
  if (data.stats.length > 0) {
    statsHtml += `<div class="stats-grid" style="margin-bottom:20px">`;
    for (const s of data.stats) {
      const unit = NORMALIZED_FIELDS.has(s.field) ? '' : '°';
      statsHtml += `
        <div class="stat-card">
          <div class="stat-name">${s.name}</div>
          <div class="stat-values">
            <div class="stat-item"><span class="stat-label">Min</span><span class="stat-value">${s.min}${unit}</span></div>
            <div class="stat-item main"><span class="stat-label">Mean</span><span class="stat-value">${s.mean}${unit}</span></div>
            <div class="stat-item"><span class="stat-label">Max</span><span class="stat-value">${s.max}${unit}</span></div>
          </div>
        </div>
      `;
    }
    statsHtml += `</div>`;
  }

  html += `
    <details class="bio-details">
      <summary class="bio-summary">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
        Biomechanical Measurements
        <svg class="bio-chevron" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><polyline points="6 9 12 15 18 9"/></svg>
      </summary>
      <div class="bio-body">
        <p class="bio-intro">Raw joint angle data captured during the recording. Min/Mean/Max across all depth frames.</p>
        ${statsHtml}
        <div class="calib-section">
          <p class="calib-intro">
            Use your actual joint angles below to fine-tune detection thresholds for your body.
          </p>
          <div id="calibration-panel"></div>
        </div>
      </div>
    </details>
  `;

  // ── Actions ───────────────────────────────────────────────
  html += `
      <div class="results-actions">
        <button class="btn-primary" id="again-btn">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-3.45"/></svg>
          Record Again
        </button>
        <button class="btn-primary" id="report-from-results-btn" style="background:var(--surface-2);color:var(--text);border:1px solid var(--border);box-shadow:none;">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
          View Report
        </button>
        <button class="btn-primary" id="history-from-results-btn" style="background:var(--surface-2);color:var(--text);border:1px solid var(--border);box-shadow:none;">History</button>
      </div>
    </div>
  `;

  views.results.innerHTML = html;
  views.results.scrollTop = 0;
  document.getElementById('again-btn').addEventListener('click', resetApp);
  document.getElementById('report-from-results-btn').addEventListener('click', () => renderReport(data, 'results'));
  document.getElementById('history-from-results-btn').addEventListener('click', loadHistory);
  renderCalibrationPanel(data);
}

// ── History / Progress ────────────────────────────────────
const SCREEN_EMOJI = { squat: '🏋️', lunge: '🦵', overhead: '🙌' };
const SCREEN_COLORS = { squat: '#6366f1', lunge: '#8b5cf6', overhead: '#0ea5e9' };

async function loadHistory() {
  showView('history');
  views.history.innerHTML = `<div class="processing-content"><div class="spinner-ring"></div></div>`;

  try {
    const assessments = await getAssessments(50);

    // Build by_screen progress (oldest-first for the trend chart)
    const by_screen = { squat: [], lunge: [], overhead: [] };
    for (const a of [...assessments].reverse()) {
      if (by_screen[a.screen_type]) {
        by_screen[a.screen_type].push({ recorded_at: a.recorded_at, worst_severity: a.worst_severity });
      }
    }

    renderHistory(assessments, by_screen);

  } catch (err) {
    views.history.innerHTML = `<div class="error-content"><div class="error-icon">⚠</div><p>${err.message}</p></div>`;
  }
}

function renderHistory(assessments, byScreen) {
  let html = `
    <div class="history-header">
      <h1>Your Progress</h1>
      <p>Track how your movement quality changes over time</p>
    </div>
  `;

  // Trend chart
  const hasData = Object.values(byScreen).some(pts => pts.length > 0);
  html += `<div class="card chart-card"><h2 class="card-title">Severity Trend</h2>`;
  if (hasData) {
    html += `<div id="trend-chart-container" class="trend-chart"></div>`;
    html += `<div class="trend-legend">`;
    for (const [screen, color] of Object.entries(SCREEN_COLORS)) {
      if (byScreen[screen]?.length) {
        html += `<div class="legend-item"><div class="legend-dot" style="background:${color}"></div>${screen.charAt(0).toUpperCase() + screen.slice(1)}</div>`;
      }
    }
    html += `</div>`;
  } else {
    html += `<p class="empty-state">No assessments yet. Record your first movement screen to see your trend.</p>`;
  }
  html += `</div>`;

  // Assessment list
  html += `<h2 class="section-title">${assessments.length} Assessment${assessments.length !== 1 ? 's' : ''}</h2>`;

  if (assessments.length === 0) {
    html += `<p class="empty-state">No assessments recorded yet.</p>`;
  } else {
    html += `<div class="assessments-list">`;
    for (const a of assessments) {
      const color = SEV_COLOR[a.worst_severity];
      const date  = new Date(a.recorded_at).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' });
      const screenName = a.screen_type.charAt(0).toUpperCase() + a.screen_type.slice(1);
      html += `
        <div class="assessment-card">
          <div class="assessment-card-header" data-id="${a.id}">
            <div class="assessment-screen-badge">${SCREEN_EMOJI[a.screen_type] ?? '📋'}</div>
            <div class="assessment-info">
              <div class="assessment-title">${screenName}${a.screen_type === 'lunge' && a.lead_side ? ` (${a.lead_side})` : ''} · ${ANGLE_LABEL[a.camera_angle] ?? a.camera_angle}</div>
              <div class="assessment-date">${date}</div>
            </div>
            <span class="assessment-sev-pill" style="background:${color}">${SEV_LABEL[a.worst_severity]}</span>
          </div>
          <div class="assessment-body" id="body-${a.id}"></div>
        </div>
      `;
    }
    html += `</div>`;
  }

  html += `
    <div class="again-btn-wrap">
      <button class="btn-primary" id="new-assessment-btn">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
        New Assessment
      </button>
    </div>
  `;

  views.history.innerHTML = html;
  document.getElementById('new-assessment-btn').addEventListener('click', resetApp);

  // Draw chart
  if (hasData) {
    const container = document.getElementById('trend-chart-container');
    drawTrendChart(container, byScreen);
  }

  // Expandable assessment cards
  document.querySelectorAll('.assessment-card-header').forEach(header => {
    header.addEventListener('click', () => toggleAssessmentDetail(header.dataset.id));
  });
}

async function toggleAssessmentDetail(id) {
  const body = document.getElementById(`body-${id}`);
  if (body.classList.contains('open')) {
    body.classList.remove('open');
    return;
  }
  if (body.innerHTML) { body.classList.add('open'); return; }

  body.innerHTML = `<div style="padding:16px;color:var(--text-3);font-size:13px">Loading…</div>`;
  body.classList.add('open');

  try {
    const data = await getAssessment(parseInt(id, 10));
    if (!data) {
      body.innerHTML = `<p style="padding:16px;color:var(--text-3);font-size:13px">Assessment not found.</p>`;
      return;
    }
    let inner = '';
    if (data.findings.length === 0) {
      inner = `<div class="no-findings" style="margin-top:10px"><span class="no-findings-icon" style="font-size:24px">✓</span><p>No compensations detected.</p></div>`;
    } else {
      for (const f of data.findings) {
        const c = SEV_COLOR[f.severity];
        inner += `
          <div class="finding-card" style="--border-color:${c}">
            <div class="finding-header">
              <span class="severity-badge" style="background:${c}">${SEV_LABEL[f.severity]}</span>
              <span class="finding-name">${f.name}</span>
            </div>
            <p class="finding-desc">${f.description}</p>
            ${f.metric_value != null ? `<p class="finding-metric">${f.metric_label}: <strong>${f.metric_value}</strong></p>` : ''}
          </div>
        `;
      }
    }
    inner += `
      <div style="padding:10px 0 4px">
        <button class="btn-ghost report-btn-history" style="font-size:13px;display:flex;align-items:center;gap:6px" data-assessment-id="${id}">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
          View Full Report
        </button>
      </div>
    `;
    body.innerHTML = inner;
    body.querySelector('.report-btn-history').addEventListener('click', () => renderReport(data, 'history'));
  } catch {
    body.innerHTML = `<p style="padding:16px;color:var(--severe);font-size:13px">Failed to load details.</p>`;
  }
}

// ── Trend chart (SVG) ─────────────────────────────────────
function drawTrendChart(container, byScreen) {
  const W   = container.clientWidth || 320;
  const H   = 160;
  const PAD = { top: 12, right: 16, bottom: 28, left: 44 };
  const SEV_NUM = { A: 0, B: 0.5, C: 1, D: 2, E: 2.5, F: 3, none: 0, mild: 1, moderate: 2, severe: 3 };
  const YLABELS = ['Pass', 'Mild', 'Mod', 'Severe'];

  const allPoints = Object.values(byScreen).flat();
  if (!allPoints.length) return;

  const dates = allPoints.map(p => new Date(p.recorded_at).getTime());
  const minT = Math.min(...dates), maxT = Math.max(...dates);
  const cW = W - PAD.left - PAD.right;
  const cH = H - PAD.top - PAD.bottom;

  const xS = t => PAD.left + (minT === maxT ? cW / 2 : ((t - minT) / (maxT - minT)) * cW);
  const yS = s => PAD.top + cH - (SEV_NUM[s] / 3) * cH;

  let svg = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;overflow:visible">`;

  // Grid + Y labels
  for (let i = 0; i <= 3; i++) {
    const y = PAD.top + cH - (i / 3) * cH;
    svg += `<line x1="${PAD.left}" y1="${y}" x2="${W - PAD.right}" y2="${y}" stroke="var(--border)" stroke-width="1"/>`;
    svg += `<text x="${PAD.left - 6}" y="${y + 4}" text-anchor="end" font-size="10" fill="var(--text-3)">${YLABELS[i]}</text>`;
  }

  // Lines + dots per screen
  for (const [screen, pts] of Object.entries(byScreen)) {
    if (!pts.length) continue;
    const color = SCREEN_COLORS[screen] || '#94a3b8';
    const d = pts.map((p, i) => {
      const x = xS(new Date(p.recorded_at).getTime());
      const y = yS(p.worst_severity);
      return `${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
    }).join(' ');
    svg += `<path d="${d}" fill="none" stroke="${color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>`;
    for (const p of pts) {
      const x = xS(new Date(p.recorded_at).getTime()), y = yS(p.worst_severity);
      svg += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="4" fill="${color}" stroke="white" stroke-width="1.5"/>`;
    }
  }

  svg += '</svg>';
  container.innerHTML = svg;
}

// ── Error ─────────────────────────────────────────────────
function showError(msg) {
  document.getElementById('error-message').textContent = msg;
  showView('error');
  document.getElementById('error-retry-btn').onclick = resetApp;
}

// ── Reset ─────────────────────────────────────────────────
function resetApp() { aggregator = null; isRecording = false; showView('setup'); }

// ── Admin threshold page ──────────────────────────────────

const THRESHOLD_GROUPS = [
  {
    label: 'Knee Valgus', tests: ['squat', 'lunge'], unit: 'norm', step: 0.01, precision: 2,
    note: 'Medial deviation of knee from hip-ankle line, normalised by hip width. Higher = worse valgus.',
    keys: [
      { key: 'knee_valgus_b', label: 'Grade B trigger' },
      { key: 'knee_valgus_c', label: 'Grade C trigger' },
      { key: 'knee_valgus_d', label: 'Grade D trigger' },
      { key: 'knee_valgus_e', label: 'Grade E trigger' },
      { key: 'knee_valgus_f', label: 'Grade F trigger' },
    ],
  },
  {
    label: 'Forward Trunk Lean', tests: ['squat', 'lunge', 'overhead'], unit: '°', step: 1, precision: 1,
    keys: [
      { key: 'trunk_lean_b', label: 'Grade B trigger' },
      { key: 'trunk_lean_c', label: 'Grade C trigger' },
      { key: 'trunk_lean_d', label: 'Grade D trigger' },
      { key: 'trunk_lean_e', label: 'Grade E trigger' },
      { key: 'trunk_lean_f', label: 'Grade F trigger' },
    ],
  },
  {
    label: 'Lateral Trunk Shift', tests: ['squat', 'lunge', 'overhead'], unit: 'norm', step: 0.005, precision: 3,
    note: 'Normalised image coordinate offset (0–1 range).',
    keys: [
      { key: 'lateral_shift_b', label: 'Grade B trigger' },
      { key: 'lateral_shift_c', label: 'Grade C trigger' },
      { key: 'lateral_shift_d', label: 'Grade D trigger' },
      { key: 'lateral_shift_e', label: 'Grade E trigger' },
      { key: 'lateral_shift_f', label: 'Grade F trigger' },
    ],
  },
  {
    label: 'Lateral Spinal Flexion', tests: ['squat', 'lunge', 'overhead'], unit: '°', step: 0.5, precision: 1,
    keys: [
      { key: 'lateral_flexion_b', label: 'Grade B trigger' },
      { key: 'lateral_flexion_c', label: 'Grade C trigger' },
      { key: 'lateral_flexion_d', label: 'Grade D trigger' },
      { key: 'lateral_flexion_e', label: 'Grade E trigger' },
      { key: 'lateral_flexion_f', label: 'Grade F trigger' },
    ],
  },
  {
    label: 'Spinal Curvature', tests: ['squat', 'lunge', 'overhead'], unit: '°', step: 0.5, precision: 1,
    note: 'Deviation from 180° (straight spine). Higher = more curvature required to flag.',
    keys: [
      { key: 'spine_curve_b', label: 'Grade B trigger' },
      { key: 'spine_curve_c', label: 'Grade C trigger' },
      { key: 'spine_curve_d', label: 'Grade D trigger' },
      { key: 'spine_curve_e', label: 'Grade E trigger' },
      { key: 'spine_curve_f', label: 'Grade F trigger' },
    ],
  },
  {
    label: 'Ankle / Heel Rise', tests: ['squat', 'lunge'], unit: '°', step: 1, precision: 1,
    note: 'Lower angle = more restricted dorsiflexion. Lower threshold = more sensitive.',
    keys: [
      { key: 'ankle_df_b', label: 'Grade B trigger' },
      { key: 'ankle_df_c', label: 'Grade C trigger' },
      { key: 'ankle_df_d', label: 'Grade D trigger' },
      { key: 'ankle_df_e', label: 'Grade E trigger' },
      { key: 'ankle_df_f', label: 'Grade F trigger' },
    ],
  },
  {
    label: 'Bilateral Symmetry', tests: ['squat', 'lunge', 'overhead'], unit: 'ratio', step: 0.01, precision: 2,
    note: 'Asymmetry ratio 0–1 where 0 = perfect symmetry.',
    keys: [
      { key: 'asymmetry_b', label: 'Grade B trigger' },
      { key: 'asymmetry_c', label: 'Grade C trigger' },
      { key: 'asymmetry_d', label: 'Grade D trigger' },
      { key: 'asymmetry_e', label: 'Grade E trigger' },
      { key: 'asymmetry_f', label: 'Grade F trigger' },
    ],
  },
  {
    label: 'Upper Trunk Flexion', tests: ['lateral'], unit: '°', step: 1, precision: 1,
    note: 'Lateral view only. Ear→shoulder segment angle from vertical.',
    keys: [
      { key: 'upper_trunk_b', label: 'Grade B trigger' },
      { key: 'upper_trunk_c', label: 'Grade C trigger' },
      { key: 'upper_trunk_d', label: 'Grade D trigger' },
      { key: 'upper_trunk_e', label: 'Grade E trigger' },
      { key: 'upper_trunk_f', label: 'Grade F trigger' },
    ],
  },
  {
    label: 'Head Forward Posture', tests: ['lateral'], unit: 'norm', step: 0.005, precision: 3,
    note: 'Lateral view only. Normalised horizontal ear-to-shoulder offset.',
    keys: [
      { key: 'head_forward_b', label: 'Grade B trigger' },
      { key: 'head_forward_c', label: 'Grade C trigger' },
      { key: 'head_forward_d', label: 'Grade D trigger' },
      { key: 'head_forward_e', label: 'Grade E trigger' },
      { key: 'head_forward_f', label: 'Grade F trigger' },
    ],
  },
  {
    label: 'Squat — Trunk Lean', tests: ['squat'], unit: '°', step: 1, precision: 1,
    note: 'Squat-specific thresholds only. Optimal squat trunk lean is 20–40°, so triggering below 45° would flag normal technique.',
    keys: [
      { key: 'squat_trunk_lean_b', label: 'Grade B trigger' },
      { key: 'squat_trunk_lean_c', label: 'Grade C trigger' },
      { key: 'squat_trunk_lean_d', label: 'Grade D trigger' },
      { key: 'squat_trunk_lean_e', label: 'Grade E trigger' },
      { key: 'squat_trunk_lean_f', label: 'Grade F trigger' },
    ],
  },
  {
    label: 'Dorsiflexion — Tibial Angle', tests: ['squat', 'lunge', 'lateral'], unit: '°', step: 0.5, precision: 1,
    note: 'Lateral view only. Angle of tibia from vertical at squat depth. Optimal: 30–40°. Lower threshold = more restricted ankle.',
    keys: [
      { key: 'tibial_angle_b', label: 'Grade B trigger (lower is worse)' },
      { key: 'tibial_angle_c', label: 'Grade C trigger (lower is worse)' },
      { key: 'tibial_angle_d', label: 'Grade D trigger (lower is worse)' },
      { key: 'tibial_angle_e', label: 'Grade E trigger (lower is worse)' },
      { key: 'tibial_angle_f', label: 'Grade F trigger (lower is worse)' },
    ],
  },
  {
    label: 'Pelvic Tilt', tests: ['squat', 'lunge', 'overhead'], unit: '°', step: 0.5, precision: 1,
    note: 'Anterior view. Angle of hip line from horizontal. Even small tilts can indicate hip weakness.',
    keys: [
      { key: 'pelvic_tilt_b', label: 'Grade B trigger' },
      { key: 'pelvic_tilt_c', label: 'Grade C trigger' },
      { key: 'pelvic_tilt_d', label: 'Grade D trigger' },
      { key: 'pelvic_tilt_e', label: 'Grade E trigger' },
      { key: 'pelvic_tilt_f', label: 'Grade F trigger' },
    ],
  },
];

function loadAdminPage() {
  showView('admin');
  const current = getThresholds();
  const thresholds = {};
  for (const key of Object.keys(DEFAULT_THRESHOLDS)) {
    thresholds[key] = {
      value:        current[key],
      default:      DEFAULT_THRESHOLDS[key],
      is_overridden: current[key] !== DEFAULT_THRESHOLDS[key],
      description:  THRESHOLD_DESCRIPTIONS[key] ?? key,
    };
  }
  renderAdminPage({ thresholds });
}

function renderAdminPage(data) {
  const thresholds = data.thresholds;

  let html = `
    <div class="admin-header">
      <h1>Threshold Settings</h1>
      <p>Adjust when compensation patterns are flagged. Changes take effect on the next analysis.</p>
    </div>
    <div class="filter-tabs">
      <button class="filter-tab active" data-filter="all">All</button>
      <button class="filter-tab" data-filter="squat">🏋️ Squat</button>
      <button class="filter-tab" data-filter="lunge">🦵 Lunge</button>
      <button class="filter-tab" data-filter="overhead">🙌 Overhead</button>
      <button class="filter-tab" data-filter="lateral">↔ Lateral View</button>
    </div>
  `;

  for (const group of THRESHOLD_GROUPS) {
    const testsAttr = group.tests.join(' ');
    const testBadges = group.tests.map(t => `<span class="test-badge">${t}</span>`).join('');

    let rows = '';
    for (const { key, label } of group.keys) {
      const t = thresholds[key];
      if (!t) continue;
      const val  = t.value;
      const def  = t.default;
      const mod  = t.is_overridden;
      rows += `
        <div class="threshold-row${mod ? ' dirty' : ''}" data-key="${key}" data-default="${def}">
          <div class="threshold-label-wrap">
            <span class="threshold-label">${label}</span>
            <span class="threshold-modified${mod ? '' : ' hidden'}">Modified</span>
          </div>
          <div class="threshold-controls">
            <div class="threshold-input-wrap">
              <input
                type="number"
                class="threshold-input"
                value="${val.toFixed(group.precision)}"
                step="${group.step}"
                min="0"
                data-original="${val}"
                title="${t.description}"
              />
              <span class="threshold-unit">${group.unit}</span>
            </div>
            <span class="threshold-default">Default: ${def}</span>
            <button class="threshold-save-btn" disabled>Save</button>
            <button class="threshold-reset-btn${mod ? '' : ' hidden'}" title="Reset to default">↺</button>
          </div>
        </div>
      `;
    }

    html += `
      <div class="threshold-group" data-tests="${testsAttr}">
        <div class="card">
          <div class="threshold-group-header">
            <h2 class="card-title" style="margin-bottom:0">${group.label}</h2>
            <div class="test-badges">${testBadges}</div>
          </div>
          ${group.note ? `<p style="font-size:12px;color:var(--text-3);margin-bottom:10px;line-height:1.4">${group.note}</p>` : ''}
          ${rows}
        </div>
      </div>
    `;
  }

  html += `<div style="padding-bottom:40px"></div>`;
  views.admin.innerHTML = html;
}

views.admin.addEventListener('input', e => {
  if (!e.target.matches('.threshold-input')) return;
  const row     = e.target.closest('.threshold-row');
  const saveBtn = row.querySelector('.threshold-save-btn');
  const changed = parseFloat(e.target.value) !== parseFloat(e.target.dataset.original);
  saveBtn.disabled = !changed || isNaN(parseFloat(e.target.value));
});

views.admin.addEventListener('click', e => {
  if (e.target.matches('.filter-tab')) {
    document.querySelectorAll('.filter-tab').forEach(t => t.classList.toggle('active', t === e.target));
    const filter = e.target.dataset.filter;
    document.querySelectorAll('.threshold-group').forEach(g => {
      g.style.display = (filter === 'all' || g.dataset.tests.split(' ').includes(filter)) ? '' : 'none';
    });
    return;
  }
  if (e.target.matches('.threshold-save-btn')) { saveThreshold(e.target.closest('.threshold-row')); return; }
  if (e.target.matches('.threshold-reset-btn')) { resetThreshold(e.target.closest('.threshold-row')); }
});

function saveThreshold(row) {
  const key     = row.dataset.key;
  const input   = row.querySelector('.threshold-input');
  const saveBtn = row.querySelector('.threshold-save-btn');
  const value   = parseFloat(input.value);
  if (isNaN(value)) return;

  try {
    saveThresholdOverrides({ [key]: value });
    input.dataset.original = value;
    row.querySelector('.threshold-modified').classList.remove('hidden');
    row.querySelector('.threshold-reset-btn').classList.remove('hidden');
    showToast('Saved');
  } catch {
    showToast('Failed to save', 'error');
  } finally {
    saveBtn.disabled = parseFloat(input.value) === parseFloat(input.dataset.original);
  }
}

function resetThreshold(row) {
  const key      = row.dataset.key;
  const input    = row.querySelector('.threshold-input');
  const resetBtn = row.querySelector('.threshold-reset-btn');
  const defVal   = parseFloat(row.dataset.default);

  try {
    clearThresholdOverride(key);
    input.value = defVal;
    input.dataset.original = defVal;
    row.querySelector('.threshold-modified').classList.add('hidden');
    resetBtn.classList.add('hidden');
    row.querySelector('.threshold-save-btn').disabled = true;
    showToast('Reset to default');
  } catch {
    showToast('Failed to reset', 'error');
  }
}

function showToast(msg, type = 'success') {
  const existing = document.querySelector('.toast');
  if (existing) existing.remove();
  const toast = document.createElement('div');
  toast.className = `toast${type === 'error' ? ' toast-error' : ''}`;
  toast.textContent = msg;
  document.body.appendChild(toast);
  requestAnimationFrame(() => requestAnimationFrame(() => toast.classList.add('visible')));
  setTimeout(() => {
    toast.classList.remove('visible');
    setTimeout(() => toast.remove(), 300);
  }, 2500);
}

// ── Threshold Calibration ────────────────────────────────

const SENS_MULT = { strict: 0.6, normal: 1.0, lenient: 1.6 };
const SENS_DESC = {
  strict:  'Strict: thresholds sit 40% closer to your optimal — flags minor deviations.',
  normal:  'Normal: standard tolerance bands (10–30° beyond your captured optimum).',
  lenient: 'Lenient: 60% wider tolerance — only flags clear compensations.',
};

// Each entry: (meanVal, screenType, sensitivityMultiplier) → [{key, value, label}]
const CALIBRATION_MAP = {
  trunk_lean_degrees: (mean, screenType, m) => {
    const p = screenType === 'squat' ? 'squat_trunk_lean' : 'trunk_lean';
    return [
      { key: `${p}_mild`,     value: mean + 10 * m, label: 'Mild lean trigger' },
      { key: `${p}_moderate`, value: mean + 20 * m, label: 'Moderate lean trigger' },
      { key: `${p}_severe`,   value: mean + 30 * m, label: 'Severe lean trigger' },
    ];
  },
  tibial_angle_left: (mean, _, m) => [
    { key: 'tibial_angle_restricted_mild',   value: mean - 5  * m, label: 'Tibial — mild restriction' },
    { key: 'tibial_angle_restricted_severe', value: mean - 15 * m, label: 'Tibial — severe restriction' },
  ],
  tibial_angle_right: (mean, _, m) => [
    { key: 'tibial_angle_restricted_mild',   value: mean - 5  * m, label: 'Tibial — mild restriction' },
    { key: 'tibial_angle_restricted_severe', value: mean - 15 * m, label: 'Tibial — severe restriction' },
  ],
  left_knee_frontal_angle: (mean, _, m) => [
    { key: 'knee_valgus_mild',     value: mean - 3  * m, label: 'Knee valgus mild trigger' },
    { key: 'knee_valgus_moderate', value: mean - 7  * m, label: 'Knee valgus moderate trigger' },
    { key: 'knee_valgus_severe',   value: mean - 15 * m, label: 'Knee valgus severe trigger' },
  ],
  right_knee_frontal_angle: (mean, _, m) => [
    { key: 'knee_valgus_mild',     value: mean - 3  * m, label: 'Knee valgus mild trigger' },
    { key: 'knee_valgus_moderate', value: mean - 7  * m, label: 'Knee valgus moderate trigger' },
    { key: 'knee_valgus_severe',   value: mean - 15 * m, label: 'Knee valgus severe trigger' },
  ],
  pelvic_tilt_degrees: (mean, _, m) => {
    const ref = Math.abs(mean);
    return [
      { key: 'pelvic_tilt_mild',     value: ref + 2 * m, label: 'Pelvic tilt — mild trigger' },
      { key: 'pelvic_tilt_moderate', value: ref + 5 * m, label: 'Pelvic tilt — moderate trigger' },
      { key: 'pelvic_tilt_severe',   value: ref + 9 * m, label: 'Pelvic tilt — severe trigger' },
    ];
  },
  lateral_flexion_degrees: (mean, _, m) => {
    const ref = Math.abs(mean);
    return [
      { key: 'lateral_flexion_mild',     value: ref + 5  * m, label: 'Lateral flexion — mild' },
      { key: 'lateral_flexion_moderate', value: ref + 10 * m, label: 'Lateral flexion — moderate' },
      { key: 'lateral_flexion_severe',   value: ref + 15 * m, label: 'Lateral flexion — severe' },
    ];
  },
  upper_trunk_angle: (mean, _, m) => [
    { key: 'upper_trunk_mild',     value: mean + 10 * m, label: 'Upper trunk — mild' },
    { key: 'upper_trunk_moderate', value: mean + 20 * m, label: 'Upper trunk — moderate' },
    { key: 'upper_trunk_severe',   value: mean + 30 * m, label: 'Upper trunk — severe' },
  ],
  lateral_trunk_shift: (mean, _, m) => {
    const ref = Math.abs(mean);
    return [
      { key: 'lateral_shift_mild',     value: ref + 0.02 * m, label: 'Lateral shift — mild' },
      { key: 'lateral_shift_moderate', value: ref + 0.04 * m, label: 'Lateral shift — moderate' },
      { key: 'lateral_shift_severe',   value: ref + 0.07 * m, label: 'Lateral shift — severe' },
    ];
  },
};

function computeCalibrationSuggestions(stats, screenType, sensitivity) {
  const m = SENS_MULT[sensitivity] || 1.0;
  const keyAccum = {}; // key → {values[], label}

  for (const stat of stats) {
    const fn = CALIBRATION_MAP[stat.field];
    if (!fn || stat.mean == null) continue;
    for (const { key, value, label } of fn(stat.mean, screenType, m)) {
      if (!keyAccum[key]) keyAccum[key] = { values: [], label };
      keyAccum[key].values.push(value);
    }
  }

  return Object.entries(keyAccum).map(([key, { values, label }]) => ({
    key,
    label,
    // Average if multiple stats contribute to the same key (e.g. left+right tibial)
    value: values.reduce((a, b) => a + b, 0) / values.length,
  }));
}

function renderCalibrationPanel(data) {
  const container = document.getElementById('calibration-panel');
  if (!container) return;

  let sensitivity = 'normal';
  // Build currentThresholds in the same shape renderAdminPage expects
  const _current = getThresholds();
  let currentThresholds = {};
  for (const key of Object.keys(DEFAULT_THRESHOLDS)) {
    currentThresholds[key] = { value: _current[key], default: DEFAULT_THRESHOLDS[key] };
  }

  function fmt(v) {
    if (v == null) return '—';
    return Math.abs(v) < 1 ? v.toFixed(3) : v.toFixed(1);
  }

  function drawPanel() {
    const suggestions = computeCalibrationSuggestions(data.stats, data.screen_type, sensitivity);

    if (suggestions.length === 0) {
      container.innerHTML = `<p style="color:var(--text-3);font-size:13px">No calibratable metrics captured. Try a lateral view for sagittal calibration, or anterior for frontal-plane calibration.</p>`;
      return;
    }

    let html = `
      <div class="sens-row">
        <span class="sens-label">Sensitivity</span>
        <div class="sens-btns">
          ${['strict', 'normal', 'lenient'].map(s =>
            `<button class="sens-btn${sensitivity === s ? ' active' : ''}" data-sens="${s}">${s[0].toUpperCase() + s.slice(1)}</button>`
          ).join('')}
        </div>
      </div>
      <p class="sens-desc-text">${SENS_DESC[sensitivity]}</p>
      <table class="calibration-table">
        <thead><tr><th>Threshold</th><th>Current</th><th>Suggested</th></tr></thead>
        <tbody>
          ${suggestions.map(s => `
            <tr>
              <td class="calib-label">
                ${s.label}
                <span class="calib-key">${s.key.replace(/_/g, ' ')}</span>
              </td>
              <td class="calib-current">${currentThresholds[s.key] ? fmt(currentThresholds[s.key].value) : '—'}</td>
              <td><input class="calib-input" type="number" step="0.1" data-key="${s.key}" value="${fmt(s.value)}"/></td>
            </tr>
          `).join('')}
        </tbody>
      </table>
      <button class="btn-primary" id="apply-calib-btn">
        Apply ${suggestions.length} Threshold Update${suggestions.length !== 1 ? 's' : ''}
      </button>
    `;
    container.innerHTML = html;

    container.querySelectorAll('.sens-btn').forEach(btn =>
      btn.addEventListener('click', () => { sensitivity = btn.dataset.sens; drawPanel(); })
    );

    document.getElementById('apply-calib-btn').addEventListener('click', () => {
      const updates = {};
      container.querySelectorAll('.calib-input').forEach(inp => {
        const v = parseFloat(inp.value);
        if (!isNaN(v)) updates[inp.dataset.key] = v;
      });
      const btn = document.getElementById('apply-calib-btn');
      try {
        saveThresholdOverrides(updates);
        // Refresh currentThresholds and re-render current column
        const refreshed = getThresholds();
        for (const key of Object.keys(DEFAULT_THRESHOLDS)) {
          currentThresholds[key] = { value: refreshed[key], default: DEFAULT_THRESHOLDS[key] };
        }
        container.querySelectorAll('.calib-current').forEach((td, i) => {
          const key = suggestions[i]?.key;
          if (key && currentThresholds[key]) td.textContent = fmt(currentThresholds[key].value);
        });
        showToast('Thresholds calibrated from recording');
        btn.textContent = `✓ Applied`;
      } catch {
        showToast('Failed to apply', 'error');
      }
    });
  }

  drawPanel();
}

// ── Report ───────────────────────────────────────────────

const RECOMMENDATIONS = {
  'Knee Valgus': {
    what: 'Your knee is collapsing inward (toward the midline) during the movement.',
    means: 'This places excess stress on the knee ligaments and cartilage. It often indicates weakness in the glutes or hip abductors, and/or tightness in the hip flexors.',
    tips: [
      'Strengthen glutes: clamshells, side-lying hip abduction, glute bridges, hip thrusts',
      'Add lateral band walks and monster walks to build hip stability',
      'Practice single-leg squats in front of a mirror to improve knee tracking awareness',
      'Check ankle mobility — limited dorsiflexion can force knees inward',
    ],
  },
  'Forward Trunk Lean': {
    what: 'Your upper body is leaning forward more than expected during the movement.',
    means: 'Excessive forward lean can increase load on the lower back and may indicate tight hip flexors, limited ankle mobility, or weakness in the core and posterior chain.',
    tips: [
      'Strengthen your core: planks, dead bugs, Pallof press',
      'Improve ankle dorsiflexion with calf stretches and ankle mobility drills',
      'Stretch hip flexors with couch stretches and lunging hip flexor stretches',
      'Practise goblet squats to reinforce an upright torso',
    ],
  },
  'Lateral Trunk Shift': {
    what: 'Your shoulders are shifting to one side rather than staying centred over your hips.',
    means: 'A lateral shift can signal unilateral hip weakness, leg-length discrepancy, or pain-avoidance behaviour. It concentrates load on one side of the spine.',
    tips: [
      'Strengthen the hip abductors on the weak side: side-lying raises, cable hip abduction',
      'Check for and address any leg-length difference with a physiotherapist',
      'Single-leg balance and single-leg deadlift progressions help correct asymmetry',
    ],
  },
  'Heel Rise': {
    what: 'Your heel is lifting off the floor, or your ankle is not bending enough during the movement.',
    means: 'Limited ankle dorsiflexion is often the culprit. This forces compensations upstream — knees cave in, trunk leans forward, or the lower back rounds.',
    tips: [
      'Daily calf and Achilles stretches (both straight-knee and bent-knee)',
      'Ankle circles, banded ankle mobilisation, and wall ankle stretches',
      'Elevate your heels temporarily while you build mobility',
      'Consider manual therapy if mobility does not improve after 4–6 weeks',
    ],
  },
  'Dorsiflexion': {
    what: 'Your ankle is not bending enough during the movement.',
    means: 'Restricted ankle dorsiflexion limits how far your knee can travel over your toes. This forces the heel up or the trunk to lean forward to compensate.',
    tips: [
      'Perform daily ankle mobility drills: wall ankle stretches, banded joint mobilisation',
      'Foam roll and stretch the calf and Achilles complex',
      'Work on soft tissue release of the peroneals and anterior shin muscles',
    ],
  },
  'Pelvic Tilt': {
    what: 'One side of your pelvis is dropping lower than the other during the movement.',
    means: 'This is often a sign of hip abductor weakness (Trendelenburg sign) on the standing or loaded leg. It can also indicate a leg-length difference.',
    tips: [
      'Strengthen hip abductors: clamshells, side-lying raises, hip thrusts with band',
      'Single-leg balance training and single-leg deadlifts to build stability',
      'Consult a physiotherapist if the pelvic drop is consistent and pronounced',
    ],
  },
  'Lateral Spinal Flexion': {
    what: 'Your spine is bending sideways during the movement.',
    means: 'Side-bending of the spine during loading can indicate lateral core weakness or a pain avoidance pattern. It increases asymmetric compressive forces on the discs.',
    tips: [
      'Strengthen the lateral core: side planks, Copenhagen planks, suitcase carries',
      'Address any hip tightness that may cause the trunk to deviate',
      'Unilateral exercises (single-arm carries, split squats) help expose and correct asymmetry',
    ],
  },
  'Spinal Segmental Curvature': {
    what: 'There is an increased curve (bend) detected between your upper and lower back segments.',
    means: 'This may reflect thoracic kyphosis (rounding in the upper back) or lumbar lordosis. Sustained postures like desk work can reinforce these patterns over time.',
    tips: [
      'Thoracic extension mobility work: foam rolling the thoracic spine, cat-cow stretches',
      'Strengthen the lower and middle trapezius: face pulls, rows, Y-T-W exercises',
      'Work on hip flexor length to reduce anterior pelvic tilt and lumbar compensation',
      'Consider a posture assessment with a physiotherapist or chiropractor',
    ],
  },
  'Head Forward Posture': {
    what: 'Your head is positioned noticeably in front of your shoulders.',
    means: 'Forward head posture increases the effective weight the neck must support. It is commonly associated with tight upper traps and pec minor, and weak deep neck flexors.',
    tips: [
      'Chin tucks: 3 × 10–15 reps daily to strengthen deep neck flexors',
      'Stretch the upper traps, scalenes, and pec minor',
      'Strengthen mid/lower traps and rhomboids: face pulls, band pull-aparts, rows',
      'Review your workstation setup — screen height and seating posture matter',
    ],
  },
  'Upper Trunk Flexion': {
    what: 'Your upper back (thoracic spine) is rounded or flexed forward more than expected.',
    means: 'Thoracic kyphosis or poor upper-back mobility can restrict overhead movement and increase load on the cervical spine and shoulders.',
    tips: [
      'Thoracic spine mobilisation: foam roller extensions, thoracic rotation stretches',
      'Strengthen upper-back postural muscles: face pulls, band pull-aparts',
      'Overhead mobility work: lat stretches, doorway shoulder stretches',
    ],
  },
  'Asymmetry': {
    what: 'There is a meaningful difference between the left and right side of the movement.',
    means: 'Bilateral asymmetry can reflect an old injury, muscle imbalance, or habitual movement pattern. Left–right imbalances increase injury risk over time, especially under load.',
    tips: [
      'Include unilateral exercises in your training: single-leg squats, split squats, single-arm rows',
      'Start unilateral sets with your weaker side and match reps on the stronger side',
      'Video yourself from the front to monitor symmetry over time',
      'Consider a professional movement screen with a physiotherapist or strength coach',
    ],
  },
};

function getRecommendationInfo(findingName) {
  for (const [keyword, info] of Object.entries(RECOMMENDATIONS)) {
    if (findingName.includes(keyword)) return info;
  }
  return null;
}

function generateSummary(data) {
  const sev = data.worst_severity;
  const count = data.findings.length;
  const screenName = data.screen_name || 'movement screen';

  if (sev === 'A' || sev === 'none') {
    return `Your ${screenName} showed no significant compensation patterns. Your movement quality looks great — keep up the good work and continue monitoring over time.`;
  }

  const sevText = { B: 'minimal', C: 'minor', D: 'moderate', E: 'significant', F: 'notable', mild: 'minor', moderate: 'moderate', severe: 'notable' }[sev] || sev;
  const findingNames = data.findings.map(f => f.name.replace(/ \(.*\)$/, '')).filter((v, i, a) => a.indexOf(v) === i);
  const listText = findingNames.length <= 2
    ? findingNames.join(' and ')
    : findingNames.slice(0, 2).join(', ') + ` and ${findingNames.length - 2} other area${findingNames.length - 2 > 1 ? 's' : ''}`;

  return `Your ${screenName} identified ${count} compensation pattern${count !== 1 ? 's' : ''}, with ${sevText} concerns in: ${listText}. These findings are not a diagnosis — they highlight areas to focus on in your training and mobility work. Use the recommendations below as a starting point, and consider working with a qualified physiotherapist or movement coach for a personalised programme.`;
}

function renderReport(data, source) {
  const sev   = data.worst_severity;
  const color = SEV_COLOR[sev];
  const icon = { A: '✓', B: '~', C: '~', D: '!', E: '▲', F: '▲', none: '✓', mild: '~', moderate: '!', severe: '▲' }[sev] ?? '?';
  const grade = {
    A: 'Excellent', B: 'Good — minimal notes', C: 'Good — minor notes',
    D: 'Fair — attention needed', E: 'Fair — significant concerns', F: 'Needs improvement',
    none: 'Excellent', mild: 'Good — minor notes', moderate: 'Fair — attention needed', severe: 'Needs improvement',
  }[sev] ?? sev;
  const gradeDesc = {
    A: 'No significant compensations detected.',
    B: 'Minimal compensations detected — keep monitoring.',
    C: 'Small compensations present — address in your routine.',
    D: 'Moderate compensations found — prioritise the recommendations below.',
    E: 'Significant compensations found — prioritise the recommendations below.',
    F: 'Significant compensations found — consider professional guidance.',
    none: 'No significant compensations detected.',
    mild: 'Small compensations present — address in your routine.',
    moderate: 'Moderate compensations found — prioritise the recommendations below.',
    severe: 'Significant compensations found — consider professional guidance.',
  }[sev] ?? '';

  const dateStr = data.recorded_at
    ? new Date(data.recorded_at).toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric' })
    : new Date().toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric' });

  let html = `<div class="report-wrap">`;

  // Masthead
  html += `
    <div class="report-masthead">
      <div class="report-logo-row">
        <span class="report-logo-mark" aria-hidden="true"></span>
        <span>MovementScreen</span>
      </div>
      <h2>Movement Assessment Report</h2>
      <p>${data.screen_name} · ${ANGLE_LABEL[data.camera_angle] ?? data.camera_angle} view · ${dateStr}</p>
    </div>
  `;

  // Grade
  html += `
    <div class="card" style="margin-bottom:16px">
      <div class="report-grade-row">
        <div class="report-grade-circle" style="color:${color};border-color:${color}">${icon}</div>
        <div class="report-grade-text">
          <h3>${grade}</h3>
          <p>${gradeDesc}</p>
        </div>
      </div>
    </div>
  `;

  // Summary
  html += `
    <p class="report-section-title">Summary</p>
    <div class="report-summary-card">${generateSummary(data)}</div>
  `;

  // Findings
  html += `<p class="report-section-title">Findings &amp; Recommendations</p>`;

  if (data.findings.length === 0) {
    html += `
      <div class="report-no-findings">
        <span class="icon">✓</span>
        <p>No compensation patterns were detected in this assessment.</p>
      </div>
    `;
  } else {
    for (const f of data.findings) {
      const c    = SEV_COLOR[f.severity];
      const info = getRecommendationInfo(f.name);
      html += `
        <div class="report-finding" style="--border-color:${c}">
          <div class="report-finding-header">
            <span class="severity-badge" style="background:${c}">${SEV_LABEL[f.severity]}</span>
            <span class="report-finding-name">${f.name}</span>
          </div>
          ${info ? `<p class="report-what">${info.what}</p>` : `<p class="report-what">${f.description}</p>`}
          ${f.metric_value != null ? `<p style="font-size:12px;color:var(--text-3);margin-bottom:8px">${f.metric_label}: ${typeof f.metric_value === 'number' ? f.metric_value.toFixed(1) : f.metric_value}</p>` : ''}
          ${info ? `
            <p class="report-tips-title">What to do</p>
            <ul class="report-tips">${info.tips.map(t => `<li>${t}</li>`).join('')}</ul>
          ` : ''}
        </div>
      `;
    }
  }

  // Stats table
  if (data.stats && data.stats.length > 0) {
    html += `
      <p class="report-section-title">Measurement Data</p>
      <table class="report-stats-table">
        <thead><tr><th>Measurement</th><th>Min</th><th>Mean</th><th>Max</th></tr></thead>
        <tbody>
          ${data.stats.map(s => { const u = ['left_knee_frontal_angle','right_knee_frontal_angle','lateral_trunk_shift','head_forward_offset'].includes(s.field) ? '' : '°'; return `<tr><td>${s.name}</td><td>${s.min}${u}</td><td>${s.mean}${u}</td><td>${s.max}${u}</td></tr>`; }).join('')}
        </tbody>
      </table>
    `;
  }

  // Disclaimer
  html += `
    <p class="report-disclaimer">
      <strong>Disclaimer:</strong> This report is generated automatically from video-based pose estimation and is intended for informational and training purposes only. It is not a medical diagnosis. Measurement accuracy depends on camera angle, lighting, and body visibility. Consult a qualified physiotherapist, sports medicine physician, or movement specialist before making changes to your training or rehabilitation programme.
    </p>
  `;

  // Actions
  html += `
    <div class="report-actions">
      <button class="btn-primary" id="report-print-btn">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 6 2 18 2 18 9"/><path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"/><rect x="6" y="14" width="12" height="8"/></svg>
        Print / Save PDF
      </button>
      <button class="btn-ghost" id="report-back-btn">← Back</button>
    </div>
  </div>`;

  views.report.innerHTML = html;
  views.report.scrollTop = 0;
  document.getElementById('report-print-btn').addEventListener('click', () => window.print());
  document.getElementById('report-back-btn').addEventListener('click', source === 'history' ? loadHistory : () => showView('results'));
  showView('report');
}

// ── Boot ─────────────────────────────────────────────────
updateHeader();
showView('setup');
