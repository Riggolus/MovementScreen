'use strict';

import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs';

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
let authUser  = null;
let authToken = null;

function loadAuth() {
  authToken = localStorage.getItem('ms_token');
  const raw = localStorage.getItem('ms_user');
  authUser  = raw ? JSON.parse(raw) : null;
}

function saveAuth(data) {
  authToken = data.access_token;
  authUser  = data.user;
  localStorage.setItem('ms_token',         data.access_token);
  localStorage.setItem('ms_refresh_token', data.refresh_token);
  localStorage.setItem('ms_user',          JSON.stringify(data.user));
}

function clearAuth() {
  authToken = null;
  authUser  = null;
  localStorage.removeItem('ms_token');
  localStorage.removeItem('ms_refresh_token');
  localStorage.removeItem('ms_user');
}

async function authFetch(url, opts = {}) {
  const headers = { ...(opts.headers || {}) };
  if (authToken) headers['Authorization'] = `Bearer ${authToken}`;

  let res = await fetch(url, { ...opts, headers });

  // Try refreshing once on 401
  if (res.status === 401) {
    const rt = localStorage.getItem('ms_refresh_token');
    if (rt) {
      const rRes = await fetch('/auth/refresh', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: rt }),
      });
      if (rRes.ok) {
        const rData = await rRes.json();
        authToken = rData.access_token;
        localStorage.setItem('ms_token', authToken);
        headers['Authorization'] = `Bearer ${authToken}`;
        res = await fetch(url, { ...opts, headers });
      } else {
        clearAuth();
        updateHeader();
        showView('auth');
        throw new Error('Session expired. Please log in again.');
      }
    }
  }
  return res;
}

// ── App state ─────────────────────────────────────────────
let currentScreen  = 'squat';
let currentSide    = 'left';
let currentAngle   = 'anterior';
let facingMode     = 'environment';
let mediaStream    = null;
let mediaRecorder  = null;
let recordedChunks = [];
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
  error:      document.getElementById('view-error'),
};
const preview        = document.getElementById('preview');
const skeletonCanvas = document.getElementById('skeleton-canvas');
const skeletonCtx    = skeletonCanvas.getContext('2d');
const timerEl        = document.getElementById('timer');
const lungeOptions   = document.getElementById('lunge-options');
const poseStatus     = document.getElementById('pose-status');
const headerNav      = document.getElementById('header-nav');

// ── View helpers ─────────────────────────────────────────
function showView(name) {
  Object.values(views).forEach(v => v.classList.remove('active'));
  views[name].classList.add('active');
}

function updateHeader() {
  if (!authUser) {
    headerNav.innerHTML = '';
    return;
  }
  headerNav.innerHTML = `
    <button class="nav-btn" id="nav-history-btn">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
      History
    </button>
    <span class="user-chip">${authUser.name.split(' ')[0]}</span>
    <button class="nav-btn danger" id="nav-logout-btn">Log out</button>
  `;
  document.getElementById('nav-history-btn').addEventListener('click', loadHistory);
  document.getElementById('nav-logout-btn').addEventListener('click', logout);
}

function logout() {
  clearAuth();
  updateHeader();
  showView('auth');
}

// ── Auth forms ────────────────────────────────────────────
document.querySelectorAll('.auth-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.auth-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    const isLogin = tab.dataset.tab === 'login';
    document.getElementById('login-form').classList.toggle('hidden', !isLogin);
    document.getElementById('register-form').classList.toggle('hidden', isLogin);
  });
});

document.getElementById('login-form').addEventListener('submit', async e => {
  e.preventDefault();
  const btn = e.target.querySelector('button[type=submit]');
  const err = document.getElementById('login-error');
  err.classList.add('hidden');
  btn.disabled = true;
  btn.textContent = 'Logging in…';
  try {
    const fd = new FormData(e.target);
    const res = await fetch('/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email: fd.get('email'), password: fd.get('password') }),
    });
    if (!res.ok) {
      const d = await res.json();
      throw new Error(d.detail || 'Login failed.');
    }
    saveAuth(await res.json());
    updateHeader();
    showView('setup');
  } catch (ex) {
    err.textContent = ex.message;
    err.classList.remove('hidden');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Log in';
  }
});

document.getElementById('register-form').addEventListener('submit', async e => {
  e.preventDefault();
  const btn = e.target.querySelector('button[type=submit]');
  const err = document.getElementById('register-error');
  err.classList.add('hidden');
  btn.disabled = true;
  btn.textContent = 'Creating account…';
  try {
    const fd = new FormData(e.target);
    const res = await fetch('/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: fd.get('name'), email: fd.get('email'), password: fd.get('password') }),
    });
    if (!res.ok) {
      const d = await res.json();
      throw new Error(d.detail || 'Registration failed.');
    }
    saveAuth(await res.json());
    updateHeader();
    showView('setup');
  } catch (ex) {
    err.textContent = ex.message;
    err.classList.remove('hidden');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Create account';
  }
});

// ── Screen / angle / side selectors ──────────────────────
document.querySelectorAll('.screen-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.screen-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentScreen = btn.dataset.screen;
    lungeOptions.classList.toggle('hidden', currentScreen !== 'lunge');
  });
});

document.querySelectorAll('.angle-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.angle-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentAngle = btn.dataset.angle;
  });
});

document.querySelectorAll('.toggle-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentSide = btn.dataset.side;
  });
});

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

  if (poseLandmarker) {
    drawingUtils = new DrawingUtils(skeletonCtx);
    startSkeletonLoop();
  }

  const mimeType = getSupportedMimeType();
  mediaRecorder  = new MediaRecorder(mediaStream, mimeType ? { mimeType } : {});
  recordedChunks = [];
  mediaRecorder.ondataavailable = e => { if (e.data.size > 0) recordedChunks.push(e.data); };
  mediaRecorder.onstop = uploadAndAnalyse;
  mediaRecorder.start(100);
  startTimer();
}

// ── Skeleton loop ─────────────────────────────────────────
function startSkeletonLoop() {
  let lastTs = -1;
  function loop() {
    if (!poseLandmarker || !mediaRecorder || mediaRecorder.state === 'inactive') return;
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
          drawingUtils.drawConnectors(lms, PoseLandmarker.POSE_CONNECTIONS, {
            color: 'rgba(99,102,241,.85)', lineWidth: 2.5,
          });
          drawingUtils.drawLandmarks(lms, {
            color: '#ffffff', fillColor: 'rgba(99,102,241,.7)', lineWidth: 1, radius: 4,
          });
          poseStatus.textContent = 'Pose detected';
          poseStatus.classList.add('detected');
        } else {
          poseStatus.textContent = 'Waiting for pose…';
          poseStatus.classList.remove('detected');
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
  mediaRecorder.stop();
  mediaStream.getTracks().forEach(t => t.stop());
  showView('processing');
}

document.getElementById('cancel-btn').addEventListener('click', () => {
  stopTimer(); stopSkeletonLoop();
  if (mediaRecorder?.state !== 'inactive') mediaRecorder.stop();
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
    if (mediaRecorder?.state !== 'inactive') mediaRecorder.stop();
    const mt = getSupportedMimeType();
    mediaRecorder = new MediaRecorder(mediaStream, mt ? { mimeType: mt } : {});
    mediaRecorder.ondataavailable = e => { if (e.data.size > 0) recordedChunks.push(e.data); };
    mediaRecorder.onstop = uploadAndAnalyse;
    mediaRecorder.start(100);
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

// ── Upload & Analyse ──────────────────────────────────────
async function uploadAndAnalyse() {
  const mimeType = recordedChunks[0]?.type || 'video/webm';
  const ext      = mimeType.includes('mp4') ? '.mp4' : '.webm';
  const blob     = new Blob(recordedChunks, { type: mimeType });

  const form = new FormData();
  form.append('video',            blob, `recording${ext}`);
  form.append('screen',           currentScreen);
  form.append('lead_side',        currentSide);
  form.append('camera_angle',     currentAngle);
  form.append('model_complexity', '1');

  try {
    const res = await authFetch('/analyse', { method: 'POST', body: form });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Server error ${res.status}`);
    }
    renderResults(await res.json());
    showView('results');
  } catch (err) {
    showError(err.message);
  }
}

// ── Results rendering ─────────────────────────────────────
const SEV_COLOR  = { none: 'var(--none)', mild: 'var(--mild)', moderate: 'var(--moderate)', severe: 'var(--severe)' };
const SEV_LABEL  = { none: 'Pass', mild: 'Mild', moderate: 'Moderate', severe: 'Severe' };
const SEV_EMOJI  = { none: '✓', mild: '●', moderate: '◆', severe: '▲' };
const ANGLE_LABEL = { anterior: 'Anterior', lateral: 'Lateral', posterior: 'Posterior' };

function renderResults(data) {
  const sev = data.worst_severity, color = SEV_COLOR[sev];

  let html = `
    <div class="results-header">
      <div class="results-grade-ring" style="border-color:${color};color:${color}">${SEV_EMOJI[sev]}</div>
      <h1>${data.screen_name}</h1>
      <p class="results-meta">${data.frame_count} frames · ${ANGLE_LABEL[data.camera_angle] ?? ''} view</p>
      <span class="overall-pill" style="background:${color}">${SEV_LABEL[sev]}</span>
      ${data.saved ? `<p class="saved-badge">✓ Saved to your history</p>` : ''}
    </div>
    <div class="results-body">
      <h2 class="section-title">Findings</h2>
  `;

  if (data.findings.length === 0) {
    html += `<div class="no-findings"><span class="no-findings-icon">✓</span><p>No compensations detected — great movement quality!</p></div>`;
  } else {
    for (const f of data.findings) {
      const c = SEV_COLOR[f.severity];
      html += `
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

  if (data.stats.length > 0) {
    html += `<h2 class="section-title">Joint Angles</h2><div class="stats-grid">`;
    for (const s of data.stats) {
      html += `
        <div class="stat-card">
          <div class="stat-name">${s.name}</div>
          <div class="stat-values">
            <div class="stat-item"><span class="stat-label">Min</span><span class="stat-value">${s.min}°</span></div>
            <div class="stat-item main"><span class="stat-label">Mean</span><span class="stat-value">${s.mean}°</span></div>
            <div class="stat-item"><span class="stat-label">Max</span><span class="stat-value">${s.max}°</span></div>
          </div>
        </div>
      `;
    }
    html += `</div>`;
  }

  html += `
      <div class="results-actions">
        <button class="btn-primary" id="again-btn">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-3.45"/></svg>
          Record Again
        </button>
        ${authUser ? `<button class="btn-primary" id="history-from-results-btn" style="background:var(--surface-2);color:var(--text);border:1px solid var(--border);box-shadow:none;">View History</button>` : ''}
      </div>
    </div>
  `;

  views.results.innerHTML = html;
  views.results.scrollTop = 0;
  document.getElementById('again-btn').addEventListener('click', resetApp);
  document.getElementById('history-from-results-btn')?.addEventListener('click', loadHistory);
}

// ── History / Progress ────────────────────────────────────
const SCREEN_EMOJI = { squat: '🏋️', lunge: '🦵', overhead: '🙌' };
const SCREEN_COLORS = { squat: '#6366f1', lunge: '#8b5cf6', overhead: '#0ea5e9' };

async function loadHistory() {
  showView('history');
  views.history.innerHTML = `<div class="processing-content"><div class="spinner-ring"></div></div>`;

  try {
    const [aRes, pRes] = await Promise.all([
      authFetch('/assessments'),
      authFetch('/progress'),
    ]);
    const { assessments } = await aRes.json();
    const { by_screen }   = await pRes.json();
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
              <div class="assessment-title">${screenName}${a.lead_side ? ` (${a.lead_side})` : ''} · ${ANGLE_LABEL[a.camera_angle] ?? a.camera_angle}</div>
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
    const res = await authFetch(`/assessments/${id}`);
    const data = await res.json();
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
    body.innerHTML = inner;
  } catch {
    body.innerHTML = `<p style="padding:16px;color:var(--severe);font-size:13px">Failed to load details.</p>`;
  }
}

// ── Trend chart (SVG) ─────────────────────────────────────
function drawTrendChart(container, byScreen) {
  const W   = container.clientWidth || 320;
  const H   = 160;
  const PAD = { top: 12, right: 16, bottom: 28, left: 44 };
  const SEV_NUM = { none: 0, mild: 1, moderate: 2, severe: 3 };
  const YLABELS = ['None', 'Mild', 'Mod', 'Severe'];

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
function resetApp() { recordedChunks = []; showView('setup'); }

// ── Mime helper ───────────────────────────────────────────
function getSupportedMimeType() {
  return ['video/webm;codecs=vp9,opus', 'video/webm;codecs=vp8,opus', 'video/webm', 'video/mp4']
    .find(t => MediaRecorder.isTypeSupported(t)) || '';
}

// ── Boot ─────────────────────────────────────────────────
loadAuth();
updateHeader();
if (authUser && authToken) {
  // Validate token is still good
  fetch('/auth/me', { headers: { Authorization: `Bearer ${authToken}` } })
    .then(r => r.ok ? showView('setup') : (clearAuth(), updateHeader(), showView('auth')))
    .catch(() => showView('setup'));
} else {
  showView('auth');
}
