const CACHE = 'movementscreen-v4';

const ASSETS = [
  '/',
  '/manifest.json',
  '/static/style.css',
  '/static/app.js',
  '/static/analysis/aggregator.js',
  '/static/analysis/compensation.js',
  '/static/analysis/geometry.js',
  '/static/analysis/joint_angles.js',
  '/static/analysis/screens.js',
  '/static/analysis/thresholds.js',
  '/static/db/local_db.js',
  '/static/icons/icon-192.png',
  '/static/icons/icon-512.png',
];

// Install: cache all local assets (individual failures don't abort install)
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE).then(cache =>
      Promise.allSettled(ASSETS.map(url => cache.add(url)))
    )
  );
  self.skipWaiting();
});

// Activate: delete old caches
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// Fetch: cache-first for same-origin, network-only for CDN (MediaPipe)
self.addEventListener('fetch', event => {
  if (event.request.method !== 'GET') return;

  const url = new URL(event.request.url);

  // Let CDN requests (MediaPipe, Google Fonts) go straight to network
  if (url.origin !== self.location.origin) return;

  event.respondWith(
    caches.match(event.request).then(cached => {
      if (cached) return cached;
      return fetch(event.request).then(response => {
        if (response.ok) {
          caches.open(CACHE).then(cache => cache.put(event.request, response.clone()));
        }
        return response;
      });
    })
  );
});
