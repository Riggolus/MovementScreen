/**
 * build.js
 * Assembles the www/ directory that Capacitor uses as its web root.
 * Mirrors the URL structure served by server.py:
 *   /             → static/index.html
 *   /manifest.json → static/manifest.json
 *   /sw.js        → static/sw.js
 *   /static/*     → static/*
 */

const fs   = require('fs');
const path = require('path');

const ROOT   = path.resolve(__dirname, '..');
const STATIC = path.join(ROOT, 'static');
const WWW    = path.join(ROOT, 'www');

function copyRecursive(src, dest) {
  const stat = fs.statSync(src);
  if (stat.isDirectory()) {
    fs.mkdirSync(dest, { recursive: true });
    for (const entry of fs.readdirSync(src)) {
      copyRecursive(path.join(src, entry), path.join(dest, entry));
    }
  } else {
    fs.mkdirSync(path.dirname(dest), { recursive: true });
    fs.copyFileSync(src, dest);
  }
}

// Clean www/
fs.rmSync(WWW, { recursive: true, force: true });
fs.mkdirSync(WWW);

// Copy static/ → www/static/
copyRecursive(STATIC, path.join(WWW, 'static'));

// Hoist root-level files to www/
for (const file of ['index.html', 'manifest.json', 'sw.js']) {
  fs.copyFileSync(path.join(STATIC, file), path.join(WWW, file));
}

console.log('www/ built successfully');
