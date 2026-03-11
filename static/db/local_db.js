/**
 * local_db.js
 * IndexedDB wrapper for offline assessment storage.
 *
 * DB name    : MovementScreenDB
 * Version    : 1
 * Object store: assessments
 *   - keyPath      : id  (autoIncrement)
 *   - index recorded_at (not unique)
 *   - index synced      (not unique)
 *
 * All public functions return Promises.
 * Raw IDBRequest events are wrapped manually — no async/await over IDB calls.
 */

const DB_NAME    = 'MovementScreenDB';
const DB_VERSION = 1;
const STORE      = 'assessments';

// ---------------------------------------------------------------------------
// openDB
// ---------------------------------------------------------------------------

/** @type {IDBDatabase|null} */
let _db = null;

/**
 * Open (or create) the IndexedDB database.
 * The resolved IDBDatabase instance is cached for subsequent calls.
 *
 * @returns {Promise<IDBDatabase>}
 */
export function openDB() {
  if (_db) return Promise.resolve(_db);

  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);

    req.onupgradeneeded = (event) => {
      const db = event.target.result;

      if (!db.objectStoreNames.contains(STORE)) {
        const store = db.createObjectStore(STORE, { keyPath: 'id', autoIncrement: true });
        store.createIndex('recorded_at', 'recorded_at', { unique: false });
        store.createIndex('synced',      'synced',      { unique: false });
      }
    };

    req.onsuccess = (event) => {
      _db = event.target.result;
      resolve(_db);
    };

    req.onerror = (event) => {
      reject(event.target.error);
    };
  });
}

// ---------------------------------------------------------------------------
// saveAssessment
// ---------------------------------------------------------------------------

/**
 * Save a new assessment record and return its auto-assigned id.
 *
 * Expected shape of `data`:
 * {
 *   screen_type, camera_angle, lead_side,
 *   frame_count, worst_severity, has_findings,
 *   recorded_at,   // ISO string
 *   screen_name,
 *   findings,      // array
 *   stats,         // array
 *   synced,        // boolean
 *   server_id,     // string|null
 * }
 *
 * @param {Object} data
 * @returns {Promise<number>} the new record id
 */
export function saveAssessment(data) {
  return openDB().then((db) => {
    return new Promise((resolve, reject) => {
      const tx  = db.transaction(STORE, 'readwrite');
      const req = tx.objectStore(STORE).add(data);

      req.onsuccess = (event) => {
        resolve(event.target.result);
      };

      req.onerror = (event) => {
        reject(event.target.error);
      };
    });
  });
}

// ---------------------------------------------------------------------------
// getAssessments
// ---------------------------------------------------------------------------

/**
 * Return up to `limit` assessments sorted by recorded_at descending.
 *
 * IndexedDB cursors iterate in ascending key order by default; we open the
 * recorded_at index with direction 'prev' to get newest-first.
 *
 * @param {number} [limit=50]
 * @returns {Promise<Object[]>}
 */
export function getAssessments(limit = 50) {
  return openDB().then((db) => {
    return new Promise((resolve, reject) => {
      const tx      = db.transaction(STORE, 'readonly');
      const index   = tx.objectStore(STORE).index('recorded_at');
      const results = [];

      const req = index.openCursor(null, 'prev');

      req.onsuccess = (event) => {
        const cursor = event.target.result;
        if (cursor && results.length < limit) {
          results.push(cursor.value);
          cursor.continue();
        } else {
          resolve(results);
        }
      };

      req.onerror = (event) => {
        reject(event.target.error);
      };
    });
  });
}

// ---------------------------------------------------------------------------
// getAssessment
// ---------------------------------------------------------------------------

/**
 * Return a single assessment by id, or null if not found.
 *
 * @param {number} id
 * @returns {Promise<Object|null>}
 */
export function getAssessment(id) {
  return openDB().then((db) => {
    return new Promise((resolve, reject) => {
      const tx  = db.transaction(STORE, 'readonly');
      const req = tx.objectStore(STORE).get(id);

      req.onsuccess = (event) => {
        resolve(event.target.result ?? null);
      };

      req.onerror = (event) => {
        reject(event.target.error);
      };
    });
  });
}

// ---------------------------------------------------------------------------
// markSynced
// ---------------------------------------------------------------------------

/**
 * Mark an assessment as synced and record the server-assigned id.
 *
 * @param {number} id       - local IndexedDB id
 * @param {string} serverId - server-side id returned after sync
 * @returns {Promise<void>}
 */
export function markSynced(id, serverId) {
  return openDB().then((db) => {
    return new Promise((resolve, reject) => {
      const tx    = db.transaction(STORE, 'readwrite');
      const store = tx.objectStore(STORE);
      const getReq = store.get(id);

      getReq.onsuccess = (event) => {
        const record = event.target.result;
        if (!record) {
          resolve();
          return;
        }
        record.synced    = true;
        record.server_id = serverId;

        const putReq = store.put(record);

        putReq.onsuccess = () => {
          resolve();
        };

        putReq.onerror = (putEvent) => {
          reject(putEvent.target.error);
        };
      };

      getReq.onerror = (event) => {
        reject(event.target.error);
      };
    });
  });
}

// ---------------------------------------------------------------------------
// getUnsynced
// ---------------------------------------------------------------------------

/**
 * Return all assessments where synced === false.
 *
 * Uses the synced index with a key range matching boolean false.
 * IndexedDB stores booleans as-is and IDBKeyRange works with them.
 *
 * @returns {Promise<Object[]>}
 */
export function getUnsynced() {
  return openDB().then((db) => {
    return new Promise((resolve, reject) => {
      const tx      = db.transaction(STORE, 'readonly');
      const index   = tx.objectStore(STORE).index('synced');
      const results = [];

      // IDBKeyRange.only(false) matches records where synced === false
      const range = IDBKeyRange.only(false);
      const req   = index.openCursor(range);

      req.onsuccess = (event) => {
        const cursor = event.target.result;
        if (cursor) {
          results.push(cursor.value);
          cursor.continue();
        } else {
          resolve(results);
        }
      };

      req.onerror = (event) => {
        reject(event.target.error);
      };
    });
  });
}
