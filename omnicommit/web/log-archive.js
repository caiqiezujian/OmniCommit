'use strict';

// The journal contains received JSON events, keyed by session and arrival order.
// Model and transport processing never wait for a storage transaction.
class MeetingLogArchive {
  constructor({onError = () => {}} = {}) {
    this.onError = onError;
    this.sessions = new Map();
    this.connection = null;
  }

  open() {
    if (this.connection) return this.connection;
    this.connection = new Promise((resolve, reject) => {
      const request = indexedDB.open('omnicommit-meeting-logs', 1);
      let blocked = false;
      request.onupgradeneeded = () => {
        request.result.createObjectStore('sessions', {keyPath: 'id'});
        request.result.createObjectStore('events');
      };
      request.onblocked = () => {
        blocked = true;
        reject(new Error('日志存储被其他页面占用'));
      };
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const db = request.result;
        if (blocked) {
          db.close();
          return;
        }
        db.onversionchange = () => {
          db.close();
          this.connection = null;
        };
        db.onclose = () => { this.connection = null; };
        resolve(db);
      };
    }).catch(error => {
      this.connection = null;
      throw error;
    });
    return this.connection;
  }

  // Resolve only after commit. Abort a whole batch if cloning or adding a row fails.
  transaction(db, stores, mode, operation) {
    return new Promise((resolve, reject) => {
      const tx = db.transaction(stores, mode);
      let request;
      tx.oncomplete = () => resolve(request?.result);
      tx.onabort = () => reject(tx.error || new Error('日志存储事务已中止'));
      try {
        request = operation(tx);
      } catch (error) {
        tx.abort();
        reject(error);
      }
    });
  }

  start(sourceKind, sourceName = '') {
    // getRandomValues also works on HTTP origins on an intranet.
    const id = Array.from(crypto.getRandomValues(new Uint8Array(16)),
      value => value.toString(16).padStart(2, '0')).join('');
    const info = {id, startedAt: new Date().toISOString(), sourceKind, sourceName};
    const state = {info, created: false, next: 0, pending: [], saving: null, error: null};
    this.sessions.set(id, state);
    state.saving = this.save(state);
    return id;
  }

  append(id, row) {
    const state = this.sessions.get(id);
    if (!state || state.error) return;
    // Corrections can mutate the transcript objects that were in an earlier event.
    const copy = typeof structuredClone === 'function' ? structuredClone(row) : JSON.parse(JSON.stringify(row));
    state.pending.push({sequence: state.next++, row: copy});
    if (!state.saving) state.saving = this.save(state);
  }

  async save(state) {
    try {
      const db = await this.open();
      do {
        const batch = state.pending;
        state.pending = [];
        const stores = state.created ? ['events'] : ['sessions', 'events'];
        await this.transaction(db, stores, 'readwrite', tx => {
          if (!state.created) tx.objectStore('sessions').add(state.info);
          const events = tx.objectStore('events');
          for (const entry of batch) events.add(entry.row, [state.info.id, entry.sequence]);
        });
        state.created = true;
      } while (state.pending.length);
    } catch (error) {
      state.error = error;
      state.pending = [];
      this.onError(error, state.info.id);
    } finally {
      state.saving = null;
    }
  }

  async flush(id) {
    const state = this.sessions.get(id);
    if (!state) return;
    await state.saving;
    if (state.error) throw state.error;
  }

  async list() {
    const db = await this.open();
    const records = await this.transaction(db, 'sessions', 'readonly', tx => tx.objectStore('sessions').getAll());
    return records.sort((a, b) => b.startedAt.localeCompare(a.startedAt) || b.id.localeCompare(a.id));
  }

  async read(id) {
    // A transaction that failed after earlier commits leaves a readable prefix.
    await this.sessions.get(id)?.saving;
    const db = await this.open();
    const range = IDBKeyRange.bound([id, 0], [id, Number.MAX_SAFE_INTEGER]);
    return this.transaction(db, 'events', 'readonly', tx => tx.objectStore('events').getAll(range));
  }
}
