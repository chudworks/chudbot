(() => {
  const USER_PREFIX = 'user:';
  const ANONYMOUS_ID_KEY = '__vibe_anonymous_id_v1';
  const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
  let memoryAnonymousId;

  const requestIdentity = async () => {
    const response = await fetch('/__vibe/api/v1/identity', {
      credentials: 'same-origin',
    });
    const body = await response.json();
    if (!response.ok) {
      throw Object.assign(
        new Error(body?.error?.message || 'Vibe identity failed'),
        { code: body?.error?.code },
      );
    }
    return body;
  };

  const anonymousId = () => {
    if (memoryAnonymousId) return memoryAnonymousId;
    try {
      const stored = localStorage.getItem(ANONYMOUS_ID_KEY);
      if (stored && UUID.test(stored)) return (memoryAnonymousId = stored);
    } catch (_) {
      // Storage can be unavailable in privacy-restricted browsing contexts.
    }
    memoryAnonymousId = crypto.randomUUID();
    try {
      localStorage.setItem(ANONYMOUS_ID_KEY, memoryAnonymousId);
    } catch (_) {
      // The identifier remains stable for this page lifetime.
    }
    return memoryAnonymousId;
  };

  const validName = (value) =>
    typeof value === 'string' &&
    value.length > 0 &&
    value.length <= 64 &&
    /^[A-Za-z0-9:_-]+$/.test(value);

  const assertRoomName = (name) => {
    if (!validName(name)) {
      throw new TypeError(
        'Vibe room names must be 1-64 ASCII letters, digits, colons, underscores, or hyphens.',
      );
    }
  };

  const assertLabel = (label) => {
    if (!validName(label) || label.startsWith(USER_PREFIX)) {
      throw new TypeError(
        'Vibe event labels must be 1-64 safe characters and cannot start with "user:".',
      );
    }
  };

  const assertStateKey = (key) => {
    if (!validName(key)) {
      throw new TypeError('Vibe user-state keys must be 1-64 safe characters.');
    }
  };

  const jsonValue = (value) => {
    const encoded = JSON.stringify(value);
    if (encoded === undefined) {
      throw new TypeError('Vibe room values must be JSON serializable.');
    }
    return JSON.parse(encoded);
  };

  const publicUser = (user) =>
    Object.freeze({
      id: user.id,
      ...(user.username == null ? {} : { username: user.username }),
      displayName: user.displayName,
      avatarUrl: user.avatarUrl ?? null,
      anonymous: user.anonymous,
      states: Object.freeze({ ...(user.states || {}) }),
    });

  const reportHandlerError = (error) => {
    queueMicrotask(() => {
      throw error;
    });
  };

  class RoomConnection {
    constructor(room, socket) {
      this._room = room;
      this._socket = socket;
      this._users = new Map();
      this._pending = new Map();
      this._requestId = 0;
      this._closed = false;
      this._welcome = new Promise((resolve, reject) => {
        this._resolveWelcome = resolve;
        this._rejectWelcome = reject;
      });
      socket.addEventListener('message', (event) => this._message(event));
      socket.addEventListener('close', () => this._close());
      socket.addEventListener('error', () => {
        if (!this._ready) this._failWelcome('connection_failed', 'The Vibe room connection failed.');
      });
    }

    get self() {
      const user = this._users.get(this._selfId);
      if (!user) throw new Error('The Vibe room connection is closed.');
      return user;
    }

    get users() {
      return Object.freeze(Array.from(this._users.values()));
    }

    async broadcast(event, label) {
      assertLabel(label);
      return this._send({ type: 'broadcast', label, event: jsonValue(event) });
    }

    async setUserState(key, value) {
      assertStateKey(key);
      return this._send({ type: 'set_state', key, value: jsonValue(value) });
    }

    async disconnect() {
      if (this._closed || this._socket.readyState === WebSocket.CLOSED) return;
      const closed = new Promise((resolve) =>
        this._socket.addEventListener('close', resolve, { once: true }),
      );
      this._socket.close(1000, 'client disconnect');
      await closed;
    }

    _join(name) {
      this._socket.addEventListener(
        'open',
        () => {
          this._socket.send(
            JSON.stringify({ type: 'join', room: name, anonymousId: anonymousId() }),
          );
        },
        { once: true },
      );
      return this._welcome;
    }

    _send(message) {
      if (!this._ready || this._closed || this._socket.readyState !== WebSocket.OPEN) {
        return Promise.reject(Object.assign(new Error('The Vibe room is not connected.'), { code: 'not_connected' }));
      }
      const id = String(++this._requestId);
      return new Promise((resolve, reject) => {
        this._pending.set(id, { resolve, reject });
        try {
          this._socket.send(JSON.stringify({ ...message, id }));
        } catch (error) {
          this._pending.delete(id);
          reject(error);
        }
      });
    }

    _message(event) {
      let message;
      try {
        message = JSON.parse(event.data);
      } catch (_) {
        return;
      }
      if (message.type === 'welcome') {
        this._users.clear();
        for (const user of message.users || []) {
          const value = publicUser(user);
          this._users.set(value.id, value);
        }
        this._selfId = message.selfId;
        this._ready = true;
        this._resolveWelcome(this);
        return;
      }
      if (message.type === 'ack') {
        const pending = this._pending.get(message.id);
        if (pending) {
          this._pending.delete(message.id);
          pending.resolve();
        }
        return;
      }
      if (message.type === 'error') {
        const error = Object.assign(new Error(message.message || 'Vibe room error'), {
          code: message.code,
        });
        const pending = this._pending.get(message.id);
        if (pending) {
          this._pending.delete(message.id);
          pending.reject(error);
        } else {
          this._failWelcome(message.code, message.message);
        }
        return;
      }
      if (!this._ready) return;
      if (message.type === 'event') {
        this._room._emit(message.label, message.event, publicUser(message.user));
      } else if (message.type === 'user:join') {
        const user = publicUser(message.user);
        this._users.set(user.id, user);
        this._room._emit('user:join', user);
      } else if (message.type === 'user:quit') {
        const user = publicUser(message.user);
        this._users.delete(user.id);
        this._room._emit('user:quit', user);
      } else if (message.type === 'user:state') {
        const user = publicUser(message.user);
        this._users.set(user.id, user);
        this._room._emitState(
          message.key,
          user,
          message.beforePresent ? message.before : undefined,
          message.after,
        );
      }
    }

    _failWelcome(code, message) {
      if (this._ready) return;
      this._rejectWelcome(Object.assign(new Error(message || 'Vibe room join failed.'), { code }));
      this._socket.close();
    }

    _close() {
      if (this._closed) return;
      this._closed = true;
      if (!this._ready) this._failWelcome('connection_closed', 'The Vibe room connection closed.');
      const error = Object.assign(new Error('The Vibe room connection closed.'), {
        code: 'connection_closed',
      });
      for (const pending of this._pending.values()) pending.reject(error);
      this._pending.clear();
      this._users.clear();
      this._room._connectionClosed(this);
    }
  }

  class Room {
    constructor(name) {
      assertRoomName(name);
      this.name = name;
      this._handlers = new Map();
      this._stateHandlers = new Map();
    }

    on(label, handler) {
      if (label !== 'user:join' && label !== 'user:quit') assertLabel(label);
      if (typeof handler !== 'function') throw new TypeError('A Vibe room handler must be a function.');
      return this._addHandler(this._handlers, label, handler);
    }

    onUserState(key, handler) {
      assertStateKey(key);
      if (typeof handler !== 'function') throw new TypeError('A Vibe room handler must be a function.');
      return this._addHandler(this._stateHandlers, key, handler);
    }

    async join() {
      if (this._active) {
        throw Object.assign(new Error('This Vibe room is already connected.'), {
          code: 'already_connected',
        });
      }
      const url = new URL('/__vibe/api/v1/rooms', location.href);
      url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const socket = new WebSocket(url);
      const connection = new RoomConnection(this, socket);
      this._active = connection;
      try {
        return await connection._join(this.name);
      } catch (error) {
        if (this._active === connection) this._active = undefined;
        throw error;
      }
    }

    _addHandler(map, key, handler) {
      let handlers = map.get(key);
      if (!handlers) map.set(key, (handlers = new Set()));
      handlers.add(handler);
      return () => {
        handlers.delete(handler);
        if (handlers.size === 0) map.delete(key);
      };
    }

    _emit(label, ...args) {
      for (const handler of this._handlers.get(label) || []) {
        try {
          handler(...args);
        } catch (error) {
          reportHandlerError(error);
        }
      }
    }

    _emitState(key, ...args) {
      for (const handler of this._stateHandlers.get(key) || []) {
        try {
          handler(...args);
        } catch (error) {
          reportHandlerError(error);
        }
      }
    }

    _connectionClosed(connection) {
      if (this._active === connection) this._active = undefined;
    }
  }

  Object.defineProperty(globalThis, 'vibe', {
    value: Object.freeze({
      version: '1',
      identity: requestIdentity,
      room: (name) => new Room(name),
    }),
    writable: false,
  });
})();
