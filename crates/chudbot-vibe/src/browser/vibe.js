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
      throw new TypeError('Vibe values must be JSON serializable.');
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

  const COLLECTION_NAME = /^[A-Za-z0-9_-]{1,64}$/;
  const isScalar = (value) =>
    value === null || ['string', 'number', 'boolean'].includes(typeof value);

  const assertCollectionName = (name) => {
    if (typeof name !== 'string' || !COLLECTION_NAME.test(name)) {
      throw new TypeError(
        'Vibe collection names must be 1-64 ASCII letters, digits, underscores, or hyphens.',
      );
    }
  };

  const assertColumn = (column) => {
    if (typeof column !== 'string' || !column || column.length > 128) {
      throw new TypeError('Vibe collection columns must be non-empty strings up to 128 characters.');
    }
  };

  const queryFilters = (filters) => {
    if (!filters || typeof filters !== 'object' || Array.isArray(filters)) {
      throw new TypeError('Vibe collection where() requires an object.');
    }
    const copy = jsonValue(filters);
    for (const [column, expected] of Object.entries(copy)) {
      assertColumn(column);
      const valid = Array.isArray(expected) ? expected.every(isScalar) : isScalar(expected);
      if (!valid) {
        throw new TypeError('Vibe collection filters accept scalar values or lists of scalar values.');
      }
    }
    return copy;
  };

  const requestCollection = async (name, payload) => {
    const response = await fetch(`/__vibe/api/v1/collections/${encodeURIComponent(name)}`, {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const body = await response.json().catch(() => null);
    if (!response.ok) {
      throw Object.assign(
        new Error(body?.error?.message || 'Vibe collection request failed.'),
        { code: body?.error?.code },
      );
    }
    return body;
  };

  class CollectionWatch {
    constructor(collection, filters, handler) {
      this._handler = handler;
      this._closed = false;
      const url = new URL(
        `/__vibe/api/v1/collections/${encodeURIComponent(collection)}/watch`,
        location.href,
      );
      url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
      this._socket = new WebSocket(url);
      this._ready = new Promise((resolve, reject) => {
        this._resolveReady = resolve;
        this._rejectReady = reject;
      });
      this._socket.addEventListener('open', () => {
        this._socket.send(JSON.stringify({ type: 'watch', where: filters }));
      });
      this._socket.addEventListener('message', (event) => this._message(event));
      this._socket.addEventListener('error', () => {
        if (!this._connected) this._rejectReady(this._error('connection_failed', 'The collection watch failed.'));
      });
      this._socket.addEventListener('close', () => {
        this._closed = true;
        if (!this._connected) this._rejectReady(this._error('connection_closed', 'The collection watch closed.'));
      });
    }

    async connect() {
      await this._ready;
      return this;
    }

    async disconnect() {
      if (this._closed || this._socket.readyState === WebSocket.CLOSED) return;
      const closed = new Promise((resolve) =>
        this._socket.addEventListener('close', resolve, { once: true }),
      );
      this._socket.close(1000, 'client disconnect');
      await closed;
    }

    _message(event) {
      let message;
      try {
        message = JSON.parse(event.data);
      } catch (_) {
        return;
      }
      if (message.type === 'ready') {
        this._connected = true;
        this._resolveReady(this);
      } else if (message.type === 'change') {
        try {
          this._handler(Object.freeze({
            type: message.changeType,
            document: message.document,
            before: message.before ?? null,
            after: message.after ?? null,
            matchesBefore: message.matchesBefore,
            matchesAfter: message.matchesAfter,
          }));
        } catch (error) {
          reportHandlerError(error);
        }
      } else if (message.type === 'error') {
        const error = this._error(message.code, message.message);
        if (!this._connected) this._rejectReady(error);
        else reportHandlerError(error);
      }
    }

    _error(code, message) {
      return Object.assign(new Error(message || 'Vibe collection watch failed.'), { code });
    }
  }

  class CollectionQuery {
    constructor(collection, state = {}) {
      this._collection = collection;
      this._state = {
        where: state.where || {},
        ...(state.limit == null ? {} : { limit: state.limit }),
        ...(state.offset == null ? {} : { offset: state.offset }),
        orderBy: state.orderBy || [],
        ...(state.select == null ? {} : { select: state.select }),
        distinct: state.distinct || false,
      };
    }

    where(filters) {
      return this._clone({ where: { ...this._state.where, ...queryFilters(filters) } });
    }

    limit(value) {
      this._assertInteger(value, 'limit');
      return this._clone({ limit: value });
    }

    offset(value) {
      this._assertInteger(value, 'offset');
      return this._clone({ offset: value });
    }

    orderBy(column, direction = 'asc') {
      assertColumn(column);
      if (direction !== 'asc' && direction !== 'desc') {
        throw new TypeError('Vibe collection order direction must be "asc" or "desc".');
      }
      return this._clone({
        orderBy: [...this._state.orderBy, { column, direction }],
      });
    }

    select(columns) {
      if (!Array.isArray(columns)) {
        throw new TypeError('Vibe collection select() requires an array of column names.');
      }
      columns.forEach(assertColumn);
      return this._clone({ select: [...columns] });
    }

    distinct(enabled = true) {
      if (typeof enabled !== 'boolean') {
        throw new TypeError('Vibe collection distinct() requires a boolean when supplied.');
      }
      return this._clone({ distinct: enabled });
    }

    find() {
      return this._execute('find');
    }

    delete() {
      return this._execute('delete');
    }

    deleteOne() {
      return this._execute('delete_one');
    }

    async count() {
      return (await this._execute('count')).count;
    }

    watch(handler) {
      if (typeof handler !== 'function') {
        throw new TypeError('Vibe collection watch() requires a handler function.');
      }
      return new CollectionWatch(this._collection, this._state.where, handler).connect();
    }

    _execute(action) {
      return requestCollection(this._collection, { action, query: this._state });
    }

    _clone(patch) {
      return new CollectionQuery(this._collection, { ...this._state, ...patch });
    }

    _assertInteger(value, name) {
      if (!Number.isSafeInteger(value) || value < 0) {
        throw new TypeError(`Vibe collection ${name} must be a non-negative safe integer.`);
      }
    }
  }

  class Collection extends CollectionQuery {
    constructor(name) {
      assertCollectionName(name);
      super(name);
      this.name = name;
    }

    put(document) {
      return requestCollection(this.name, { action: 'put', document: jsonValue(document) });
    }

    insert(document) {
      return requestCollection(this.name, { action: 'insert', document: jsonValue(document) });
    }

    update(document) {
      return requestCollection(this.name, { action: 'update', document: jsonValue(document) });
    }
  }

  Object.defineProperty(globalThis, 'vibe', {
    value: Object.freeze({
      version: '1',
      identity: requestIdentity,
      room: (name) => new Room(name),
      collection: (name) => new Collection(name),
    }),
    writable: false,
  });
})();
