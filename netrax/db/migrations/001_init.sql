-- NETRAX v2 schema
CREATE TABLE people (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE COLLATE NOCASE,
    role        TEXT NOT NULL DEFAULT 'allowed' CHECK (role IN ('owner', 'allowed', 'unknown')),
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    last_seen   REAL
);

CREATE TABLE face_embeddings (
    id          INTEGER PRIMARY KEY,
    person_id   INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    embedding   BLOB NOT NULL,          -- float32, L2-normalised (SFace: 128 values)
    dim         INTEGER NOT NULL,
    photo_path  TEXT,                   -- relative to the data dir
    quality     REAL,
    created_at  REAL NOT NULL
);
CREATE INDEX idx_emb_person ON face_embeddings(person_id);

CREATE TABLE events (
    id            INTEGER PRIMARY KEY,
    ts            REAL NOT NULL,
    type          TEXT NOT NULL,        -- seen | unknown | gesture | object | posture | system
    label         TEXT NOT NULL,        -- person name, gesture, object class, ...
    person_id     INTEGER REFERENCES people(id) ON DELETE SET NULL,
    track_id      INTEGER,
    confidence    REAL,
    data          TEXT,                 -- JSON
    snapshot_path TEXT                  -- relative to the data dir
);
CREATE INDEX idx_events_ts ON events(ts);
CREATE INDEX idx_events_type_ts ON events(type, ts);
