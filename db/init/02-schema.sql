CREATE TABLE pages (
    source_file    TEXT NOT NULL,              -- path relative to the notes folder
    page_num       INTEGER NOT NULL CHECK (page_num >= 1),
    content_hash   TEXT NOT NULL CHECK (content_hash <> ''),  -- SHA-256 of the page image

    -- the extraction, and what it was derived from besides the page image
    transcript     TEXT NOT NULL CHECK (transcript <> ''),
    concepts       TEXT[] NOT NULL CHECK (cardinality(concepts) > 0),
    summary        TEXT NOT NULL CHECK (summary <> ''),
    prompt_version TEXT NOT NULL CHECK (prompt_version <> ''),
    claude_model   TEXT NOT NULL CHECK (claude_model <> ''),

    -- the embedding, and what it was derived from
    embedded_text  TEXT NOT NULL CHECK (embedded_text <> ''),  -- the exact text embedded
    embed_model    TEXT NOT NULL CHECK (embed_model <> ''),
    embedding      vector NOT NULL,            -- any dimension, so the model can change

    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (source_file, page_num)
);
