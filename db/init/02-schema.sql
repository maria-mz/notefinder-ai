CREATE TABLE pages (
    source_file    TEXT NOT NULL,              -- path relative to the notes folder
    page_num       INTEGER NOT NULL CHECK (page_num >= 1),
    content_hash   TEXT NOT NULL,              -- SHA-256 of the rendered page image

    -- the extraction, and what it was derived from besides the page image
    transcript     TEXT NOT NULL,
    concepts       TEXT[] NOT NULL,
    summary        TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    claude_model   TEXT NOT NULL,

    -- the embedding, and what it was derived from
    embedded_text  TEXT NOT NULL,              -- exactly the text that was embedded
    embed_model    TEXT NOT NULL,
    embedding      vector NOT NULL,            -- no fixed dimension, so the model can change

    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (source_file, page_num)
);
