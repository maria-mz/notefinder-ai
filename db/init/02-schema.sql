CREATE TABLE pages (
    source_file    TEXT NOT NULL,              -- path relative to the notes folder
    page_num       INTEGER NOT NULL CHECK (page_num >= 1),
    content_hash   TEXT NOT NULL,              -- SHA-256 of the rendered page image

    transcript     TEXT NOT NULL,
    concepts       TEXT[] NOT NULL,
    summary        TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    embedding      vector(768) NOT NULL,

    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (source_file, page_num)
);