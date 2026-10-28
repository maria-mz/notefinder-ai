CREATE TABLE pages (
    source_file    TEXT NOT NULL,              -- path relative to notes/ folder
    page_num       INTEGER NOT NULL CHECK (page_num >= 1),
    content_hash   TEXT NOT NULL,              -- SHA-256 of the rendered page image

    status         TEXT NOT NULL DEFAULT 'pending'
                   CHECK (status IN ('pending', 'processing', 'done', 'failed')),
    attempts       INTEGER NOT NULL DEFAULT 0,
    last_error     TEXT,

    transcript     TEXT,
    concepts       TEXT[],
    summary        TEXT,
    prompt_version TEXT,
    embedding      vector(768),

    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (source_file, page_num),
    CONSTRAINT done_has_results CHECK (
        status <> 'done'
        OR (transcript IS NOT NULL AND concepts IS NOT NULL
            AND summary IS NOT NULL AND prompt_version IS NOT NULL
            AND embedding IS NOT NULL)
    )
);