CREATE TABLE vibe_collection_documents (
    site_id UUID NOT NULL REFERENCES vibe_sites(id) ON DELETE CASCADE,
    collection TEXT NOT NULL,
    id UUID NOT NULL,
    document JSONB NOT NULL CHECK (jsonb_typeof(document) = 'object'),
    inserted_by TEXT NOT NULL,
    inserted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_by TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (site_id, collection, id)
);
