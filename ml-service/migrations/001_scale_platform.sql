BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS catalog_items (
    item_id text PRIMARY KEY,
    source_system text NOT NULL DEFAULT 'client',
    is_historical boolean NOT NULL,
    active boolean NOT NULL DEFAULT true,
    item_type text NOT NULL,
    gender text,
    brand text,
    sleeve text,
    provision text,
    pattern text,
    range_name text,
    fit text,
    fabric text,
    fashion text,
    lifecycle text,
    colour text,
    mrp numeric(12,2) NOT NULL CHECK (mrp > 0),
    image_url text,
    image_checksum text,
    embedding halfvec(512),
    embedding_model text,
    embedding_created_at timestamptz,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS catalog_filter_idx
    ON catalog_items (is_historical, active, item_type, gender, brand, mrp);
CREATE INDEX IF NOT EXISTS catalog_metadata_idx ON catalog_items USING gin (metadata);
CREATE INDEX IF NOT EXISTS catalog_embedding_hnsw_idx
    ON catalog_items USING hnsw (embedding halfvec_cosine_ops)
    WITH (m = 24, ef_construction = 128);

CREATE TABLE IF NOT EXISTS item_performance (
    performance_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    item_id text NOT NULL REFERENCES catalog_items(item_id) ON DELETE CASCADE,
    season text NOT NULL,
    channel text NOT NULL DEFAULT 'all',
    region text NOT NULL DEFAULT 'all',
    order_quantity integer NOT NULL CHECK (order_quantity >= 0),
    dispatch_quantity integer CHECK (dispatch_quantity >= 0),
    sales_quantity integer NOT NULL CHECK (sales_quantity >= 0),
    sell_through numeric(8,5),
    normalized_demand numeric(14,4) NOT NULL CHECK (normalized_demand >= 0),
    stockout_days integer,
    markdown_rate numeric(8,5),
    gross_margin numeric(14,4),
    season_end date,
    quality_flags text[] NOT NULL DEFAULT '{}',
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (item_id, season, channel, region)
);

CREATE INDEX IF NOT EXISTS performance_item_idx ON item_performance (item_id, season_end DESC);
CREATE INDEX IF NOT EXISTS performance_hierarchy_idx ON item_performance (season, channel, region);

CREATE TABLE IF NOT EXISTS recommendation_runs (
    recommendation_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id text NOT NULL UNIQUE,
    upcoming_item_id text NOT NULL,
    model_version text NOT NULL,
    status text NOT NULL CHECK (status IN ('succeeded', 'failed', 'approved', 'overridden')),
    request jsonb NOT NULL,
    response jsonb,
    latency_ms integer,
    planner_id text,
    approved_quantity integer,
    error text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS recommendation_item_idx
    ON recommendation_runs (upcoming_item_id, created_at DESC);

CREATE TABLE IF NOT EXISTS similarity_feedback (
    feedback_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    upcoming_item_id text NOT NULL,
    historical_item_id text NOT NULL REFERENCES catalog_items(item_id),
    recommendation_id uuid REFERENCES recommendation_runs(recommendation_id),
    accepted boolean NOT NULL,
    relevance smallint CHECK (relevance BETWEEN 0 AND 4),
    planner_id text,
    notes text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS feedback_pair_idx
    ON similarity_feedback (upcoming_item_id, historical_item_id, created_at DESC);

CREATE TABLE IF NOT EXISTS batch_jobs (
    job_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    job_type text NOT NULL CHECK (job_type IN ('recommendation_batch', 'catalog_ingestion', 'embedding_refresh')),
    status text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'succeeded', 'failed')),
    progress smallint NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 100),
    payload jsonb NOT NULL,
    result jsonb,
    error text,
    attempts smallint NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    finished_at timestamptz,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS batch_jobs_queue_idx ON batch_jobs (status, created_at)
    WHERE status = 'queued';

CREATE TABLE IF NOT EXISTS model_registry (
    model_version text PRIMARY KEY,
    model_type text NOT NULL,
    artifact_uri text NOT NULL,
    training_cutoff date NOT NULL,
    metrics jsonb NOT NULL,
    feature_schema jsonb NOT NULL,
    status text NOT NULL CHECK (status IN ('candidate', 'shadow', 'active', 'retired')),
    created_at timestamptz NOT NULL DEFAULT now(),
    activated_at timestamptz
);

CREATE UNIQUE INDEX IF NOT EXISTS one_active_model_per_type
    ON model_registry (model_type) WHERE status = 'active';

COMMIT;
