"""Database operations for CROSS ruling extraction pipeline.

Connects directly to the hts-api Supabase PostgreSQL database.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import psycopg

logger = logging.getLogger(__name__)

_connection: psycopg.Connection | None = None

_CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS cross_ruling_extractions (
    ruling_number   TEXT NOT NULL,
    product_idx     INTEGER NOT NULL DEFAULT 0,
    description     TEXT NOT NULL,
    hts_code        TEXT,
    product_use     TEXT,
    materials       TEXT,
    reasoning       TEXT,
    hts_text        TEXT,
    country         TEXT,
    extraction_model TEXT DEFAULT 'gpt-5.4-nano',
    tokens_used     INTEGER,
    extracted_at    TIMESTAMPTZ DEFAULT now(),
    raw_response    JSONB,
    PRIMARY KEY (ruling_number, product_idx)
);
"""

_UPSERT_SQL = """\
INSERT INTO cross_ruling_extractions
    (ruling_number, product_idx, description, hts_code, product_use,
     materials, reasoning, hts_text, country, extraction_model,
     tokens_used, raw_response)
VALUES
    (%(ruling_number)s, %(product_idx)s, %(description)s, %(hts_code)s,
     %(product_use)s, %(materials)s, %(reasoning)s, %(hts_text)s,
     %(country)s, %(extraction_model)s, %(tokens_used)s, %(raw_response)s)
ON CONFLICT (ruling_number, product_idx) DO UPDATE SET
    description = EXCLUDED.description,
    hts_code = EXCLUDED.hts_code,
    product_use = EXCLUDED.product_use,
    materials = EXCLUDED.materials,
    reasoning = EXCLUDED.reasoning,
    hts_text = EXCLUDED.hts_text,
    country = EXCLUDED.country,
    extraction_model = EXCLUDED.extraction_model,
    tokens_used = EXCLUDED.tokens_used,
    extracted_at = now(),
    raw_response = EXCLUDED.raw_response;
"""

_UNEXTRACTED_SQL = """\
SELECT cr.ruling_number, cr.subject, cr.tariffs, cr.ruling_date, crt.ruling_text
FROM cross_rulings cr
JOIN cross_ruling_texts crt USING (ruling_number)
LEFT JOIN (SELECT DISTINCT ruling_number FROM cross_ruling_extractions) cre
    ON cr.ruling_number = cre.ruling_number
WHERE cre.ruling_number IS NULL
    AND cr.categories ILIKE '%%classification%%'
    AND NOT cr.is_revoked
    AND array_length(cr.tariffs, 1) > 0
    AND length(crt.ruling_text) > 100
ORDER BY cr.ruling_date DESC NULLS LAST
LIMIT %s OFFSET %s;
"""

_COUNTS_SQL = """\
SELECT
    (SELECT count(*) FROM cross_rulings cr
     JOIN cross_ruling_texts crt USING (ruling_number)
     WHERE cr.categories ILIKE '%%classification%%'
       AND NOT cr.is_revoked
       AND array_length(cr.tariffs, 1) > 0
       AND length(crt.ruling_text) > 100) AS total,
    (SELECT count(DISTINCT ruling_number) FROM cross_ruling_extractions
     WHERE description != '__EXTRACTION_FAILED__') AS extracted,
    (SELECT count(DISTINCT ruling_number) FROM cross_ruling_extractions
     WHERE description = '__EXTRACTION_FAILED__') AS failed;
"""


def get_db_url() -> str:
    """Get database URL from environment or hts-api config."""
    for var in ("DATABASE_URL", "HTS_DATABASE_URL", "SUPABASE_DB_URL"):
        url = os.environ.get(var)
        if url:
            return url

    env_path = Path.home() / "Projects" / "hts-api" / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("DATABASE_URL="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
            if line.startswith("SUPABASE_DB_URL="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")

    raise RuntimeError(
        "No database URL found. Set DATABASE_URL env var or ensure "
        "~/Projects/hts-api/.env has DATABASE_URL defined."
    )


def get_connection() -> psycopg.Connection:
    """Get or create a lazy singleton database connection."""
    global _connection
    if _connection is None or _connection.closed:
        db_url = get_db_url()
        _connection = psycopg.connect(db_url, autocommit=True)
        logger.info("Connected to database")
    return _connection


def close_connection() -> None:
    """Close the singleton connection if open."""
    global _connection
    if _connection is not None and not _connection.closed:
        _connection.close()
        _connection = None


def ensure_table() -> None:
    """Create the extractions table if it doesn't exist."""
    conn = get_connection()
    conn.execute(_CREATE_TABLE_SQL)
    logger.info("Ensured cross_ruling_extractions table exists")


def get_unextracted_rulings(limit: int = 5000, offset: int = 0) -> list[dict]:
    """Fetch rulings that haven't been extracted yet."""
    conn = get_connection()
    cur = conn.execute(_UNEXTRACTED_SQL, (limit, offset))
    columns = [desc.name for desc in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]


def get_total_counts() -> dict:
    """Get extraction progress counts."""
    conn = get_connection()
    cur = conn.execute(_COUNTS_SQL)
    row = cur.fetchone()
    total, extracted, failed = row
    return {
        "total": total,
        "extracted": extracted,
        "failed": failed,
        "remaining": total - extracted - failed,
    }


def upsert_extractions(rows: list[dict], model: str = "gpt-5.4-nano") -> None:
    """Batch upsert extraction rows into the database."""
    if not rows:
        return
    conn = get_connection()
    for row in rows:
        params = {
            "ruling_number": row["ruling_number"],
            "product_idx": row["product_idx"],
            "description": row["description"],
            "hts_code": row.get("hts_code"),
            "product_use": row.get("product_use"),
            "materials": row.get("materials"),
            "reasoning": row.get("reasoning"),
            "hts_text": row.get("hts_text"),
            "country": row.get("country"),
            "extraction_model": model,
            "tokens_used": row.get("tokens_used"),
            "raw_response": json.dumps(row.get("raw_response")) if row.get("raw_response") else None,
        }
        conn.execute(_UPSERT_SQL, params)
    logger.debug("Upserted %d extraction rows", len(rows))


def mark_failed(ruling_number: str, error_msg: str, model: str = "gpt-5.4-nano") -> None:
    """Insert a sentinel row to mark a ruling as failed extraction."""
    conn = get_connection()
    params = {
        "ruling_number": ruling_number,
        "product_idx": 0,
        "description": "__EXTRACTION_FAILED__",
        "hts_code": None,
        "product_use": None,
        "materials": None,
        "reasoning": None,
        "hts_text": None,
        "country": None,
        "extraction_model": model,
        "tokens_used": 0,
        "raw_response": json.dumps({"error": error_msg}),
    }
    conn.execute(_UPSERT_SQL, params)


def export_to_jsonl(output_path: str, min_desc_length: int = 20) -> int:
    """Export successful extractions joined with ruling metadata to JSONL.

    Returns the number of records written.
    """
    conn = get_connection()
    cur = conn.execute(
        """\
        SELECT
            e.ruling_number, e.product_idx, e.description, e.hts_code,
            e.product_use, e.materials, e.reasoning, e.hts_text, e.country,
            e.tokens_used, e.extracted_at,
            cr.subject, cr.tariffs, cr.ruling_date, cr.ruling_url
        FROM cross_ruling_extractions e
        JOIN cross_rulings cr ON e.ruling_number = cr.ruling_number
        WHERE e.description != '__EXTRACTION_FAILED__'
          AND length(e.description) >= %s
        ORDER BY cr.ruling_date DESC NULLS LAST, e.ruling_number, e.product_idx;
        """,
        (min_desc_length,),
    )

    columns = [desc.name for desc in cur.description]
    count = 0
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for row in cur:
            record = dict(zip(columns, row))
            # Serialize dates and timestamps
            for key in ("ruling_date", "extracted_at"):
                if record.get(key) is not None:
                    record[key] = str(record[key])
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            count += 1

    return count


def get_validation_sample(sample_size: int = 100) -> list[dict]:
    """Get a random sample of extractions for validation."""
    conn = get_connection()
    cur = conn.execute(
        """\
        SELECT
            e.ruling_number, e.product_idx, e.description, e.hts_code,
            e.product_use, e.materials, e.reasoning, e.hts_text, e.country,
            cr.subject, cr.tariffs
        FROM cross_ruling_extractions e
        JOIN cross_rulings cr ON e.ruling_number = cr.ruling_number
        WHERE e.description != '__EXTRACTION_FAILED__'
        ORDER BY random()
        LIMIT %s;
        """,
        (sample_size,),
    )
    columns = [desc.name for desc in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]
