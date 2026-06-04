"""Async orchestration for CROSS ruling LLM extraction.

Modeled on hts-api/src/hts/core/graph/classifier.py:
async + semaphore + chunked processing + periodic DB flush + tenacity retry.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from collections.abc import Callable
from typing import Any

from openai import AsyncOpenAI, APITimeoutError, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from hts_lora.extraction.db import (
    ensure_table,
    get_unextracted_rulings,
    mark_failed,
    upsert_extractions,
)
from hts_lora.extraction.parser import parse_extraction
from hts_lora.extraction.prompts import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)

# Shutdown flag for graceful SIGINT handling
_shutdown = False


def _handle_sigint(signum: int, frame: Any) -> None:
    global _shutdown
    _shutdown = True
    logger.info("Shutdown requested — finishing current chunk...")


@retry(
    retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=30),
)
async def _extract_one(
    ruling: dict,
    semaphore: asyncio.Semaphore,
    client: AsyncOpenAI,
    model: str,
) -> list[dict] | None:
    """Extract structured data from a single ruling via LLM call."""
    async with semaphore:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(ruling)},
            ],
            temperature=0.0,
            max_completion_tokens=10000,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason
        tokens = response.usage.total_tokens if response.usage else 0

        if finish_reason == "length":
            logger.warning(
                "Response truncated for %s (%d tariffs)",
                ruling["ruling_number"],
                len(ruling.get("tariffs") or []),
            )

        return parse_extraction(content, ruling, tokens)


async def _extract_batch_async(
    rulings: list[dict],
    concurrency: int,
    chunk_size: int,
    chunk_delay: float,
    flush_size: int,
    model: str,
    progress_callback: Callable | None,
) -> dict[str, int]:
    """Async batch extraction with chunked processing and periodic DB flush."""
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(concurrency)

    total = len(rulings)
    num_chunks = (total + chunk_size - 1) // chunk_size

    extracted = 0
    products = 0
    failed = 0
    tokens_total = 0
    pending_rows: list[dict] = []

    for chunk_idx in range(num_chunks):
        if _shutdown:
            logger.info("Shutdown flag set — stopping after chunk %d/%d", chunk_idx, num_chunks)
            break

        start = chunk_idx * chunk_size
        end = min(start + chunk_size, total)
        chunk = rulings[start:end]

        tasks = [_extract_one(r, semaphore, client, model) for r in chunk]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            ruling = chunk[i]
            ruling_number = ruling["ruling_number"]

            if isinstance(result, Exception):
                logger.error("Extraction failed for %s: %s", ruling_number, result)
                mark_failed(ruling_number, str(result), model)
                failed += 1
            elif result is None:
                mark_failed(ruling_number, "parse_failure", model)
                failed += 1
            else:
                pending_rows.extend(result)
                products += len(result)
                extracted += 1
                for row in result:
                    tokens_total += row.get("tokens_used", 0)

        # Periodic DB flush
        if len(pending_rows) >= flush_size:
            try:
                upsert_extractions(pending_rows, model)
                logger.debug("Flushed %d rows to DB", len(pending_rows))
            except Exception as e:
                logger.error("DB flush failed (%d rows): %s", len(pending_rows), e)
                failed += len(set(r["ruling_number"] for r in pending_rows))
            pending_rows = []

        if progress_callback:
            progress_callback(
                processed=extracted + failed,
                total=total,
                tokens=tokens_total,
                extracted=extracted,
                failed=failed,
            )

        # Delay between chunks (rate limiting)
        if chunk_idx < num_chunks - 1 and chunk_delay > 0:
            await asyncio.sleep(chunk_delay)

    # Final flush
    if pending_rows:
        try:
            upsert_extractions(pending_rows, model)
            logger.debug("Final flush: %d rows", len(pending_rows))
        except Exception as e:
            logger.error("Final DB flush failed (%d rows): %s", len(pending_rows), e)
            failed += len(set(r["ruling_number"] for r in pending_rows))

    return {
        "extracted": extracted,
        "products": products,
        "failed": failed,
        "tokens_total": tokens_total,
    }


def extract_all(
    concurrency: int = 100,
    chunk_size: int = 500,
    chunk_delay: float = 0.5,
    flush_size: int = 200,
    max_rulings: int | None = None,
    model: str = "gpt-5.4-nano",
    progress_callback: Callable | None = None,
) -> dict[str, Any]:
    """Run the full extraction pipeline.

    Sync wrapper around async engine — safe for CLI use.

    Returns summary dict with counts, tokens, cost estimate, and timing.
    """
    global _shutdown
    _shutdown = False

    # Register graceful shutdown handler
    prev_handler = signal.signal(signal.SIGINT, _handle_sigint)

    try:
        ensure_table()

        # Paginate to collect all unextracted rulings
        rulings: list[dict] = []
        page_size = 5000
        offset = 0
        while True:
            remaining_needed = (max_rulings - len(rulings)) if max_rulings else page_size
            fetch_size = min(page_size, remaining_needed)
            if fetch_size <= 0:
                break
            batch = get_unextracted_rulings(limit=fetch_size, offset=offset)
            if not batch:
                break
            rulings.extend(batch)
            offset += len(batch)
            if max_rulings and len(rulings) >= max_rulings:
                rulings = rulings[:max_rulings]
                break

        if not rulings:
            return {
                "extracted": 0,
                "products": 0,
                "failed": 0,
                "tokens_total": 0,
                "cost_estimate": 0.0,
                "time_seconds": 0.0,
                "total_candidates": 0,
            }

        t0 = time.time()

        result = asyncio.run(
            _extract_batch_async(
                rulings=rulings,
                concurrency=concurrency,
                chunk_size=chunk_size,
                chunk_delay=chunk_delay,
                flush_size=flush_size,
                model=model,
                progress_callback=progress_callback,
            )
        )

        elapsed = time.time() - t0
        tokens = result["tokens_total"]

        # Cost estimate for gpt-5.4-nano: $0.20/M input, $1.25/M output
        # Approximate split: ~80% input, ~20% output
        input_tokens = int(tokens * 0.8)
        output_tokens = int(tokens * 0.2)
        cost = (input_tokens * 0.20 / 1_000_000) + (output_tokens * 1.25 / 1_000_000)

        return {
            "extracted": result["extracted"],
            "products": result["products"],
            "failed": result["failed"],
            "tokens_total": tokens,
            "cost_estimate": round(cost, 2),
            "time_seconds": round(elapsed, 1),
            "total_candidates": len(rulings),
        }
    finally:
        signal.signal(signal.SIGINT, prev_handler)
