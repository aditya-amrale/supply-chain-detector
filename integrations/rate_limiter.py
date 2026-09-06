"""
Token bucket rate limiter for the FastAPI dashboard API.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


# ── Token bucket ──────────────────────────────────────────────────────────────

@dataclass
class TokenBucket:
    capacity:    float
    refill_rate: float
    tokens:      float = field(init=False)
    last_refill: float = field(init=False)

    def __post_init__(self):
        self.tokens      = float(self.capacity)
        self.last_refill = time.monotonic()

    def consume(self, cost: float = 1.0) -> bool:
        now     = time.monotonic()
        elapsed = now - self.last_refill

        self.tokens      = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False

    @property
    def wait_seconds(self) -> float:
        if self.tokens >= 1.0:
            return 0.0
        return (1.0 - self.tokens) / self.refill_rate


# ── Rate limiter ──────────────────────────────────────────────────────────────

class RateLimitExceeded(Exception):
    def __init__(self, ip: str, wait_seconds: float, scope: str = "per_ip"):
        self.ip           = ip
        self.wait_seconds = wait_seconds
        self.scope        = scope
        super().__init__(
            f"Rate limit exceeded [{scope}] for {ip}. "
            f"Retry in {wait_seconds:.1f}s."
        )


class RateLimiter:
    def __init__(
        self,
        per_ip_capacity:    float = 10.0,
        per_ip_refill:      float = 1.0,
        global_capacity:    float = 100.0,
        global_refill:      float = 10.0,
    ):
        self._per_ip_capacity = per_ip_capacity
        self._per_ip_refill   = per_ip_refill
        self._global          = TokenBucket(global_capacity, global_refill)
        self._buckets: dict[str, TokenBucket] = {}
        self._lock    = Lock()

        logger.info(
            f"RateLimiter initialized: "
            f"per-IP={per_ip_capacity} burst / {per_ip_refill}/s  "
            f"global={global_capacity} burst / {global_refill}/s"
        )

    def check(self, ip: str, cost: float = 1.0) -> None:
        with self._lock:
            if ip not in self._buckets:
                self._buckets[ip] = TokenBucket(
                    self._per_ip_capacity, self._per_ip_refill
                )

            ip_bucket = self._buckets[ip]

            if not self._global.consume(cost):
                wait = self._global.wait_seconds
                logger.warning(f"Global rate limit hit (requested by {ip})")
                raise RateLimitExceeded(ip, wait, scope="global")

            if not ip_bucket.consume(cost):
                wait = ip_bucket.wait_seconds
                logger.warning(f"Per-IP rate limit hit for {ip}")
                self._global.tokens = min(self._global.capacity, self._global.tokens + cost)
                raise RateLimitExceeded(ip, wait, scope="per_ip")

    def get_status(self, ip: str) -> dict:
        with self._lock:
            ip_bucket = self._buckets.get(ip)
            return {
                "ip":              ip,
                "ip_tokens":       round(ip_bucket.tokens, 2) if ip_bucket else self._per_ip_capacity,
                "ip_capacity":     self._per_ip_capacity,
                "global_tokens":   round(self._global.tokens, 2),
                "global_capacity": self._global.capacity,
            }

    def cleanup_stale(self, max_age_seconds: float = 3600.0) -> int:
        now    = time.monotonic()
        stale  = [
            ip for ip, bucket in self._buckets.items()
            if (now - bucket.last_refill) > max_age_seconds
        ]
        with self._lock:
            for ip in stale:
                del self._buckets[ip]
        if stale:
            logger.debug(f"Cleaned up {len(stale)} stale rate limit buckets")
        return len(stale)


# ── Scan cache ────────────────────────────────────────────────────────────────

@dataclass
class CacheEntry:
    value:      Any
    created_at: float = field(default_factory=time.monotonic)


class ScanCache:
    def __init__(self, ttl_seconds: float = 300.0, max_size: int = 1000):
        self.ttl      = ttl_seconds
        self.max_size = max_size
        self._cache: dict[str, CacheEntry] = {}
        self._lock  = Lock()
        self._hits  = 0
        self._misses= 0

    def _key(self, registry: str, name: str) -> str:
        return f"{registry.lower().strip()}:{name.lower().strip()}"

    def get(self, registry: str, name: str) -> Any | None:
        key = self._key(registry, name)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None
            if time.monotonic() - entry.created_at > self.ttl:
                del self._cache[key]
                self._misses += 1
                return None
            self._hits += 1
            logger.debug(f"Cache hit: {key}")
            return entry.value

    def set(self, registry: str, name: str, value: Any) -> None:
        key = self._key(registry, name)
        with self._lock:
            if len(self._cache) >= self.max_size:
                oldest = min(self._cache.items(), key=lambda kv: kv[1].created_at)
                del self._cache[oldest[0]]
                logger.debug(f"Cache evicted: {oldest[0]}")
            self._cache[key] = CacheEntry(value=value)

    def invalidate(self, registry: str, name: str) -> bool:
        key = self._key(registry, name)
        with self._lock:
            existed = key in self._cache
            self._cache.pop(key, None)
        return existed

    def clear(self) -> int:
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
        return count

    @property
    def stats(self) -> dict:
        with self._lock:
            total  = self._hits + self._misses
            return {
                "entries":    len(self._cache),
                "max_size":   self.max_size,
                "ttl":        self.ttl,
                "hits":       self._hits,
                "misses":     self._misses,
                "hit_rate":   round(self._hits / total, 3) if total > 0 else 0.0,
            }


# ── FastAPI middleware helper ─────────────────────────────────────────────────

def make_rate_limit_middleware(limiter: RateLimiter):
    async def middleware(request, call_next):
        try:
            client_ip = getattr(request.client, "host", "unknown")
            limiter.check(client_ip)
        except RateLimitExceeded as exc:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={
                    "error":       "rate_limit_exceeded",
                    "message":     str(exc),
                    "retry_after": round(exc.wait_seconds, 1),
                },
                headers={"Retry-After": str(int(exc.wait_seconds) + 1)},
            )
        return await call_next(request)

    return middleware


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rate limiter demo")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.demo:
        print("\n── Token bucket demo (per-IP limit: 5 burst, 1/s refill) ──────\n")
        limiter = RateLimiter(per_ip_capacity=5, per_ip_refill=1.0,
                              global_capacity=100, global_refill=10.0)
        cache   = ScanCache(ttl_seconds=10.0)

        for i in range(1, 9):
            try:
                limiter.check("192.168.1.1")
                status = limiter.get_status("192.168.1.1")
                print(f"  Request {i:>2}: ALLOWED  tokens={status['ip_tokens']:.1f}")
            except RateLimitExceeded as e:
                print(f"  Request {i:>2}: BLOCKED  ({e.scope}, retry in {e.wait_seconds:.1f}s)")

        print("\n── Cache demo ──────────────────────────────────────────────────\n")
        cache.set("pypi", "requests", {"score": 5.0, "level": "low"})
        print(f"  GET requests   : {cache.get('pypi', 'requests')}")
        print(f"  GET numpy      : {cache.get('pypi', 'numpy')}")
        print(f"  Cache stats    : {cache.stats}")