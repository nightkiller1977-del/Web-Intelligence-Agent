# app/storage.py
import json
import logging
import os
import socket
import time
import uuid
from collections import OrderedDict
from typing import Dict, Any, List, Optional
from app.config import settings

logger = logging.getLogger("web-intelligence")

# Bound the in-memory operation cache so a long-lived local process cannot grow
# without limit. Web results stay queryable; input/reconciliation metadata is
# dropped first so the evicted bytes are the large ones.
OPERATION_CACHE_LIMIT = 500
# Idempotency reservations outlive evicted results: dropping one would let a
# retry of old work be accepted as new (and paid for again). Kept as a bounded
# FIFO tombstone independent of the operation payload cache.
IDEMPOTENCY_RESERVATION_LIMIT = 10000
IDEMPOTENCY_KEY_TTL_SECONDS = 86400
IDEMPOTENCY_ADMISSION_TTL_SECONDS = 30

# Claim a durable idempotency key together with a short admission lease. A
# process that exits before persisting the operation leaves no permanent
# tombstone: once the short lease expires, a retry can atomically reclaim it.
# A persisted operation always wins, even after the admission lease is gone.
_CLAIM_IDEMPOTENCY_LUA = """
local existing = redis.call('GET', KEYS[1])
if existing then
  if redis.call('HGET', KEYS[3], existing) then
    return existing
  end
  if redis.call('GET', KEYS[2]) == existing then
    return existing
  end
end
redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])
redis.call('SET', KEYS[2], ARGV[1], 'EX', ARGV[3])
return ''
"""

_RELEASE_IDEMPOTENCY_LUA = """
local existing = redis.call('GET', KEYS[1])
if not existing then
  return 0
end
if ARGV[1] ~= '' and existing ~= ARGV[1] then
  return 0
end
redis.call('DEL', KEYS[1], KEYS[2])
return 1
"""

# Atomic acquire of a concurrency slot. Expired slots are reclaimed first, so a
# crashed instance's slot frees itself instead of pinning the counter forever,
# and each slot carries its own expiry rather than sharing one global TTL.
_ACQUIRE_SLOT_LUA = """
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)
if redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[2]) then
  return 0
end
redis.call('ZADD', KEYS[1], now + tonumber(ARGV[1]), ARGV[3])
return 1
"""

# Atomic save of an operation together with its active-index membership, so a
# crash between the state write and the index update cannot leave a queued/
# running record the reconciliation scan never discovers.
_SAVE_OPERATION_LUA = """
redis.call('HSET', KEYS[1], ARGV[1], ARGV[2])
if ARGV[3] == '1' then
  redis.call('SADD', KEYS[2], ARGV[1])
else
  redis.call('SREM', KEYS[2], ARGV[1])
end
return 1
"""

_SAVE_ADMITTED_OPERATION_LUA = """
if redis.call('GET', KEYS[3]) ~= ARGV[1] then
  return 0
end
redis.call('HSET', KEYS[1], ARGV[1], ARGV[2])
redis.call('SADD', KEYS[2], ARGV[1])
if redis.call('GET', KEYS[4]) == ARGV[1] then
  redis.call('DEL', KEYS[4])
end
return 1
"""

# Atomic stale transition: only fail the operation if no live owner lease
# exists AND the stored operation is still queued/running. The state recheck
# closes the snapshot race where a worker completes and releases its lease
# after our list_operations() snapshot, which would otherwise let us overwrite
# a completed result with a stale failure.
_FAIL_IF_UNOWNED_LUA = """
if redis.call('EXISTS', KEYS[1]) ~= 0 then
  return 0
end
local raw = redis.call('HGET', KEYS[2], ARGV[1])
if not raw then
  return 0
end
local op = cjson.decode(raw)
if op['status'] ~= 'queued' and op['status'] ~= 'running' then
  return 0
end
redis.call('HSET', KEYS[2], ARGV[1], ARGV[2])
return 1
"""

# Atomic owner compare-and-{delete,expire}: act only if the stored owner is us.
_CAD_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""

# Atomic heartbeat: extend the ownership lease only if we still own it, and
# renew this operation's own concurrency slot only if that slot still exists.
# Returns 1 when both were renewed, 0 when we no longer own the lease, and -1
# when ownership is valid but the slot was already reaped (lost).
_RENEW_LUA = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then
  return 0
end
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[3]))
if redis.call('ZSCORE', KEYS[2], ARGV[2]) then
  local t = redis.call('TIME')
  local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
  redis.call('ZADD', KEYS[2], now + tonumber(ARGV[3]), ARGV[2])
  return 1
end
return -1
"""

# Atomic budget reservation: admit only if the shared total plus this
# operation's estimate stays within the ceiling, then reserve it in one step.
# Prevents concurrent replicas from all passing a non-atomic read-then-admit.
# The window TTL is set in the same atomic script, so a crash cannot leave a
# key that has crossed the limit with no expiry.
_RESERVE_SPEND_LUA = """
local total = tonumber(redis.call('GET', KEYS[1]) or '0')
local amount = tonumber(ARGV[1])
local op_id = ARGV[3]
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
-- The window marker identifies the current daily window. It is created with the
-- total and expires with it, so a reservation made in an earlier window can be
-- recognized and never subtracted from a later window's total.
local win = redis.call('GET', KEYS[3])
if redis.call('TTL', KEYS[1]) < 0 then
  win = tostring(now)
  redis.call('SET', KEYS[3], win, 'EX', 86400)
  redis.call('SET', KEYS[1], redis.call('GET', KEYS[1]) or '0', 'EX', 86400)
elseif not win then
  win = tostring(now)
  redis.call('SET', KEYS[3], win, 'EX', redis.call('TTL', KEYS[1]))
end
if op_id ~= '' then
  local existing = redis.call('HGET', KEYS[2], op_id)
  if existing then
    -- Idempotent within the same window: a retry must not reserve twice. A
    -- hold from an earlier window is stale and is replaced below.
    local sep = string.find(existing, ':', 1, true)
    if string.sub(existing, sep + 1) == win then
      return 1
    end
  end
end
if total + amount > tonumber(ARGV[2]) then
  return 0
end
redis.call('INCRBYFLOAT', KEYS[1], ARGV[1])
if op_id ~= '' then
  -- Create the entry before expiring the hash, otherwise the first reservation
  -- on a fresh database leaves an unexpiring hash behind.
  redis.call('HSET', KEYS[2], op_id, ARGV[1] .. ':' .. win)
  redis.call('EXPIRE', KEYS[2], 86400)
end
return 1
"""

# Atomic budget reconcile: replace a reservation with the observed cost in one
# step so freed capacity cannot be grabbed between a release and a charge.
_RECONCILE_SPEND_LUA = """
local total = tonumber(redis.call('GET', KEYS[1]) or '0')
local reserved = tonumber(ARGV[1])
local op_id = ARGV[4]
if op_id ~= '' then
  -- Retry-safe: if this operation's reservation is gone, its reconciliation
  -- already happened (or was released), so do not charge it twice.
  local existing = redis.call('HGET', KEYS[2], op_id)
  if not existing then
    return 0
  end
  local sep = string.find(existing, ':', 1, true)
  reserved = tonumber(string.sub(existing, 1, sep - 1))
  -- A hold from an earlier window must not be subtracted from the current
  -- window's total (the old total is already gone); just drop it.
  if redis.call('GET', KEYS[3]) ~= string.sub(existing, sep + 1) then
    redis.call('HDEL', KEYS[2], op_id)
    return 0
  end
end
local next_total = total - reserved + tonumber(ARGV[2])
if next_total < 0 then next_total = 0 end
if next_total > tonumber(ARGV[3]) then next_total = tonumber(ARGV[3]) end
redis.call('SET', KEYS[1], tostring(next_total), 'KEEPTTL')
if redis.call('TTL', KEYS[1]) < 0 then
  redis.call('EXPIRE', KEYS[1], 86400)
end
if op_id ~= '' then
  redis.call('HDEL', KEYS[2], op_id)
end
return 1
"""

# Atomic budget release: give back a reservation (or reconcile it down to the
# observed cost). Uses INCRBYFLOAT on the existing key so the daily window TTL
# is preserved; SET would reset the window to a fresh 24 hours on every
# release, so regular completions could keep the "daily" total from ever
# resetting.
_RELEASE_SPEND_LUA = """
local amount = tonumber(ARGV[1])
local op_id = ARGV[2]
if op_id ~= '' then
  -- Retry-safe: releasing an operation whose reservation is already gone is a
  -- no-op, so a lost response cannot subtract the hold twice.
  local existing = redis.call('HGET', KEYS[2], op_id)
  if not existing then
    return 0
  end
  local sep = string.find(existing, ':', 1, true)
  amount = tonumber(string.sub(existing, 1, sep - 1))
  -- A hold from an earlier window is stale: drop it without touching the
  -- current window's total, which no longer contains that reservation.
  if redis.call('GET', KEYS[3]) ~= string.sub(existing, sep + 1) then
    redis.call('HDEL', KEYS[2], op_id)
    return 0
  end
end
local next_total = tonumber(redis.call('GET', KEYS[1]) or '0') - amount
if next_total < 0 then next_total = 0 end
redis.call('SET', KEYS[1], tostring(next_total), 'KEEPTTL')
if redis.call('TTL', KEYS[1]) < 0 then
  redis.call('EXPIRE', KEYS[1], 86400)
end
if op_id ~= '' then
  redis.call('HDEL', KEYS[2], op_id)
end
return 1
"""
class StorageUnavailable(RuntimeError):
    """Raised at startup when a required durable backend cannot be reached.

    Selected via REDIS_REQUIRED so a remote deployment fails closed instead of
    silently serving from process-local memory, which would lose operations on
    the next restart or route a request to an instance with no state.
    """


def _new_instance_id() -> str:
    # Include a random component so two instances created in the same process
    # and millisecond (for example a peer in tests, or a fast restart) never
    # share an id - owner compare-and-set would otherwise treat them as one.
    return f"{socket.gethostname()}:{os.getpid()}:{int(time.time() * 1000)}:{uuid.uuid4().hex[:12]}"


class _InstanceIdentity:
    """Stable per-process identity used to scope liveness/lease claims."""

    def __init__(self):
        self.instance_id = _new_instance_id()

    def lease_id(self, lease_id: Optional[str] = None) -> str:
        return f"{self.instance_id}:{lease_id or 'default'}"

class BaseStorage:
    async def init(self):
        pass

    async def save_operation(self, op_id: str, data: Dict[str, Any]):
        raise NotImplementedError()

    async def save_admitted_operation(
        self, op_id: str, data: Dict[str, Any], idempotency_key: str
    ) -> bool:
        """Persist initial queued state only while admission still owns its key."""
        raise NotImplementedError()

    async def get_operation(self, op_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError()

    async def list_operations(self) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError()

    async def delete_operation(self, op_id: str):
        raise NotImplementedError()

    async def claim_idempotency_key(self, key: str, op_id: str) -> Optional[str]:
        """Atomically claim an idempotency key for op_id.
        Returns None on success, or the existing op_id if already claimed."""
        raise NotImplementedError()

    async def release_idempotency_key(self, key: str, op_id: Optional[str] = None) -> bool:
        """Release a previously claimed idempotency key.
        When op_id is provided, only releases keys still mapped to that operation."""
        raise NotImplementedError()

    async def claim_operation_id(self, op_id: str, idempotency_key: str) -> bool:
        """Atomically reserve operation_id for an idempotency key."""
        raise NotImplementedError()

    async def release_operation_id(self, op_id: str, idempotency_key: Optional[str] = None) -> bool:
        """Release a previously reserved operation_id."""
        raise NotImplementedError()

    async def push_progress_event(self, op_id: str, event: Dict[str, Any]):
        raise NotImplementedError()

    async def get_progress_events(self, op_id: str) -> List[Dict[str, Any]]:
        raise NotImplementedError()

    async def get_progress_events_after(self, op_id: str, cursor: str | None):
        raise NotImplementedError()

    async def mark_stale_operations(self):
        raise NotImplementedError()

    async def acquire_concurrency_slot(self, op_id: str) -> bool:
        """Reserve a service-wide concurrent-operation slot for op_id."""
        raise NotImplementedError()

    async def release_concurrency_slot(self, op_id: str) -> None:
        """Release the slot reserved for op_id."""
        raise NotImplementedError()

    async def begin_operation(self, op_id: str) -> bool:
        """Record that this instance is now running op_id. False if another
        live instance already owns it."""
        raise NotImplementedError()

    async def touch_operation(self, op_id: str) -> None:
        """Heartbeat this instance's ownership of op_id."""
        raise NotImplementedError()

    async def release_operation_lease(self, op_id: str) -> None:
        """Drop this instance's ownership lease for a finished operation."""
        raise NotImplementedError()

    async def add_daily_spend(self, amount_usd: float) -> None:
        """Atomically add an estimated spend amount to the shared daily total."""
        raise NotImplementedError()

    async def get_daily_spend(self) -> float:
        """Return the shared daily spend estimate, or 0 when unavailable."""
        raise NotImplementedError()

    async def reserve_daily_spend(self, amount_usd: float, limit_usd: float) -> bool:
        """Atomically reserve budget if total+amount stays within limit."""
        raise NotImplementedError()

    async def release_daily_spend(self, amount_usd: float) -> None:
        """Give back a reservation (or reconcile it to the observed cost)."""
        raise NotImplementedError()

    async def reconcile_daily_spend(self, reserved_usd: float, actual_usd: float, limit_usd: float) -> None:
        """Atomically replace a reservation with the observed cost, clamped to
        the daily ceiling so a large actual cost cannot push the shared total
        past the limit and silently overspend."""
        raise NotImplementedError()

class InMemoryStorage(BaseStorage):
    def __init__(self):
        # OrderedDict keyed by canonical op_id. Ownership leases live in a
        # separate mapping so a caller-supplied operation ID can never collide
        # with internal metadata.
        self.operations: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.events: Dict[str, List[Dict[str, Any]]] = {}
        self.idempotency_keys: "OrderedDict[str, str]" = OrderedDict()
        # Operation-ID claims are tombstones that must outlive evicted result
        # payloads too, so an evicted id cannot be silently reused by a
        # different idempotency key. Bounded like the reservation map.
        self.operation_claims: "OrderedDict[str, str]" = OrderedDict()
        self.operation_owners: Dict[str, str] = {}
        self.instance = _InstanceIdentity()
        # Identifiable slots (not a bare counter) so each release frees exactly
        # the slot it reserved.
        self._concurrency_slots: set[str] = set()
        self._daily_spend_usd = 0.0
        self._spend_window_started = time.time()
        # Per-operation reservations, so a moved retry cannot double-charge.
        self._spend_reservations: Dict[str, float] = {}

    def _evict_if_needed(self):
        # Evict oldest terminal operations first, and never evict queued/running
        # work: dropping a live operation would make it unqueryable while it is
        # still executing. Reservations are retained (see below).
        if len(self.operations) <= OPERATION_CACHE_LIMIT:
            return
        for evicted_key in [
            key for key, op in self.operations.items()
            if op.get("status") not in ("queued", "running")
        ]:
            if len(self.operations) <= OPERATION_CACHE_LIMIT:
                break
            self.operations.pop(evicted_key, None)
            self.events.pop(evicted_key, None)
            self.operation_owners.pop(evicted_key, None)
            # operation_claims and idempotency_keys are deliberately retained as
            # tombstones (bounded separately) so an evicted operationId cannot
            # be reused by an unrelated idempotency key.
            # Idempotency reservations are intentionally retained so a retry of
            # evicted work still resolves to its original operation instead of
            # being accepted (and paid for) as new work.
            logger.info("Evicted operation %s from the in-memory cache (limit %d)", evicted_key, OPERATION_CACHE_LIMIT)

    def _evict_idempotency_if_needed(self):
        while len(self.idempotency_keys) > IDEMPOTENCY_RESERVATION_LIMIT:
            self.idempotency_keys.popitem(last=False)
        while len(self.operation_claims) > IDEMPOTENCY_RESERVATION_LIMIT:
            self.operation_claims.popitem(last=False)

    async def save_operation(self, op_id: str, data: Dict[str, Any]):
        existing = self.operations.get(op_id)
        new_data = dict(existing) if existing else {}
        new_data.update(data)
        self.operations[op_id] = new_data
        self.operations.move_to_end(op_id)
        self._evict_if_needed()

    async def save_admitted_operation(
        self, op_id: str, data: Dict[str, Any], idempotency_key: str
    ) -> bool:
        if self.idempotency_keys.get(idempotency_key) != op_id:
            return False
        await self.save_operation(op_id, data)
        return True

    async def get_operation(self, op_id: str) -> Optional[Dict[str, Any]]:
        return self.operations.get(op_id)

    async def list_operations(self) -> Dict[str, Dict[str, Any]]:
        return self.operations

    async def delete_operation(self, op_id: str):
        self.operations.pop(op_id, None)
        self.events.pop(op_id, None)
        self.operation_owners.pop(op_id, None)
        # Explicit deletion clears the tombstone too; only bounded *eviction*
        # retains it.
        self.operation_claims.pop(op_id, None)
        for key, existing_op_id in list(self.idempotency_keys.items()):
            if existing_op_id == op_id:
                self.idempotency_keys.pop(key, None)

    async def claim_idempotency_key(self, key: str, op_id: str) -> Optional[str]:
        existing = self.idempotency_keys.get(key)
        if existing:
            return existing
        self.idempotency_keys[key] = op_id
        self.idempotency_keys.move_to_end(key)
        self._evict_idempotency_if_needed()
        return None

    async def release_idempotency_key(self, key: str, op_id: Optional[str] = None) -> bool:
        existing = self.idempotency_keys.get(key)
        if not existing or (op_id is not None and existing != op_id):
            return False
        self.idempotency_keys.pop(key, None)
        return True

    async def claim_operation_id(self, op_id: str, idempotency_key: str) -> bool:
        existing_claim = self.operation_claims.get(op_id)
        if existing_claim:
            return existing_claim == idempotency_key

        existing_operation = self.operations.get(op_id)
        if existing_operation:
            existing_key = existing_operation.get("idempotency_key")
            if existing_key and existing_key != idempotency_key:
                return False

        self.operation_claims[op_id] = idempotency_key
        self.operation_claims.move_to_end(op_id)
        self._evict_idempotency_if_needed()
        return True

    async def release_operation_id(self, op_id: str, idempotency_key: Optional[str] = None) -> bool:
        existing_claim = self.operation_claims.get(op_id)
        if not existing_claim or (idempotency_key is not None and existing_claim != idempotency_key):
            return False
        self.operation_claims.pop(op_id, None)
        return True

    async def push_progress_event(self, op_id: str, event: Dict[str, Any]):
        if op_id not in self.events:
            self.events[op_id] = []
        self.events[op_id].append(event)

    async def get_progress_events(self, op_id: str) -> List[Dict[str, Any]]:
        return self.events.get(op_id, [])

    async def get_progress_events_after(self, op_id: str, cursor: str | None):
        events = self.events.get(op_id, [])
        start = int(cursor or "0")
        return [(str(index + 1), event) for index, event in enumerate(events[start:], start)]

    async def mark_stale_operations(self):
        # Single-process backend: every queued/running operation belongs to this
        # process, so anything still live and unowned at startup was abandoned
        # by a restart. Operations with a live in-process owner are preserved:
        # unlike Redis there is no cross-instance lease to consult, so a
        # recurring scan must not fail work this same process is still running.
        for op_id, op in list(self.operations.items()):
            if op.get("status") in ("queued", "running") and op_id not in self.operation_owners:
                op["status"] = "failed"
                op["error"] = {"code": "STALE_OPERATION", "message": "Operation was abandoned after a service restart.", "retryable": True}
                logger.warning("Marked stale operation %s as failed", op_id)

    async def acquire_concurrency_slot(self, op_id: str) -> bool:
        # In-memory backend is single-process, so this set is the whole service.
        # It enforces the same MAX_CONCURRENT_OPS ceiling the shared Redis slot
        # set does, just without cross-instance visibility.
        if len(self._concurrency_slots) >= settings.MAX_CONCURRENT_OPS:
            return False
        self._concurrency_slots.add(self.instance.lease_id(op_id))
        return True

    async def release_concurrency_slot(self, op_id: str) -> None:
        self._concurrency_slots.discard(self.instance.lease_id(op_id))

    async def begin_operation(self, op_id: str) -> bool:
        # Honor the exclusivity contract (mirrors Redis SET NX): any live owner
        # must win. Without this check, an evicted idempotency reservation could
        # let a retry start a duplicate task alongside a still-running operation.
        if op_id in self.operation_owners:
            return False
        self.operation_owners[op_id] = self.instance.instance_id
        return True

    async def touch_operation(self, op_id: str) -> None:
        return None

    async def release_operation_lease(self, op_id: str) -> None:
        # Only the owner may release, so a late finisher cannot clear a lease a
        # newer owner has since taken.
        if self.operation_owners.get(op_id) == self.instance.instance_id:
            self.operation_owners.pop(op_id, None)

    def _reset_spend_window_if_needed(self):
        now = time.time()
        if now - self._spend_window_started >= 86400:
            self._daily_spend_usd = 0.0
            self._spend_reservations.clear()
            self._spend_window_started = now

    async def add_daily_spend(self, amount_usd: float) -> None:
        self._reset_spend_window_if_needed()
        self._daily_spend_usd += float(amount_usd)

    async def get_daily_spend(self) -> float:
        self._reset_spend_window_if_needed()
        return self._daily_spend_usd

    async def reserve_daily_spend(self, amount_usd: float, limit_usd: float, op_id: str = "") -> bool:
        self._reset_spend_window_if_needed()
        if op_id and op_id in self._spend_reservations:
            return True
        if self._daily_spend_usd + amount_usd > limit_usd:
            return False
        self._daily_spend_usd += float(amount_usd)
        if op_id:
            self._spend_reservations[op_id] = float(amount_usd)
        return True

    async def release_daily_spend(self, amount_usd: float, op_id: str = "") -> None:
        self._reset_spend_window_if_needed()
        amount = float(amount_usd)
        if op_id:
            if op_id not in self._spend_reservations:
                return
            amount = self._spend_reservations.pop(op_id)
        self._daily_spend_usd = max(0.0, self._daily_spend_usd - amount)

    async def reconcile_daily_spend(self, reserved_usd: float, actual_usd: float, limit_usd: float, op_id: str = "") -> None:
        self._reset_spend_window_if_needed()
        reserved = float(reserved_usd)
        if op_id:
            if op_id not in self._spend_reservations:
                return
            reserved = self._spend_reservations.pop(op_id)
        reconciled = self._daily_spend_usd - reserved + float(actual_usd)
        self._daily_spend_usd = max(0.0, min(float(limit_usd), reconciled))

class RedisStorage(BaseStorage):
    def __init__(self):
        self.redis = None
        self.fallback = InMemoryStorage()
        self.degraded = False
        self.instance = _InstanceIdentity()

    async def init(self):
        if not settings.REDIS_URL:
            if settings.REDIS_REQUIRED:
                raise StorageUnavailable(
                    "STORAGE_BACKEND=redis with REDIS_REQUIRED=true but REDIS_URL is not configured."
                )
            logger.warning("REDIS_URL is not configured; falling back to in-memory storage.")
            self.degraded = True
            return

        try:
            import redis.asyncio as aioredis
            self.redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
            await self.redis.ping()
            logger.info("Connected to Redis storage backend")
        except Exception as exc:
            if settings.REDIS_REQUIRED:
                raise StorageUnavailable(f"Redis is required but unreachable: {exc}") from exc
            logger.exception("Unable to initialize Redis; falling back to in-memory storage.")
            self.degraded = True

    async def save_operation(self, op_id: str, data: Dict[str, Any]):
        if self.degraded:
            return await self.fallback.save_operation(op_id, data)
        existing = await self.get_operation(op_id)
        new_data = dict(existing) if existing else {}
        new_data.update(data)
        # Persist the state and its active-index membership in one atomic step.
        # A crash between the two would otherwise leave a queued/running record
        # that the reconciliation index never scans, stuck forever.
        is_active = "1" if new_data.get("status") in ("queued", "running") else "0"
        await self.redis.eval(
            _SAVE_OPERATION_LUA,
            2,
            "research:operations",
            "research:active_ops",
            op_id,
            json.dumps(new_data),
            is_active,
        )

    async def save_admitted_operation(
        self, op_id: str, data: Dict[str, Any], idempotency_key: str
    ) -> bool:
        if self.degraded:
            return await self.fallback.save_admitted_operation(
                op_id, data, idempotency_key
            )
        saved = await self.redis.eval(
            _SAVE_ADMITTED_OPERATION_LUA,
            4,
            "research:operations",
            "research:active_ops",
            f"research:idempotency:{idempotency_key}",
            f"research:idempotency_pending:{idempotency_key}",
            op_id,
            json.dumps(data),
        )
        return bool(saved)

    async def get_operation(self, op_id: str) -> Optional[Dict[str, Any]]:
        if self.degraded:
            return await self.fallback.get_operation(op_id)
        val = await self.redis.hget("research:operations", op_id)
        return json.loads(val) if val else None

    async def list_operations(self) -> Dict[str, Dict[str, Any]]:
        if self.degraded:
            return await self.fallback.list_operations()
        vals = await self.redis.hgetall("research:operations")
        return {k: json.loads(v) for k, v in vals.items()}

    async def delete_operation(self, op_id: str):
        if self.degraded:
            return await self.fallback.delete_operation(op_id)
        await self.redis.hdel("research:operations", op_id)
        await self.redis.srem("research:active_ops", op_id)
        await self.redis.delete(f"research:events:{op_id}")
        await self.redis.delete(f"research:operation_claims:{op_id}")
        await self.redis.delete(f"research:owners:{op_id}")

    async def claim_idempotency_key(self, key: str, op_id: str) -> Optional[str]:
        if self.degraded:
            return await self.fallback.claim_idempotency_key(key, op_id)
        idem_key = f"research:idempotency:{key}"
        pending_key = f"research:idempotency_pending:{key}"
        existing = await self.redis.eval(
            _CLAIM_IDEMPOTENCY_LUA,
            3,
            idem_key,
            pending_key,
            "research:operations",
            op_id,
            IDEMPOTENCY_KEY_TTL_SECONDS,
            IDEMPOTENCY_ADMISSION_TTL_SECONDS,
        )
        return existing or None

    async def release_idempotency_key(self, key: str, op_id: Optional[str] = None) -> bool:
        if self.degraded:
            return await self.fallback.release_idempotency_key(key, op_id)
        idem_key = f"research:idempotency:{key}"
        pending_key = f"research:idempotency_pending:{key}"
        deleted = await self.redis.eval(
            _RELEASE_IDEMPOTENCY_LUA,
            2,
            idem_key,
            pending_key,
            op_id or "",
        )
        return bool(deleted)

    async def claim_operation_id(self, op_id: str, idempotency_key: str) -> bool:
        if self.degraded:
            return await self.fallback.claim_operation_id(op_id, idempotency_key)

        claim_key = f"research:operation_claims:{op_id}"
        script = """
        local operations_key = KEYS[1]
        local claim_key = KEYS[2]
        local op_id = ARGV[1]
        local idempotency_key = ARGV[2]
        local ttl_seconds = tonumber(ARGV[3])

        local operation = redis.call('HGET', operations_key, op_id)
        if operation then
            local ok, decoded = pcall(cjson.decode, operation)
            if ok and decoded['idempotency_key'] and decoded['idempotency_key'] ~= idempotency_key then
                return 0
            end
        end

        local existing_claim = redis.call('GET', claim_key)
        if existing_claim then
            if existing_claim == idempotency_key then
                return 1
            end
            return 0
        end

        redis.call('SET', claim_key, idempotency_key, 'EX', ttl_seconds)
        return 1
        """
        claimed = await self.redis.eval(
            script,
            2,
            "research:operations",
            claim_key,
            op_id,
            idempotency_key,
            86400
        )
        return bool(claimed)

    async def release_operation_id(self, op_id: str, idempotency_key: Optional[str] = None) -> bool:
        if self.degraded:
            return await self.fallback.release_operation_id(op_id, idempotency_key)

        claim_key = f"research:operation_claims:{op_id}"
        if idempotency_key is not None:
            existing = await self.redis.get(claim_key)
            if existing != idempotency_key:
                return False
        deleted = await self.redis.delete(claim_key)
        return bool(deleted)

    async def add_daily_spend(self, amount_usd: float) -> None:
        if self.degraded:
            return await self.fallback.add_daily_spend(amount_usd)
        # Atomic increment shared across replicas; the window key carries a TTL
        # so the total resets daily without a process-local timer.
        spend_key = "research:spend:daily"
        total = await self.redis.incrbyfloat(spend_key, float(amount_usd))
        if await self.redis.ttl(spend_key) < 0:
            await self.redis.expire(spend_key, 86400)
        return total

    async def get_daily_spend(self) -> float:
        if self.degraded:
            return await self.fallback.get_daily_spend()
        value = await self.redis.get("research:spend:daily")
        return float(value) if value else 0.0

    async def reserve_daily_spend(self, amount_usd: float, limit_usd: float, op_id: str = "") -> bool:
        if self.degraded:
            return await self.fallback.reserve_daily_spend(amount_usd, limit_usd, op_id)
        reserved = await self.redis.eval(
            _RESERVE_SPEND_LUA,
            3,
            "research:spend:daily",
            "research:spend:reservations",
            "research:spend:window",
            float(amount_usd),
            float(limit_usd),
            op_id,
        )
        return bool(reserved)

    async def release_daily_spend(self, amount_usd: float, op_id: str = "") -> None:
        if self.degraded:
            return await self.fallback.release_daily_spend(amount_usd, op_id)
        await self.redis.eval(
            _RELEASE_SPEND_LUA, 3, "research:spend:daily", "research:spend:reservations",
            "research:spend:window", float(amount_usd), op_id,
        )

    async def reconcile_daily_spend(self, reserved_usd: float, actual_usd: float, limit_usd: float, op_id: str = "") -> None:
        if self.degraded:
            return await self.fallback.reconcile_daily_spend(reserved_usd, actual_usd, limit_usd, op_id)
        # Single atomic swap: drop the reservation and add the observed cost so
        # the freed capacity is never briefly visible to another replica, and
        # clamp to the ceiling so a large actual cost cannot overspend. Keyed by
        # op_id, so a retried reconciliation after a lost response cannot charge
        # the completed operation twice or release its hold without charging.
        await self.redis.eval(
            _RECONCILE_SPEND_LUA,
            3,
            "research:spend:daily",
            "research:spend:reservations",
            "research:spend:window",
            float(reserved_usd),
            float(actual_usd),
            float(limit_usd),
            op_id,
        )

    async def mark_stale_operations(self):
        if self.degraded:
            return await self.fallback.mark_stale_operations()
        # Multi-instance backend: only fail operations that no live instance
        # still owns. A rolling deploy or a second replica must not overwrite the
        # state of work actively running elsewhere.
        #
        # Scan only the bounded active-operation index rather than HGETALL over
        # every retained (possibly large) result, so recurring reconciliation
        # does not become an unbounded transfer/parse of historical reports.
        active_op_ids = await self.redis.smembers("research:active_ops")
        # One-time compatibility backfill: queued/running records written before
        # the active index existed are only in research:operations. Without this
        # they would never be scanned and could stay nonterminal forever.
        if await self.redis.set("research:active_ops:backfilled", "1", nx=True, ex=86400):
            for op_id, op in (await self.list_operations()).items():
                if op.get("status") in ("queued", "running"):
                    await self.redis.sadd("research:active_ops", op_id)
                    active_op_ids.add(op_id)
        if not active_op_ids:
            return
        ops = {}
        for op_id in active_op_ids:
            op = await self.get_operation(op_id)
            if op is not None:
                ops[op_id] = op
        for op_id, op in ops.items():
            if op.get("status") not in ("queued", "running"):
                continue
            owner_key = f"research:owners:{op_id}"
            # The owner check and the terminal-state write must be atomic:
            # otherwise reconciliation can observe no owner and then overwrite
            # the state of an operation a peer has meanwhile started running.
            stale_op = dict(op)
            stale_op["status"] = "failed"
            stale_op["error"] = {"code": "STALE_OPERATION", "message": "Operation was abandoned after a service restart.", "retryable": True}
            transitioned = await self.redis.eval(
                _FAIL_IF_UNOWNED_LUA,
                2,
                owner_key,
                "research:operations",
                op_id,
                json.dumps(stale_op),
            )
            if transitioned:
                await self.redis.srem("research:active_ops", op_id)
                logger.warning("Marked stale operation %s as failed", op_id)
            else:
                logger.info("Leaving operation %s running; live owner lease present", op_id)

    async def acquire_concurrency_slot(self, op_id: str) -> bool:
        if self.degraded:
            return await self.fallback.acquire_concurrency_slot(op_id)
        # Each slot is an individual sorted-set member carrying its own expiry.
        # The Lua script reclaims expired members first and admits the new slot
        # only below the ceiling, so a crashed instance frees its own slot and a
        # single shared TTL can neither over-admit nor pin a leaked count. The
        # expiry timestamp comes from Redis TIME, so a replica with a skewed
        # clock cannot reap live peers' slots or create already-expired ones.
        member = self.instance.lease_id(op_id)
        acquired = await self.redis.eval(
            _ACQUIRE_SLOT_LUA,
            1,
            "research:concurrency:slots",
            settings.CONCURRENCY_LEASE_TTL_SECONDS,
            settings.MAX_CONCURRENT_OPS,
            member,
        )
        return bool(acquired)

    async def release_concurrency_slot(self, op_id: str) -> None:
        if self.degraded:
            return await self.fallback.release_concurrency_slot(op_id)
        # Idempotent: only the member this instance reserved is removed.
        await self.redis.zrem("research:concurrency:slots", self.instance.lease_id(op_id))

    async def begin_operation(self, op_id: str) -> bool:
        if self.degraded:
            return await self.fallback.begin_operation(op_id)
        owner_key = f"research:owners:{op_id}"
        # NX is authoritative: if a lease already exists, another instance owns
        # this operation and we must not run it concurrently.
        acquired = await self.redis.set(
            owner_key, self.instance.instance_id, nx=True, ex=settings.CONCURRENCY_LEASE_TTL_SECONDS
        )
        return bool(acquired)

    async def touch_operation(self, op_id: str) -> bool:
        """Heartbeat ownership. Returns False when this instance has lost
        ownership or its slot, so the caller can abort work that is no longer
        the exclusive owner."""
        if self.degraded:
            return await self.fallback.touch_operation(op_id)
        owner_key = f"research:owners:{op_id}"
        slot_key = self.instance.lease_id(op_id)
        # Renew the ownership lease and this operation's own slot atomically.
        # The old slot is renewed only if it still exists, so a delayed
        # heartbeat cannot resurrect a reaped slot and push the set above
        # MAX_CONCURRENT_OPS.
        result = await self.redis.eval(
            _RENEW_LUA,
            2,
            owner_key,
            "research:concurrency:slots",
            self.instance.instance_id,
            slot_key,
            settings.CONCURRENCY_LEASE_TTL_SECONDS,
        )
        if result == 0:
            logger.warning("Ownership lease for operation %s was lost; heartbeat cannot renew it.", op_id)
            return False
        if result == -1:
            logger.warning("Concurrency slot for operation %s was lost; heartbeat cannot renew it.", op_id)
            return False
        return True

    async def release_operation_lease(self, op_id: str) -> None:
        if self.degraded:
            return await self.fallback.release_operation_lease(op_id)
        owner_key = f"research:owners:{op_id}"
        # Atomic compare-and-delete: a lease that expired between a read and a
        # write must not let this late finisher clear a newer owner's lease.
        await self.redis.eval(_CAD_LUA, 1, owner_key, self.instance.instance_id)

    async def push_progress_event(self, op_id: str, event: Dict[str, Any]):
        if self.degraded:
            return await self.fallback.push_progress_event(op_id, event)
        stream_key = f"research:events:{op_id}"
        # Write to Redis Stream
        await self.redis.xadd(stream_key, {"event": json.dumps(event)}, maxlen=1000)

    async def get_progress_events(self, op_id: str) -> List[Dict[str, Any]]:
        if self.degraded:
            return await self.fallback.get_progress_events(op_id)
        stream_key = f"research:events:{op_id}"
        try:
            raw_entries = await self.redis.xrange(stream_key)
            events = []
            for _, fields in raw_entries:
                if "event" in fields:
                    events.append(json.loads(fields["event"]))
            return events
        except Exception:
            return []

    async def get_progress_events_after(self, op_id: str, cursor: str | None):
        if self.degraded:
            return await self.fallback.get_progress_events_after(op_id, cursor)
        minimum = f"({cursor}" if cursor else "-"
        try:
            entries = await self.redis.xrange(f"research:events:{op_id}", min=minimum)
            return [(entry_id, json.loads(fields["event"])) for entry_id, fields in entries if "event" in fields]
        except Exception:
            return []

# Singleton Storage factory resolver
storage: BaseStorage = RedisStorage() if settings.STORAGE_BACKEND == "redis" else InMemoryStorage()
