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

# Atomic acquire of a concurrency slot. Expired slots are reclaimed first, so a
# crashed instance's slot frees itself instead of pinning the counter forever,
# and each slot carries its own expiry rather than sharing one global TTL.
_ACQUIRE_SLOT_LUA = """
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[1])
if redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[3]) then
  return 0
end
redis.call('ZADD', KEYS[1], tonumber(ARGV[1]) + tonumber(ARGV[2]), ARGV[4])
return 1
"""

# Atomic stale transition: only fail the operation if no live owner lease exists.
_FAIL_IF_UNOWNED_LUA = """
if redis.call('EXISTS', KEYS[1]) == 0 then
  redis.call('HSET', KEYS[2], ARGV[1], ARGV[2])
  return 1
end
return 0
"""

# Atomic owner compare-and-{delete,expire}: act only if the stored owner is us.
_CAD_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""
_CAE_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('EXPIRE', KEYS[1], tonumber(ARGV[2]))
end
return 0
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

class InMemoryStorage(BaseStorage):
    def __init__(self):
        # OrderedDict keyed by canonical op_id. Ownership leases live in a
        # separate mapping so a caller-supplied operation ID can never collide
        # with internal metadata.
        self.operations: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.events: Dict[str, List[Dict[str, Any]]] = {}
        self.idempotency_keys: "OrderedDict[str, str]" = OrderedDict()
        self.operation_claims: Dict[str, str] = {}
        self.operation_owners: Dict[str, str] = {}
        self.instance = _InstanceIdentity()
        # Identifiable slots (not a bare counter) so each release frees exactly
        # the slot it reserved.
        self._concurrency_slots: set[str] = set()

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
            self.operation_claims.pop(evicted_key, None)
            self.operation_owners.pop(evicted_key, None)
            # Idempotency reservations are intentionally retained so a retry of
            # evicted work still resolves to its original operation instead of
            # being accepted (and paid for) as new work.
            logger.info("Evicted operation %s from the in-memory cache (limit %d)", evicted_key, OPERATION_CACHE_LIMIT)

    def _evict_idempotency_if_needed(self):
        while len(self.idempotency_keys) > IDEMPOTENCY_RESERVATION_LIMIT:
            self.idempotency_keys.popitem(last=False)

    async def save_operation(self, op_id: str, data: Dict[str, Any]):
        existing = self.operations.get(op_id)
        new_data = dict(existing) if existing else {}
        new_data.update(data)
        self.operations[op_id] = new_data
        self.operations.move_to_end(op_id)
        self._evict_if_needed()

    async def get_operation(self, op_id: str) -> Optional[Dict[str, Any]]:
        return self.operations.get(op_id)

    async def list_operations(self) -> Dict[str, Dict[str, Any]]:
        return self.operations

    async def delete_operation(self, op_id: str):
        self.operations.pop(op_id, None)
        self.events.pop(op_id, None)
        self.operation_claims.pop(op_id, None)
        self.operation_owners.pop(op_id, None)
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

    async def mark_stale_operations(self):
        # Single-process backend: every queued/running operation belongs to this
        # process, so anything still live at startup was abandoned by a restart.
        for op_id, op in list(self.operations.items()):
            if op.get("status") in ("queued", "running"):
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
        self.operation_owners[op_id] = self.instance.instance_id
        return True

    async def touch_operation(self, op_id: str) -> None:
        return None

    async def release_operation_lease(self, op_id: str) -> None:
        # Only the owner may release, so a late finisher cannot clear a lease a
        # newer owner has since taken.
        if self.operation_owners.get(op_id) == self.instance.instance_id:
            self.operation_owners.pop(op_id, None)

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
        await self.redis.hset("research:operations", op_id, json.dumps(new_data))

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
        await self.redis.delete(f"research:events:{op_id}")
        await self.redis.delete(f"research:operation_claims:{op_id}")
        await self.redis.delete(f"research:owners:{op_id}")

    async def claim_idempotency_key(self, key: str, op_id: str) -> Optional[str]:
        if self.degraded:
            return await self.fallback.claim_idempotency_key(key, op_id)
        idem_key = f"research:idempotency:{key}"
        was_set = await self.redis.set(idem_key, op_id, nx=True, ex=86400)
        if was_set:
            return None
        return await self.redis.get(idem_key)

    async def release_idempotency_key(self, key: str, op_id: Optional[str] = None) -> bool:
        if self.degraded:
            return await self.fallback.release_idempotency_key(key, op_id)
        idem_key = f"research:idempotency:{key}"
        if op_id is not None:
            existing = await self.redis.get(idem_key)
            if existing != op_id:
                return False
        deleted = await self.redis.delete(idem_key)
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

    async def mark_stale_operations(self):
        if self.degraded:
            return await self.fallback.mark_stale_operations()
        # Multi-instance backend: only fail operations that no live instance
        # still owns. A rolling deploy or a second replica must not overwrite the
        # state of work actively running elsewhere.
        ops = await self.list_operations()
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
                logger.warning("Marked stale operation %s as failed", op_id)
            else:
                logger.info("Leaving operation %s running; live owner lease present", op_id)

    async def acquire_concurrency_slot(self, op_id: str) -> bool:
        if self.degraded:
            return await self.fallback.acquire_concurrency_slot(op_id)
        # Each slot is an individual sorted-set member carrying its own expiry.
        # The Lua script reclaims expired members fist and admits the new slot
        # only below the ceiling, so a crashed instance frees its own slot and a
        # single shared TTL can neither over-admit nor pin a leaked count.
        now = time.time()
        member = self.instance.lease_id(op_id)
        acquired = await self.redis.eval(
            _ACQUIRE_SLOT_LUA,
            1,
            "research:concurrency:slots",
            now,
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

    async def touch_operation(self, op_id: str) -> None:
        if self.degraded:
            return await self.fallback.touch_operation(op_id)
        owner_key = f"research:owners:{op_id}"
        # Atomic compare-and-expire: only extend a lease we still own.
        await self.redis.eval(
            _CAE_LUA, 1, owner_key, self.instance.instance_id, settings.CONCURRENCY_LEASE_TTL_SECONDS
        )

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

# Singleton Storage factory resolver
storage: BaseStorage = RedisStorage() if settings.STORAGE_BACKEND == "redis" else InMemoryStorage()
