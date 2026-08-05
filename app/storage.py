# app/storage.py
import json
import logging
import time
from typing import Dict, Any, List, Optional
from app.config import settings

logger = logging.getLogger("web-intelligence")

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

class InMemoryStorage(BaseStorage):
    def __init__(self):
        self.operations: Dict[str, Dict[str, Any]] = {}
        self.events: Dict[str, List[Dict[str, Any]]] = {}
        self.idempotency_keys: Dict[str, str] = {}
        self.operation_claims: Dict[str, str] = {}

    async def save_operation(self, op_id: str, data: Dict[str, Any]):
        existing = self.operations.get(op_id)
        new_data = dict(existing) if existing else {}
        new_data.update(data)
        self.operations[op_id] = new_data

    async def get_operation(self, op_id: str) -> Optional[Dict[str, Any]]:
        return self.operations.get(op_id)

    async def list_operations(self) -> Dict[str, Dict[str, Any]]:
        return self.operations

    async def delete_operation(self, op_id: str):
        self.operations.pop(op_id, None)
        self.events.pop(op_id, None)
        self.operation_claims.pop(op_id, None)
        for key, existing_op_id in list(self.idempotency_keys.items()):
            if existing_op_id == op_id:
                self.idempotency_keys.pop(key, None)

    async def claim_idempotency_key(self, key: str, op_id: str) -> Optional[str]:
        existing = self.idempotency_keys.get(key)
        if existing:
            return existing
        self.idempotency_keys[key] = op_id
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
        for op_id, op in list(self.operations.items()):
            if op.get("status") in ("queued", "running"):
                op["status"] = "failed"
                op["error"] = {"code": "STALE_OPERATION", "message": "Operation was abandoned after a service restart.", "retryable": True}
                logger.warning("Marked stale operation %s as failed", op_id)

class RedisStorage(BaseStorage):
    def __init__(self):
        self.redis = None
        self.fallback = InMemoryStorage()
        self.degraded = False

    async def init(self):
        if not settings.REDIS_URL:
            logger.warning("REDIS_URL is not configured; falling back to in-memory storage.")
            self.degraded = True
            return

        try:
            import redis.asyncio as aioredis
            self.redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
            await self.redis.ping()
            logger.info("Connected to Redis storage backend")
        except Exception:
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
        ops = await self.list_operations()
        for op_id, op in ops.items():
            if op.get("status") in ("queued", "running"):
                op["status"] = "failed"
                op["error"] = {"code": "STALE_OPERATION", "message": "Operation was abandoned after a service restart.", "retryable": True}
                await self.redis.hset("research:operations", op_id, json.dumps(op))
                logger.warning("Marked stale operation %s as failed", op_id)

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
