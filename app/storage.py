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
        
    async def claim_lock(self, op_id: str, instance_id: str) -> bool:
        raise NotImplementedError()
        
    async def push_progress_event(self, op_id: str, event: Dict[str, Any]):
        raise NotImplementedError()
        
    async def get_progress_events(self, op_id: str) -> List[Dict[str, Any]]:
        raise NotImplementedError()

class InMemoryStorage(BaseStorage):
    def __init__(self):
        self.operations: Dict[str, Dict[str, Any]] = {}
        self.locks: Dict[str, str] = {}
        self.events: Dict[str, List[Dict[str, Any]]] = {}
        
    async def save_operation(self, op_id: str, data: Dict[str, Any]):
        self.operations[op_id] = data
        
    async def get_operation(self, op_id: str) -> Optional[Dict[str, Any]]:
        return self.operations.get(op_id)
        
    async def list_operations(self) -> Dict[str, Dict[str, Any]]:
        return self.operations
        
    async def delete_operation(self, op_id: str):
        self.operations.pop(op_id, None)
        self.locks.pop(op_id, None)
        self.events.pop(op_id, None)
        
    async def claim_lock(self, op_id: str, instance_id: str) -> bool:
        if op_id in self.locks and self.locks[op_id] != instance_id:
            return False
        self.locks[op_id] = instance_id
        return True
        
    async def push_progress_event(self, op_id: str, event: Dict[str, Any]):
        if op_id not in self.events:
            self.events[op_id] = []
        self.events[op_id].append(event)
        
    async def get_progress_events(self, op_id: str) -> List[Dict[str, Any]]:
        return self.events.get(op_id, [])

class RedisStorage(BaseStorage):
    def __init__(self):
        self.redis = None
        
    async def init(self):
        import redis.asyncio as aioredis
        self.redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        logger.info("Connected to Redis storage backend")
        
    async def save_operation(self, op_id: str, data: Dict[str, Any]):
        await self.redis.hset("research:operations", op_id, json.dumps(data))
        
    async def get_operation(self, op_id: str) -> Optional[Dict[str, Any]]:
        val = await self.redis.hget("research:operations", op_id)
        return json.loads(val) if val else None
        
    async def list_operations(self) -> Dict[str, Dict[str, Any]]:
        vals = await self.redis.hgetall("research:operations")
        return {k: json.loads(v) for k, v in vals.items()}
        
    async def delete_operation(self, op_id: str):
        await self.redis.hdel("research:operations", op_id)
        await self.redis.delete(f"research:locks:{op_id}")
        await self.redis.delete(f"research:events:{op_id}")
        
    async def claim_lock(self, op_id: str, instance_id: str) -> bool:
        # Atomic SET key NX EX 600 (Distributed Lock)
        lock_key = f"research:locks:{op_id}"
        success = await self.redis.set(lock_key, instance_id, nx=True, ex=600)
        return bool(success)
        
    async def push_progress_event(self, op_id: str, event: Dict[str, Any]):
        stream_key = f"research:events:{op_id}"
        # Write to Redis Stream
        await self.redis.xadd(stream_key, {"event": json.dumps(event)}, maxlen=1000)
        
    async def get_progress_events(self, op_id: str) -> List[Dict[str, Any]]:
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
