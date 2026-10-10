# app/cancellation.py
import asyncio
import logging
from typing import Awaitable, Callable, Dict, Optional

from app.researcher_adapter import DEFAULT_INGEST_WAIT_TIMEOUT_S

logger = logging.getLogger("web-intelligence")

CANCEL_CHANNEL = "research:cancel"
TERMINAL_STATUSES = {"completed", "partial", "failed", "cancelled"}


class CancellationManager:
    def __init__(self):
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self._redis = None
        self._pubsub = None
        self._listener_task: Optional[asyncio.Task] = None

    async def init(self, redis_client=None):
        if redis_client is None:
            return
        self._redis = redis_client
        self._pubsub = redis_client.pubsub()
        await self._pubsub.subscribe(CANCEL_CHANNEL)
        self._listener_task = asyncio.create_task(self._listen())

    async def _listen(self):
        try:
            async for msg in self._pubsub.listen():
                if msg["type"] != "message":
                    continue
                op_id = msg["data"]
                if isinstance(op_id, bytes):
                    op_id = op_id.decode()
                task = self.active_tasks.get(op_id)
                if task and not task.done():
                    logger.info("Received cross-instance cancel for operation %s", op_id)
                    task.cancel()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Cancellation listener died")

    def register_task(self, op_id: str, task: asyncio.Task):
        self.active_tasks[op_id] = task
        logger.info("Registered active task for operation: %s", op_id)

    def unregister_task(self, op_id: str):
        self.active_tasks.pop(op_id, None)
        logger.debug("Unregistered task for operation: %s", op_id)

    async def cancel_task(
        self,
        op_id: str,
        operation_lookup: Optional[Callable[[str], Awaitable[Optional[dict]]]] = None
    ) -> bool:
        task = self.active_tasks.get(op_id)
        if task and not task.done():
            # A task stays registered for the whole of its own cleanup
            # (ingest-await, then spend/lease/concurrency-slot release) so
            # quiesce_tasks() can still find and await it if shutdown lands
            # during that window - it only unregisters once that cleanup is
            # actually done. That means a result can already be durably
            # terminal (completed/partial/failed/cancelled) while the task
            # is still registered here, so this local branch must check the
            # stored status itself before cancelling, exactly as the
            # cross-instance branch below already does - otherwise a client
            # that polls the result, sees it terminal, and then calls cancel
            # during that cleanup window flips an immutable, already-
            # persisted result to a spurious "cancelled" response.
            # Bounded: the Redis client has no socket timeout (app/storage.py's
            # aioredis.from_url call), so a connection that accepts but never
            # answers could otherwise stall this lookup forever - and unlike
            # the cross-instance branch below, cancelling a task this replica
            # already owns needs nothing from Redis at all. A hung status
            # check must never be the reason a cancel request for your own
            # task never reaches task.cancel(); on timeout, proceed exactly
            # as if the lookup had found nothing to refuse on.
            if operation_lookup:
                try:
                    op = await asyncio.wait_for(operation_lookup(op_id), timeout=DEFAULT_INGEST_WAIT_TIMEOUT_S)
                except asyncio.TimeoutError:
                    logger.warning(
                        "Timed out checking stored status before cancelling operation %s; proceeding with cancellation.",
                        op_id
                    )
                    op = None
                if op and op.get("status") in TERMINAL_STATUSES:
                    logger.info(
                        "Refusing local cancel for already finalized operation %s with status %s",
                        op_id,
                        op.get("status")
                    )
                    return False
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.info("Task successfully cancelled for operation: %s", op_id)
            self.unregister_task(op_id)
            # The check above can be stale by the time it matters: the task
            # can race ahead and persist its own terminal result in the gap
            # between that await and this point, making task.cancel() either
            # a no-op (already done) or a cancellation its own cancellation-
            # safe cleanup absorbs - either way it finishes normally, not
            # with CancelledError, so the except above never fires. There is
            # no further race to check against here, though: `task` has now
            # fully finished (this await only returns once it has), and
            # nothing else writes this operation's status once its owning
            # task has returned - so this result is authoritative. A
            # non-"cancelled" terminal status here means the cancellation
            # simply arrived too late to take effect.
            #
            # That "nothing else writes it" guarantee depends on an ordering
            # invariant in background_research_task (app/api.py), not
            # anything enforced here: it always persists this operation's
            # terminal status, if it persists one at all, before any of its
            # own cleanup (ingest wait, releases, unregistration) and before
            # returning - so the owning task's return implies its final
            # status write, if any, already happened. Moving a status write
            # to after cleanup there - or adding a second writer for the
            # same op_id - would silently reopen the race this check exists
            # to close.
            #
            # "If it persists one at all" matters here too: on a lost-lease
            # exit (_LeaseLostError in app/api.py), this task deliberately
            # persists nothing, so a status this stale (still "running",
            # left by whichever worker now actually owns it) can be exactly
            # what storage still shows once this await returns - the status
            # is neither "cancelled" nor one of TERMINAL_STATUSES, so the
            # narrower check below would fall through to a false "cancelled"
            # if it only looked for an unexpected *terminal* status. Cancel
            # really took effect only when the authoritative post-await
            # status is "cancelled" - anything else, terminal or not, means
            # either the cancellation arrived too late or this worker lost
            # ownership without it, and the real state belongs to whoever
            # wrote (or will write) that status.
            #
            # Bounded for the same reason the pre-cancel check above is: the
            # cancellation itself already happened (or didn't need to,
            # task.cancel() having raced past a task that already finished)
            # by this point, purely in-process - a hung Redis read here must
            # not turn a cancel this replica actually carried out into a
            # hang. On timeout, fall back to reporting what we're actually
            # sure of: the local task was cancelled and awaited.
            if operation_lookup:
                try:
                    op = await asyncio.wait_for(operation_lookup(op_id), timeout=DEFAULT_INGEST_WAIT_TIMEOUT_S)
                except asyncio.TimeoutError:
                    logger.warning(
                        "Timed out checking stored status after cancelling operation %s; reporting cancelled based on the local task alone.",
                        op_id
                    )
                    op = None
                if op and op.get("status") != "cancelled":
                    logger.info(
                        "Operation %s is %s (not cancelled) after awaiting its task; reporting that instead",
                        op_id,
                        op.get("status")
                    )
                    return False
            return True

        # Task not on this instance — broadcast via Redis pub/sub
        if self._redis:
            # Bounded, same reason as the local branch above. Unlike there,
            # though, a timeout here can't fall back to "proceed exactly as
            # if nothing was found" - that already means something specific
            # (refuse, since the operation is unknown) for a lookup that
            # actually ran. On timeout we genuinely don't know either way,
            # so skip the refusal entirely rather than either answer: this
            # instance has no task of its own at stake here, only whether to
            # publish a broadcast, and a redundant or unnecessary one is
            # harmless (the receiving instance only acts if it actually owns
            # a still-running task for this op_id) - unlike silently
            # dropping a cancel request because Redis happened to be slow.
            if operation_lookup:
                try:
                    op = await asyncio.wait_for(operation_lookup(op_id), timeout=DEFAULT_INGEST_WAIT_TIMEOUT_S)
                except asyncio.TimeoutError:
                    logger.warning(
                        "Timed out checking stored status before broadcasting cancel for operation %s; broadcasting anyway.",
                        op_id
                    )
                else:
                    if not op:
                        logger.warning("Refusing cross-instance cancel for unknown operation: %s", op_id)
                        return False
                    if op.get("status") in TERMINAL_STATUSES:
                        logger.info(
                            "Refusing cross-instance cancel for already finalized operation %s with status %s",
                            op_id,
                            op.get("status")
                        )
                        return False
            logger.info("Broadcasting cancel for operation %s to other instances", op_id)
            await self._redis.publish(CANCEL_CHANNEL, op_id)
            return True

        logger.warning("No active task found to cancel for operation: %s", op_id)
        return False

    async def quiesce_tasks(self, timeout=None) -> int:
        """Cancel and await in-flight research tasks before shutdown proceeds.

        Shutdown must not let a still-running research task finish after the
        pending-ingest snapshot was taken, or its outcome ingest would be
        scheduled into a closing event loop. Cancelling and awaiting the active
        tasks first makes that snapshot complete.

        The default must cover a task's own worst-case cleanup time, not just
        a nominal grace period: background_research_task's `finally` block
        (app/api.py) runs exactly three cancellation-safe waits serially on
        either path it can take - ingest, lease, concurrency slot on success,
        or spend, lease, concurrency slot on failure/cancellation when a
        spend hold was reserved - each individually bounded by
        DEFAULT_INGEST_WAIT_TIMEOUT_S. A shorter timeout here would let this
        call return, and shutdown proceed to drain ingests and close storage
        connections, while a task is still mid-cleanup - abandoning its
        owner-lease/concurrency-slot/spend-hold release exactly as this PR is
        fixing for the outcome ingest.

        Set to 4x, not exactly 3x, that per-step bound: this wait's own
        deadline starts counting the moment it is called, strictly before
        each task's own cancellation is actually delivered and its cleanup
        loop computes its own three per-step deadlines - so even with zero
        added overhead, a task's full serial cleanup can run past this
        wait's deadline if the two were set to the exact same multiple.
        Scheduling and logging overhead across three sequential waits adds
        more on top of that. The extra step's worth of headroom is enough
        to absorb both without needing to measure or tune the exact gap.

        Looked up fresh on each call (rather than bound as a literal
        default) so tests can monkeypatch DEFAULT_INGEST_WAIT_TIMEOUT_S the
        same way they do elsewhere.
        """
        if timeout is None:
            timeout = 4 * DEFAULT_INGEST_WAIT_TIMEOUT_S
        pending = [task for task in self.active_tasks.values() if not task.done()]
        if not pending:
            return 0
        for task in pending:
            task.cancel()
        done, _still_pending = await asyncio.wait(pending, timeout=timeout)
        for task in done:
            # Retrieve the outcome so a cancelled/failed task is not reported as
            # an unhandled exception while the loop is tearing down.
            if not task.cancelled():
                task.exception()
        return len(pending)

    async def shutdown(self):
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        if self._pubsub:
            await self._pubsub.unsubscribe(CANCEL_CHANNEL)


cancellation_manager = CancellationManager()
