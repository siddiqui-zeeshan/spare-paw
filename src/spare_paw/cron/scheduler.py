"""APScheduler-based cron scheduler for persistent scheduled jobs.

Loads enabled cron jobs from SQLite on startup, schedules them via
AsyncIOScheduler with CronTrigger, and provides CRUD operations that
keep the scheduler and database in sync.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from spare_paw.db import get_db

if TYPE_CHECKING:
    from spare_paw.gateway import AppState

logger = logging.getLogger(__name__)


class CronScheduler:
    """Manages APScheduler lifecycle and job CRUD backed by SQLite."""

    def __init__(self, app_state: AppState) -> None:
        self._app_state = app_state
        self._scheduler: AsyncIOScheduler | None = None

    async def start(self) -> None:
        """Create the scheduler, load all enabled crons from DB, and start."""
        self._scheduler = AsyncIOScheduler()

        db = await get_db()
        async with db.execute(
            "SELECT id, schedule, prompt, model, tools_allowed "
            "FROM cron_jobs WHERE enabled = 1"
        ) as cursor:
            rows = await cursor.fetchall()

        for row in rows:
            cron_id = row["id"]
            schedule = row["schedule"]
            prompt = row["prompt"]
            model = row["model"]
            tools_allowed = row["tools_allowed"]

            # tools_allowed is stored as JSON list or NULL
            tools_list: list[str] | None = None
            if tools_allowed:
                import json

                try:
                    tools_list = json.loads(tools_allowed)
                except (json.JSONDecodeError, TypeError):
                    tools_list = None

            try:
                trigger = CronTrigger.from_crontab(schedule)
                self._scheduler.add_job(
                    self._run_cron,
                    trigger=trigger,
                    id=cron_id,
                    args=[cron_id, prompt, model, tools_list],
                    replace_existing=True,
                )
                logger.info("Scheduled cron %s (%s)", cron_id, schedule)
            except (ValueError, TypeError) as exc:
                logger.error(
                    "Failed to schedule cron %s with expression %r: %s",
                    cron_id,
                    schedule,
                    exc,
                )

        self._scheduler.start()
        logger.info(
            "Cron scheduler started with %d jobs", len(self._scheduler.get_jobs())
        )

    async def stop(self) -> None:
        """Shut down the scheduler gracefully."""
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=False)
            self._scheduler = None
            logger.info("Cron scheduler stopped")

    async def add_job(
        self,
        cron_id: str,
        schedule: str,
        prompt: str,
        model: str | None = None,
        tools_allowed: list[str] | None = None,
    ) -> None:
        """Add a new cron job to the scheduler.

        The job must already exist in the cron_jobs table.
        """
        if self._scheduler is None:
            raise RuntimeError("Scheduler is not running")

        trigger = CronTrigger.from_crontab(schedule)
        self._scheduler.add_job(
            self._run_cron,
            trigger=trigger,
            id=cron_id,
            args=[cron_id, prompt, model, tools_allowed],
            replace_existing=True,
        )
        logger.info("Added cron job %s (%s)", cron_id, schedule)

    async def remove_job(self, cron_id: str) -> None:
        """Remove a cron job from the scheduler."""
        if self._scheduler is None:
            return
        try:
            self._scheduler.remove_job(cron_id)
            logger.info("Removed cron job %s", cron_id)
        except Exception:
            logger.debug("Job %s not found in scheduler (may already be removed)", cron_id)

    async def pause_job(self, cron_id: str) -> None:
        """Pause a cron job without removing it."""
        if self._scheduler is None:
            return
        try:
            self._scheduler.pause_job(cron_id)
            logger.info("Paused cron job %s", cron_id)
        except Exception:
            logger.warning("Failed to pause job %s", cron_id, exc_info=True)

    async def resume_job(self, cron_id: str) -> None:
        """Resume a paused cron job."""
        if self._scheduler is None:
            return
        try:
            self._scheduler.resume_job(cron_id)
            logger.info("Resumed cron job %s", cron_id)
        except Exception:
            logger.warning("Failed to resume job %s", cron_id, exc_info=True)

    def get_next_run(self, cron_id: str) -> datetime | None:
        """Return the next scheduled fire time for a cron job, or None."""
        if self._scheduler is None:
            return None
        job = self._scheduler.get_job(cron_id)
        if job is None:
            return None
        return job.next_run_time

    async def _run_cron(
        self,
        cron_id: str,
        prompt: str,
        model: str | None,
        tools_allowed: list[str] | None,
    ) -> None:
        """Job callback — delegates to executor.execute_cron."""
        from spare_paw.cron.executor import execute_cron

        await execute_cron(self._app_state, cron_id, prompt, model, tools_allowed)


async def init_scheduler(app_state: AppState) -> CronScheduler:
    """Convenience function to create, start, and return a CronScheduler.

    Called by gateway._async_main() during startup.
    """
    scheduler = CronScheduler(app_state)
    await scheduler.start()
    return scheduler
