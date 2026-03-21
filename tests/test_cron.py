"""Tests for cron scheduler and executor."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spare_paw.cron.scheduler import CronScheduler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app_state(**overrides) -> MagicMock:
    """Build a mock AppState with sensible defaults."""
    app_state = MagicMock()
    app_state.config = MagicMock()
    app_state.config.get = MagicMock(side_effect=lambda key, default=None: {
        "models.cron": None,
        "models.main_agent": "test/model",
        "agent.system_prompt": "You are a helpful assistant. Time: {current_time}",
        "agent.max_tool_iterations": 5,
        "telegram.owner_id": 12345,
        "_tool_registry": None,
    }.get(key, default))
    app_state.tool_registry = None
    app_state.router_client = AsyncMock()
    app_state.executor = None
    app_state.application = MagicMock()
    app_state.application.bot = AsyncMock()
    app_state.backend = AsyncMock()
    return app_state


# ---------------------------------------------------------------------------
# CronScheduler CRUD
# ---------------------------------------------------------------------------

def _make_scheduler() -> CronScheduler:
    """Create a CronScheduler with a real AsyncIOScheduler started in paused mode.

    Must be called from within a running event loop (i.e. inside an async test).
    """
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    sched = CronScheduler(MagicMock())
    sched._scheduler = AsyncIOScheduler()
    sched._scheduler.start(paused=True)
    return sched


class TestCronSchedulerCRUD:
    """Tests for add_job / remove_job / pause_job / resume_job."""

    @pytest.mark.asyncio
    async def test_add_job(self):
        scheduler = _make_scheduler()
        try:
            await scheduler.add_job("job-1", "*/5 * * * *", "do something")
            job = scheduler._scheduler.get_job("job-1")
            assert job is not None
        finally:
            scheduler._scheduler.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_remove_job(self):
        scheduler = _make_scheduler()
        try:
            await scheduler.add_job("job-2", "0 * * * *", "hourly task")
            await scheduler.remove_job("job-2")
            assert scheduler._scheduler.get_job("job-2") is None
        finally:
            scheduler._scheduler.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_remove_nonexistent_job_does_not_raise(self):
        """Removing a job that doesn't exist should not raise."""
        scheduler = _make_scheduler()
        try:
            await scheduler.remove_job("no-such-job")  # should not raise
        finally:
            scheduler._scheduler.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_pause_job(self):
        scheduler = _make_scheduler()
        try:
            await scheduler.add_job("job-3", "0 9 * * *", "morning task")
            await scheduler.pause_job("job-3")
            job = scheduler._scheduler.get_job("job-3")
            assert job is not None
            # A paused job has next_run_time == None
            assert job.next_run_time is None
        finally:
            scheduler._scheduler.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_resume_job(self):
        scheduler = _make_scheduler()
        try:
            await scheduler.add_job("job-4", "0 9 * * *", "morning task")
            await scheduler.pause_job("job-4")
            await scheduler.resume_job("job-4")
            job = scheduler._scheduler.get_job("job-4")
            assert job is not None
            # After resume, verify the call didn't raise.
        finally:
            scheduler._scheduler.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_add_job_raises_when_scheduler_not_running(self):
        """add_job raises RuntimeError when scheduler is None."""
        sched = CronScheduler(MagicMock())
        with pytest.raises(RuntimeError, match="not running"):
            await sched.add_job("x", "* * * * *", "p")

    @pytest.mark.asyncio
    async def test_pause_noop_when_scheduler_none(self):
        """pause_job is a no-op when scheduler hasn't been started."""
        sched = CronScheduler(MagicMock())
        await sched.pause_job("x")  # should not raise

    @pytest.mark.asyncio
    async def test_resume_noop_when_scheduler_none(self):
        sched = CronScheduler(MagicMock())
        await sched.resume_job("x")  # should not raise


# ---------------------------------------------------------------------------
# execute_cron — success path
# ---------------------------------------------------------------------------

class TestExecuteCronSuccess:
    """Test execute_cron sends the result to Telegram on success."""

    @pytest.mark.asyncio
    @patch("spare_paw.cron.executor._maybe_delete_once", new_callable=AsyncMock)
    @patch("spare_paw.cron.executor._update_cron_result", new_callable=AsyncMock)
    @patch("spare_paw.cron.executor.run_tool_loop", new_callable=AsyncMock)
    async def test_sends_result_to_telegram(self, mock_tool_loop, mock_update, _mock_delete):
        from spare_paw.cron.executor import execute_cron

        mock_tool_loop.return_value = "Cron result text"
        app_state = _make_app_state()

        await execute_cron(app_state, "cron-1", "what is the weather?", None, None)

        # Tool loop was called
        mock_tool_loop.assert_awaited_once()

        # Backend sent the result
        app_state.backend.send_text.assert_awaited_once()
        sent_text = app_state.backend.send_text.call_args[0][0]
        assert "Cron result text" in sent_text

        # DB was updated with the result
        mock_update.assert_awaited_once()
        _ = mock_update.call_args[1] if mock_update.call_args[1] else {}
        positional = mock_update.call_args[0]
        # First positional arg is cron_id
        assert positional[0] == "cron-1"


# ---------------------------------------------------------------------------
# execute_cron — error path
# ---------------------------------------------------------------------------

class TestExecuteCronError:
    """Test execute_cron handles errors and sends warning message."""

    @pytest.mark.asyncio
    @patch("spare_paw.cron.executor._maybe_delete_once", new_callable=AsyncMock)
    @patch("spare_paw.cron.executor._update_cron_result", new_callable=AsyncMock)
    @patch("spare_paw.cron.executor.run_tool_loop", new_callable=AsyncMock)
    async def test_sends_warning_on_failure(self, mock_tool_loop, mock_update, _mock_delete):
        from spare_paw.cron.executor import execute_cron

        mock_tool_loop.side_effect = RuntimeError("model exploded")
        app_state = _make_app_state()

        await execute_cron(app_state, "cron-err", "broken prompt", None, None)

        # Backend sent a warning message
        app_state.backend.send_text.assert_awaited_once()
        sent_text = app_state.backend.send_text.call_args[0][0]
        assert "cron-err" in sent_text
        assert "failed" in sent_text.lower() or "RuntimeError" in sent_text

        # DB was updated with the error
        mock_update.assert_awaited_once()
        positional = mock_update.call_args[0]
        assert positional[0] == "cron-err"
        # The error kwarg should be set
        assert "error" in (mock_update.call_args[1] or {}) or (
            len(positional) >= 3 and positional[2] is None  # result=None
        )

    @pytest.mark.asyncio
    @patch("spare_paw.cron.executor._maybe_delete_once", new_callable=AsyncMock)
    @patch("spare_paw.cron.executor._update_cron_result", new_callable=AsyncMock)
    @patch("spare_paw.cron.executor.run_tool_loop", new_callable=AsyncMock)
    async def test_error_notification_failure_does_not_propagate(self, mock_tool_loop, mock_update, _mock_delete):  # noqa: E501
        """If sending the error notification itself fails, execute_cron still doesn't raise."""
        from spare_paw.cron.executor import execute_cron

        mock_tool_loop.side_effect = RuntimeError("model down")
        app_state = _make_app_state()
        # Make the backend raise when trying to send the warning
        app_state.backend.send_text = AsyncMock(
            side_effect=Exception("Telegram down")
        )

        # Should not raise even though both tool_loop and notification failed
        await execute_cron(app_state, "cron-x", "prompt", None, None)

        # DB update should still have been attempted
        mock_update.assert_awaited_once()


# ---------------------------------------------------------------------------
# One-shot auto-delete
# ---------------------------------------------------------------------------


class TestOneShotCronAutoDelete:
    """Test _maybe_delete_once auto-deletes crons with metadata.once=true."""

    @pytest.mark.asyncio
    async def test_one_shot_cron_auto_deletes(self):
        """A cron with metadata={"once": true} should be deleted after execution."""
        from spare_paw.cron.executor import _maybe_delete_once

        # Build a mock DB that returns a row with once=true metadata
        mock_row = {"metadata": '{"once": true}'}
        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=mock_row)

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_cursor)
        mock_db.commit = AsyncMock()

        app_state = _make_app_state()
        app_state.scheduler = AsyncMock()
        app_state.scheduler.remove_job = AsyncMock()

        with patch("spare_paw.cron.executor.get_db", return_value=mock_db):
            await _maybe_delete_once(app_state, "cron-once-1")

        # Verify DELETE was issued
        delete_calls = [
            call for call in mock_db.execute.call_args_list
            if "DELETE" in str(call)
        ]
        assert len(delete_calls) == 1
        assert "cron-once-1" in str(delete_calls[0])

        # Verify scheduler was notified
        app_state.scheduler.remove_job.assert_awaited_once_with("cron-once-1")

    @pytest.mark.asyncio
    async def test_non_once_cron_is_not_deleted(self):
        """A cron without once=true in metadata should NOT be deleted."""
        from spare_paw.cron.executor import _maybe_delete_once

        mock_row = {"metadata": '{"repeat": true}'}
        mock_cursor = AsyncMock()
        mock_cursor.fetchone = AsyncMock(return_value=mock_row)

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_cursor)

        app_state = _make_app_state()
        app_state.scheduler = AsyncMock()

        with patch("spare_paw.cron.executor.get_db", return_value=mock_db):
            await _maybe_delete_once(app_state, "cron-repeat")

        # Only the SELECT should have been called, no DELETE
        calls_str = str(mock_db.execute.call_args_list)
        assert "DELETE" not in calls_str
