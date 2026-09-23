"""The run claim: one object per run, identity-compared, released whole.

``CronService`` keeps a run's per-run state -- trigger, start stamp, marker
token, generation, monotonic start, jitter, tracked task -- on one
:class:`_RunClaim` stored under the job id in ``_claims``. These tests pin the
object's contract rather than any one consumer of it: a claim is its identity
(two like-for-like claims are two runs), releasing it drops every field at
once (so a field added later inherits the fence), a claim that is not the
stored one never releases the run that is, and a claim ``cancel()`` or the
reaper has taken keeps the job occupied without being the run's to release --
and is its taker's alone to finish: a second cancel, a reap or a finished-task
discard that meets a taken claim leaves it to the teardown that took it.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kiro_crew.cron import _JOB_TIMEOUT_SECS, CronJob, CronSchedule, CronService, _RunClaim
from kiro_crew.cron_history import CronHistoryStore


def _service() -> CronService:
    return CronService(base_dir=None)


def _live_task() -> MagicMock:
    return MagicMock(done=MagicMock(return_value=False))


def _one_sweep() -> Any:
    """Let ``_reaper_loop`` run exactly one sweep, then end it."""
    return patch("asyncio.sleep", AsyncMock(side_effect=[None, asyncio.CancelledError]))


class TestClaimIdentity:
    def test_two_like_for_like_claims_are_two_runs(self) -> None:
        first = _RunClaim(trigger="manual", claimed_at=1.0, marker_run="same")
        second = _RunClaim(trigger="manual", claimed_at=1.0, marker_run="same")
        assert first is not second
        assert first != second, "a claim compares by identity, never by its fields"
        assert first == first

    def test_marker_token_is_minted_per_claim(self) -> None:
        tokens = {_RunClaim(trigger="scheduled", claimed_at=0.0).marker_run for _ in range(8)}
        assert len(tokens) == 8
        assert all(len(token) == 32 for token in tokens)  # uuid4 hex

    def test_claim_run_stores_the_claim_it_returns(self) -> None:
        svc = _service()
        before = time.time()
        claim = svc._claim_run("j1", "manual")
        assert svc._claims["j1"] is claim
        assert claim.trigger == "manual"
        assert before <= claim.claimed_at <= time.time()
        assert claim.task is None and claim.generation is None
        assert claim.started_monotonic is None and claim.jitter is None
        assert claim.taken is False
        assert svc.is_running("j1")
        assert svc.running_since("j1") == claim.claimed_at

    def test_holds_claim_is_identity_not_equality(self) -> None:
        svc = _service()
        stored = svc._claim_run("j1", "scheduled")
        lookalike = _RunClaim(
            trigger=stored.trigger, claimed_at=stored.claimed_at, marker_run=stored.marker_run
        )
        assert svc._holds_claim("j1", stored)
        assert not svc._holds_claim("j1", lookalike)
        assert not svc._holds_claim("j1", None)
        assert not svc._holds_claim("other", stored)


class TestReleaseClearsEveryField:
    def test_one_release_drops_the_whole_claim(self) -> None:
        svc = _service()
        claim = svc._claim_run("j1", "scheduled")
        claim.generation = 7
        claim.started_monotonic = time.monotonic()
        claim.jitter = 12.5
        claim.task = _live_task()

        assert svc._release_claim("j1", claim) is True

        assert "j1" not in svc._claims
        assert not svc.is_running("j1")
        assert svc.running_since("j1") is None
        # Nothing about the run survives anywhere else on the service: the
        # claim was the only place its state lived.
        for value in vars(svc).values():
            if isinstance(value, (dict, set)) and value is not svc._claims:
                assert "j1" not in value

    def test_no_per_run_dict_exists_beside_the_claims(self) -> None:
        """The six job-id dicts the claim replaced must not come back.

        Each of them was fenced by hand at every release site, and a field
        that missed one leaked (the reaper's jitter orphan). A new per-run
        field belongs on ``_RunClaim``, where the one release covers it.
        """
        svc = _service()
        for retired in (
            "_executing",
            "_running_tasks",
            "_job_start_times",
            "_job_start_monotonic",
            "_job_jitter",
            "_job_run_meta",
        ):
            assert not hasattr(svc, retired), f"{retired} exists beside _claims"
        assert {f.name for f in dataclasses.fields(_RunClaim)} == {
            "trigger",
            "claimed_at",
            "marker_run",
            "generation",
            "started_monotonic",
            "jitter",
            "task",
            "taken",
        }

    def test_release_of_an_unstored_claim_is_a_noop(self) -> None:
        svc = _service()
        stray = _RunClaim(trigger="manual", claimed_at=0.0)
        assert svc._release_claim("j1", stray) is False
        assert svc._release_claim("j1", None) is False
        assert not svc.is_running("j1")


class TestStaleClaimNeverReleasesTheCurrentRun:
    def test_a_previous_runs_claim_leaves_the_replacement_alone(self) -> None:
        svc = _service()
        previous = svc._claim_run("j1", "manual")
        assert svc._release_claim("j1", previous) is True
        replacement = svc._claim_run("j1", "scheduled")

        # The previous run's late finalizer / backstop / wrapper release:
        assert svc._release_claim("j1", previous) is False
        assert not svc._holds_claim("j1", previous)
        assert svc._claims["j1"] is replacement
        assert svc.is_running("j1")

    def test_a_taken_claim_is_no_longer_the_runs_to_release(self) -> None:
        svc = _service()
        claim = svc._claim_run("j1", "manual")
        claim.task = _live_task()

        taken = svc._take_claim("j1")
        assert taken is claim and claim.taken is True
        # Occupied through the teardown, silent to the badge, fenced off the run.
        assert svc.is_running("j1")
        assert svc.running_since("j1") is None
        assert not svc._holds_claim("j1", claim)
        assert svc._release_claim("j1", claim) is False
        assert svc._claims["j1"] is claim
        # A second teardown finds nothing to take.
        assert svc._take_claim("j1") is None
        assert svc._take_claim("other") is None

    def test_finish_pops_only_a_taken_claim_and_cancels_its_task(self) -> None:
        svc = _service()
        claim = svc._claim_run("j1", "manual")
        claim.task = _live_task()

        # Not taken: a replacement run's claim is never a teardown's to finish.
        svc._finish_taken_claim("j1")
        assert svc._claims["j1"] is claim
        claim.task.cancel.assert_not_called()

        svc._take_claim("j1")
        svc._finish_taken_claim("j1")
        assert "j1" not in svc._claims
        claim.task.cancel.assert_called_once()
        # Idempotent: a concurrent teardown that reached this step first left nothing.
        svc._finish_taken_claim("j1")

    def test_finish_does_not_cancel_a_task_that_already_ended(self) -> None:
        svc = _service()
        claim = svc._claim_run("j1", "scheduled")
        claim.task = MagicMock(done=MagicMock(return_value=True))
        svc._take_claim("j1")
        svc._finish_taken_claim("j1")
        assert "j1" not in svc._claims
        claim.task.cancel.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_reaper_sweep_leaves_a_taken_claim_to_its_teardown(self) -> None:
        """A claim cancel() or an earlier reap took is theirs to finish.

        The teardown that took it is inside its kill awaits; a sweep that
        measured the run against its deadline anyway would reap it a second
        time and write a second terminal row. The same run, not yet taken, is
        reaped.
        """
        svc = _service()
        svc._sessions = MagicMock()
        overdue = time.time() - _JOB_TIMEOUT_SECS - 60
        claim = svc._claims["j1"] = _RunClaim(
            trigger="scheduled", claimed_at=overdue, task=_live_task()
        )

        with patch.object(svc, "_force_reap", new_callable=AsyncMock) as reap, _one_sweep():
            with pytest.raises(asyncio.CancelledError):
                await svc._reaper_loop()
        reap.assert_awaited_once()
        assert reap.await_args is not None and reap.await_args.args[0] == "j1"

        assert svc._take_claim("j1") is claim
        with patch.object(svc, "_force_reap", new_callable=AsyncMock) as reap, _one_sweep():
            with pytest.raises(asyncio.CancelledError):
                await svc._reaper_loop()
        reap.assert_not_awaited()
        assert svc._claims["j1"] is claim, "the sweep released a claim mid-teardown"


def _cancel_fixture(tmp_path: Any) -> tuple[CronService, CronJob, _RunClaim, MagicMock]:
    """A service with one live agent run, a history store and a mocked session."""
    svc = CronService(base_dir=None, on_job=AsyncMock())
    svc._history = CronHistoryStore(base_dir=tmp_path)
    svc._sessions = MagicMock()
    svc._sessions.reset = AsyncMock()
    job = CronJob(
        id="j1",
        name="job",
        message="m",
        schedule=CronSchedule(kind="every", every_secs=300),
        created_ts=time.time(),
    )
    svc._jobs = [job]
    task = _live_task()
    claim = svc._claims["j1"] = _RunClaim(trigger="manual", claimed_at=time.time() - 5, task=task)
    return svc, job, claim, task


class TestASecondTeardownLeavesTheFirstAlone:
    """A taken claim is its taker's to finish; a later teardown must not pop it.

    ``cancel()`` and ``_force_reap`` take the claim, then await their kills,
    then pop it (``_finish_taken_claim``). A second ``cancel()`` (a
    double-clicked Cancel, or a Cancel during a live reap) that lands inside
    that await finds the claim already taken. Carrying on anyway would pop
    the FIRST teardown's claim early -- the job reads idle while its kill is
    still in flight -- write a second terminal row, and draw a generation for
    it that can outrank a replacement run's, so that run's own result is
    dropped as an older record.
    """

    @pytest.mark.asyncio
    async def test_a_second_cancel_inside_the_firsts_kill_await_answers_not_running(
        self, tmp_path: Any
    ) -> None:
        svc, job, claim, task = _cancel_fixture(tmp_path)
        parked = asyncio.Event()
        release = asyncio.Event()
        resets = 0

        async def _reset(*_args: Any, **_kwargs: Any) -> None:
            # Park only the FIRST teardown's kill; a later reset returns at once
            # so a cancel that wrongly carries on runs to its terminal write.
            nonlocal resets
            resets += 1
            if resets == 1:
                parked.set()
                await release.wait()

        svc._sessions.reset = AsyncMock(side_effect=_reset)

        with patch("kiro_crew.sel.sel"), patch.object(svc, "_save"):
            first = asyncio.get_running_loop().create_task(svc.cancel("j1"))
            await asyncio.wait_for(parked.wait(), timeout=5)
            assert claim.taken and svc._claims["j1"] is claim
            assert svc._cancelled_jobs.has("j1", claim)

            second = await svc.cancel("j1")

            assert (
                svc._claims.get("j1") is claim
            ), "the second cancel popped the first teardown's claim"
            assert second is False, "the second cancel answered True for a run another cancel owns"
            task.cancel.assert_not_called()
            _runs, total = await svc._history.get_job_history("j1")
            assert total == 0, "the second cancel wrote a cancelled row for a run it did not cancel"
            # Only the taker's reset is in flight; the second cancel killed nothing.
            assert resets == 1

            release.set()
            assert await asyncio.wait_for(first, timeout=5) is True

        assert "j1" not in svc._claims
        task.cancel.assert_called_once()
        runs, total = await svc._history.get_job_history("j1")
        assert total == 1, "two overlapping cancels of one run wrote two cancelled rows"
        assert runs[0]["status"] == "cancelled" and runs[0]["trigger"] == "manual"
        assert (job.last_error or "").startswith("Cancelled by user after")

    @pytest.mark.asyncio
    async def test_a_reap_landing_on_a_taken_claim_leaves_the_teardown_alone(
        self, tmp_path: Any
    ) -> None:
        """The reaper's own sweep skips a taken claim; ``_force_reap`` must too.

        A sweep can still reach it: the sweep iterates a snapshot of the claims,
        and while it awaits an earlier job's reap this job's run can end, a
        replacement be claimed and a ``cancel()`` take that replacement -- the
        snapshot's stale, untaken claim then measures the job overdue.
        """
        svc, job, claim, task = _cancel_fixture(tmp_path)
        assert svc._take_claim("j1") is claim  # a cancel() inside its kill await
        svc._cancelled_jobs.mark("j1", claim)
        generation_before = svc._run_generations.get("j1", 0)

        with patch("kiro_crew.sel.sel"), patch.object(svc, "_save"):
            await svc._force_reap("j1", _JOB_TIMEOUT_SECS + 60)

        assert svc._claims.get("j1") is claim, "the reap popped the cancel's taken claim"
        task.cancel.assert_not_called()
        svc._sessions.reset.assert_not_awaited()
        assert not svc._reaped_jobs.has(
            "j1", claim
        ), "the reap marked a run cancel() already marked"
        assert job.last_status != "error" and not (job.last_error or "").startswith("Reaped")
        assert (
            svc._run_generations.get("j1", 0) == generation_before
        ), "the reap drew a generation for a terminal row it does not own"
        _runs, total = await svc._history.get_job_history("j1")
        assert total == 0, "the reap wrote a reaped row beside the cancel's cancelled row"

    def test_discard_finished_run_leaves_a_taken_claim_to_its_teardown(self) -> None:
        """A taken claim whose task has ended is not an orphan: its taker pops it.

        The run task ending is the expected effect of the teardown's kill, so
        a Run click or a second Cancel that calls ``discard_finished_run``
        between the task ending and the teardown resuming would otherwise pop
        the claim: the job reads idle, a replacement run is accepted, and the
        teardown's terminal row then outranks that run's generation.
        """
        svc = _service()
        claim = svc._claim_run("j1", "manual")
        claim.task = MagicMock(done=MagicMock(return_value=True))
        assert svc._take_claim("j1") is claim

        assert (
            svc.discard_finished_run("j1") is False
        ), "discard_finished_run popped a claim a teardown had taken"
        assert svc._claims["j1"] is claim
        assert svc.is_running("j1")

        # The teardown's own finish still pops it, task done or not.
        svc._finish_taken_claim("j1")
        assert "j1" not in svc._claims


class TestTaskTracking:
    def test_attach_run_task_fills_only_an_empty_slot(self) -> None:
        svc = _service()
        claim = svc._claim_run("j1", "manual")
        wrapper = _live_task()
        svc.attach_run_task("j1", wrapper)
        assert claim.task is wrapper

        other = _live_task()
        svc.attach_run_task("j1", other)
        assert claim.task is wrapper, "a live run's task handle was overwritten"

        svc.attach_run_task("unclaimed", other)  # no claim: nothing to track
        assert "unclaimed" not in svc._claims

    @pytest.mark.asyncio
    async def test_discard_finished_run_releases_only_a_done_task(self) -> None:
        svc = _service()
        claim = svc._claim_run("j1", "scheduled")
        assert svc.discard_finished_run("j1") is False  # no task tracked yet

        gate = asyncio.Event()

        async def _run() -> None:
            await gate.wait()

        claim.task = asyncio.get_running_loop().create_task(_run())
        assert svc.discard_finished_run("j1") is False
        assert svc._claims["j1"] is claim

        gate.set()
        await claim.task
        assert svc.discard_finished_run("j1") is True
        assert "j1" not in svc._claims
        assert svc.discard_finished_run("j1") is False
