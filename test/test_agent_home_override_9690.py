"""A non-default ``KIROCREW_HOME`` must not rewrite the shared ``~/.kiro/agents`` specs.

The failure these pin: a throwaway gateway booted with ``KIROCREW_HOME=<scratch>``
(no ``KIRO_HOME``, not a worktree, not under the temp root) runs
``rebuild_agent_config`` on boot, rewrites the operator's machine-wide
``~/.kiro/agents/*.json`` and pins ``KIROCREW_HOME=<scratch>`` into every
managed MCP server's ``env``. Every ``kirocrew-core`` stub the REAL gateway's
sessions spawn afterwards resolves ``config_dir()`` to the scratch home, finds
no signed session-pid mapping there, and every strict-identity tool is refused
with "signed pid mapping did not verify" -- while ``kirocrew doctor`` on the real
gateway reports a healthy trust root.

Three layers are pinned here:

* ``config.paths.foreign_data_home`` / ``adopt_isolated_kiro_home`` -- a
  non-default data home is given its OWN kiro home (``<data home>/kiro``) via
  ``KIRO_HOME``, the same recipe pods and the E2E harness already use by hand;
* ``agent._decline_shared_agent_home`` -- and if a caller bypasses that prologue,
  the write guard refuses the shared target outright (audited);
* ``doctor_spec_home`` -- ``kirocrew doctor`` names a spec whose pin disagrees
  with the data home it runs on, with the one-command remedy.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import json
import os
import re
import stat
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from conftest import requires_symlinks
from kiro_crew import pinned_fs
from kiro_crew.config import paths
from kiro_crew.config.paths import (
    adopt_isolated_kiro_home,
    foreign_data_home,
    isolated_agents_dir,
    isolated_kiro_home,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

requires_pinned_walk = pytest.mark.skipif(
    not pinned_fs.supports_pinned_walk(),
    reason="descriptor-pinned publish requires dir_fd walks; the reparse-point fallback runs elsewhere",
)


def _relocate_main_homes(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    """Point the default/legacy data homes at tmp so no test names the real ones."""
    default = tmp_path / "user" / ".kiro" / "crew"
    legacy = tmp_path / "user" / ".kirocrew"
    monkeypatch.setattr(paths, "_default_home", lambda: default)
    monkeypatch.setattr(paths, "_legacy_home", lambda: legacy)
    return default, legacy


def _unlink_path(path: str | os.PathLike[str], kwargs: dict) -> Path:
    """Render a by-name or descriptor-relative unlink as one absolute path."""
    dir_fd = kwargs.get("dir_fd")
    if dir_fd is None:
        return Path(path)
    parent = pinned_fs.fd_real_path(dir_fd)
    assert parent is not None, "test needs the pinned directory's real path"
    return Path(parent) / path


# --------------------------------------------------------------------------
# foreign_data_home: which instance owns ~/.kiro
# --------------------------------------------------------------------------
class TestForeignDataHome:
    def test_no_override_is_the_main_instance(self, monkeypatch, tmp_path):
        _relocate_main_homes(monkeypatch, tmp_path)
        monkeypatch.delenv("KIROCREW_HOME", raising=False)
        assert foreign_data_home() is None

    def test_override_naming_the_default_home_is_still_main(self, monkeypatch, tmp_path):
        default, _ = _relocate_main_homes(monkeypatch, tmp_path)
        default.mkdir(parents=True)
        # Lexical re-spellings of the same home (a trailing slash here; ``~`` and
        # ``..`` segments fold the same way) read as main. A symlink alias is the
        # documented gap -- see ``foreign_data_home``'s docstring -- and is not
        # promised here.
        monkeypatch.setenv("KIROCREW_HOME", str(default) + "/")
        assert foreign_data_home() is None

    def test_override_naming_the_legacy_home_is_still_main(self, monkeypatch, tmp_path):
        _, legacy = _relocate_main_homes(monkeypatch, tmp_path)
        legacy.mkdir(parents=True)
        monkeypatch.setenv("KIROCREW_HOME", str(legacy))
        assert foreign_data_home() is None

    def test_any_other_valid_override_is_foreign(self, monkeypatch, tmp_path):
        _relocate_main_homes(monkeypatch, tmp_path)
        scratch = tmp_path / "scratch" / "runtime-3c0a7e8b" / "pfshot" / "home"
        scratch.mkdir(parents=True)
        monkeypatch.setenv("KIROCREW_HOME", str(scratch))
        assert foreign_data_home() == scratch.resolve()

    def test_an_unsafe_override_is_not_foreign(self, monkeypatch, tmp_path):
        """A refused override (``/``) falls back to the default home everywhere
        else, so it must read as the main instance here too -- otherwise the
        write guard would refuse the real install its own specs. ``Path.home`` is
        pinned because the default-path resolution drops a breadcrumb beside it."""
        _relocate_main_homes(monkeypatch, tmp_path)
        (tmp_path / "user").mkdir()
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "user"))
        monkeypatch.setenv("KIROCREW_HOME", "/")
        assert foreign_data_home() is None


# --------------------------------------------------------------------------
# adopt_isolated_kiro_home: the prologue export
# --------------------------------------------------------------------------
class TestAdoptIsolatedKiroHome:
    def test_default_home_exports_nothing(self, monkeypatch, tmp_path):
        _relocate_main_homes(monkeypatch, tmp_path)
        monkeypatch.delenv("KIROCREW_HOME", raising=False)
        monkeypatch.delenv("KIRO_HOME", raising=False)
        assert adopt_isolated_kiro_home() is None
        assert "KIRO_HOME" not in os.environ

    def test_foreign_home_adopts_its_own_kiro_home(self, monkeypatch, tmp_path):
        _relocate_main_homes(monkeypatch, tmp_path)
        scratch = tmp_path / "scratch-home"
        scratch.mkdir()
        monkeypatch.setenv("KIROCREW_HOME", str(scratch))
        monkeypatch.delenv("KIRO_HOME", raising=False)

        adopted = adopt_isolated_kiro_home()

        assert adopted == isolated_kiro_home(scratch.resolve())
        assert os.environ["KIRO_HOME"] == str(adopted)
        # The whole point: the agents dir kiro-cli and every writer now resolve
        # is the dedicated one the write guard's private-target exemption admits,
        # not the machine-wide ``~/.kiro/agents``.
        assert paths.kiro_home() == adopted
        assert paths.ambient_agents_dir() == isolated_agents_dir(scratch.resolve())
        assert paths.kiro_sessions_dir().is_relative_to(adopted)

    def test_an_explicit_kiro_home_is_never_overridden(self, monkeypatch, tmp_path):
        """``KIRO_HOME`` set by the operator is a choice -- including naming the
        shared ``~/.kiro`` on purpose -- and the prologue must not second-guess it."""
        _relocate_main_homes(monkeypatch, tmp_path)
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path / "scratch-home"))
        chosen = tmp_path / "user" / ".kiro"
        monkeypatch.setenv("KIRO_HOME", str(chosen))
        assert adopt_isolated_kiro_home() is None
        assert os.environ["KIRO_HOME"] == str(chosen)

    def test_explicit_kiro_home_memoizes_success_by_raw_override(self, monkeypatch, tmp_path):
        chosen = tmp_path / "chosen-kiro-home"
        chosen.mkdir()
        expected = chosen.resolve()
        monkeypatch.setenv("KIRO_HOME", str(chosen))
        real_resolve = Path.resolve
        resolved: list[str] = []

        def _counting_resolve(self, *args, **kwargs):
            resolved.append(str(self))
            return real_resolve(self, *args, **kwargs)

        monkeypatch.setattr(Path, "resolve", _counting_resolve)

        assert paths._explicit_kiro_home() == expected
        assert paths._explicit_kiro_home() == expected
        assert resolved == [str(chosen)]

        other = tmp_path / "other-kiro-home"
        other.mkdir()
        other_expected = real_resolve(other)
        monkeypatch.setenv("KIRO_HOME", str(other))
        assert paths._explicit_kiro_home() == other_expected
        assert resolved == [str(chosen), str(other)]

    def test_explicit_kiro_home_does_not_cache_resolution_failures(self, monkeypatch, tmp_path):
        chosen = tmp_path / "recovering-kiro-home"
        chosen.mkdir()
        expected = chosen.resolve()
        monkeypatch.setenv("KIRO_HOME", str(chosen))
        attempts = 0

        def _flaky_resolve(self, *args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("temporary resolution failure")
            return expected

        monkeypatch.setattr(Path, "resolve", _flaky_resolve)

        assert paths._explicit_kiro_home() is None
        assert paths._explicit_kiro_home() == expected
        assert attempts == 2

    @pytest.mark.skipif(sys.platform == "win32", reason="/etc is a POSIX system dir")
    def test_an_invalid_kiro_home_is_left_to_the_resolver(self, monkeypatch, tmp_path):
        """``KIRO_HOME=/etc`` is one ``kiro_home()`` discards -- but judging that
        means resolving it, and the prologue does no filesystem work. The declared
        value is left alone; the resolver falls back to the shared ``~/.kiro`` with
        its own warning, and the write guard keeps that dir read-only for a foreign
        home (``test_an_invalid_kiro_home_is_not_an_opt_in``), so the outcome is
        the ``KIRO_HOME=~/.kiro`` opt-out, not a bypass."""
        import os as _os

        _relocate_main_homes(monkeypatch, tmp_path)
        scratch = tmp_path / "scratch-home"
        scratch.mkdir()
        monkeypatch.setenv("KIROCREW_HOME", str(scratch))
        monkeypatch.setenv("KIRO_HOME", "/etc")
        calls: list[str] = []
        for name in ("stat", "lstat", "readlink"):
            real = getattr(_os, name)

            def _spy(*a, _n=name, _real=real, **k):
                calls.append(_n)
                return _real(*a, **k)

            monkeypatch.setattr(_os, name, _spy)

        assert adopt_isolated_kiro_home() is None
        assert os.environ["KIRO_HOME"] == "/etc" and calls == []
        with patch("pathlib.Path.home", return_value=tmp_path / "user"):
            assert paths.kiro_home() == tmp_path / "user" / ".kiro"

    def test_adoption_logs_the_opt_out_once_at_info(self, monkeypatch, tmp_path, caplog):
        """The adoption line names the ``KIRO_HOME=~/.kiro`` opt-out and its cost; the
        upgrade transition itself is reported by ``kirocrew doctor`` (Data Home),
        because telling a first start from a later one would need a filesystem
        probe on the boot path."""
        _relocate_main_homes(monkeypatch, tmp_path)
        scratch = tmp_path / "scratch-home"
        scratch.mkdir()
        monkeypatch.setenv("KIROCREW_HOME", str(scratch))
        monkeypatch.delenv("KIRO_HOME", raising=False)
        with caplog.at_level("INFO", logger="kiro_crew.config.paths"):
            adopt_isolated_kiro_home()
        records = [r for r in caplog.records if r.name == "kiro_crew.config.paths"]
        assert [r.levelname for r in records] == ["INFO"]
        assert "export KIRO_HOME=" in records[0].getMessage()
        assert "Transcripts already moved into" in records[0].getMessage()
        assert "kirocrew doctor" in records[0].getMessage()

    def test_adoption_touches_no_filesystem(self, monkeypatch, tmp_path):
        """The prologue runs before the gateway binds, and a stat or resolve on a
        roaming/UNC home is a network round-trip that would hold readiness
        hostage. After ``config_dir()`` is memoised (which ``ensure_data_home()``
        does first in every prologue), adoption must issue no filesystem call."""
        import os as _os

        _relocate_main_homes(monkeypatch, tmp_path)
        scratch = tmp_path / "scratch-home"
        scratch.mkdir()
        monkeypatch.setenv("KIROCREW_HOME", str(scratch))
        monkeypatch.delenv("KIRO_HOME", raising=False)
        paths.config_dir()  # what ensure_data_home() does: resolve + memoise once
        # The memo re-prime is part of adoption and is held to the same rule: it
        # SEEDS ``hooks``' UNC-root memo from the already-resolved data home. The
        # suite's agents-dir pin would make the seed defer to the resolving
        # prime, so it is lifted here to exercise the production branch.
        import kiro_crew.hooks  # noqa: F401 -- the memo must be loaded to be re-primed

        monkeypatch.setattr(paths, "_agents_dir_override", None)
        # The log line is covered by ``test_adoption_logs_the_opt_out_once_at_info``;
        # silenced here so a stat-ing log handler in the test process (a
        # WatchedFileHandler) cannot be mistaken for adoption's own work.
        monkeypatch.setattr(paths.logger, "info", lambda *a, **k: None)
        # Resolve the expectation BEFORE the spies go in: ``Path.resolve()`` is
        # itself a stat/lstat, and it must not be counted against adoption.
        expected = isolated_kiro_home(scratch.resolve())

        calls: list[str] = []
        for name in ("stat", "lstat", "scandir", "listdir", "readlink", "mkdir"):
            real = getattr(_os, name)

            def _spy(*a, _n=name, _real=real, **k):
                calls.append(_n)
                return _real(*a, **k)

            monkeypatch.setattr(_os, name, _spy)

        adopted = adopt_isolated_kiro_home()
        seen = list(calls)

        assert seen == [], f"adoption touched the filesystem: {seen}"
        assert adopted == expected

    @requires_symlinks
    def test_a_kiro_home_symlink_cycle_is_invalid_not_a_crash(self, monkeypatch, tmp_path):
        """A ``KIRO_HOME`` that cannot be resolved (a link cycle) is a bad value, not
        an abort: the resolver reads it as unset and falls back, and the prologue
        -- which does not resolve it at all -- leaves the declared value alone."""
        _relocate_main_homes(monkeypatch, tmp_path)
        loop = tmp_path / "loop"
        loop.symlink_to(loop)
        scratch = tmp_path / "scratch-home"
        scratch.mkdir()
        monkeypatch.setenv("KIROCREW_HOME", str(scratch))
        monkeypatch.setenv("KIRO_HOME", str(loop))

        assert paths._explicit_kiro_home() is None
        with patch("pathlib.Path.home", return_value=tmp_path / "user"):
            assert paths.kiro_home() == tmp_path / "user" / ".kiro"
        assert adopt_isolated_kiro_home() is None
        assert os.environ["KIRO_HOME"] == str(loop)

    def test_idempotent(self, monkeypatch, tmp_path):
        _relocate_main_homes(monkeypatch, tmp_path)
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path / "scratch-home"))
        monkeypatch.delenv("KIRO_HOME", raising=False)
        first = adopt_isolated_kiro_home()
        assert first is not None
        assert adopt_isolated_kiro_home() is None  # already adopted -> no-op
        assert os.environ["KIRO_HOME"] == str(first)

    def test_isolated_agents_dir_derives_from_isolated_kiro_home(self, tmp_path):
        """One definition of the recipe: the write guard's privacy test and the
        exported ``KIRO_HOME`` cannot drift apart."""
        home = tmp_path / "h"
        assert isolated_agents_dir(home) == isolated_kiro_home(home) / "agents"


# --------------------------------------------------------------------------
# The write guard: a non-default data home does not own the shared agents dir
# --------------------------------------------------------------------------
def _pretend_target_is_shared(monkeypatch, agent_mod, agents_dir: Path) -> None:
    """Same seam ``test_agent_home_isolation`` uses: present *agents_dir* as both the
    write target and what the ambient environment resolves."""
    monkeypatch.setattr(agent_mod, "KIRO_AGENTS_DIR", agents_dir)
    monkeypatch.setattr(agent_mod, "ambient_agents_dir", lambda: agents_dir)


def _durable_primary_checkout(monkeypatch, agent_mod) -> None:
    """The failing shape: NOT a linked worktree, NOT under the temp root."""
    monkeypatch.setattr(agent_mod, "__file__", "/durable-install/KiroCrew/src/kiro_crew/agent.py")


def _capture_sel(monkeypatch, agent_mod) -> list[dict]:
    events: list[dict] = []

    class _Sel:
        def log_api_access(self, **kw):
            events.append(kw)

    monkeypatch.setattr(agent_mod, "sel", lambda: _Sel())
    return events


class TestWriteGuardRefusesForeignHome:
    @staticmethod
    def _seed_default_writer_spec(shared: Path) -> None:
        """A shared spec as the DEFAULT-home writer leaves it: no pinned home.

        Under the guard's provenance arm an EMPTY shared dir is
        writable by design (a fresh relocated install must not be locked out),
        so every refusal test seeds the poisoning-relevant population: specs
        that belong to someone else.
        """
        import json

        shared.mkdir(parents=True, exist_ok=True)
        (shared / "kirocrew.json").write_text(
            json.dumps(
                {"name": "kirocrew", "mcpServers": {"kirocrew-core": {"command": "kirocrew"}}}
            ),
            encoding="utf-8",
        )

    def test_scratch_home_without_kiro_home_is_declined_and_audited(self, monkeypatch, tmp_path):
        """The reproduction, at the guard: ``KIROCREW_HOME=<scratch>``, no
        ``KIRO_HOME``, durable checkout, shared target -> refused, not written."""
        from kiro_crew import agent

        _relocate_main_homes(monkeypatch, tmp_path)
        events = _capture_sel(monkeypatch, agent)
        scratch = tmp_path / "scratch-home"
        scratch.mkdir()
        monkeypatch.setenv("KIROCREW_HOME", str(scratch))
        monkeypatch.delenv("KIRO_HOME", raising=False)
        monkeypatch.delenv("KIROCREW_POD", raising=False)
        _durable_primary_checkout(monkeypatch, agent)
        shared = tmp_path / "user" / ".kiro" / "agents"
        _pretend_target_is_shared(monkeypatch, agent, shared)
        self._seed_default_writer_spec(shared)

        declined = agent._decline_shared_agent_home()

        assert declined == shared / agent.AGENT_FILENAME
        denied = [e for e in events if e.get("outcome") == "denied"]
        assert len(denied) == 1, events
        assert denied[0]["operation"] == "agent_home_write"
        assert str(shared) in denied[0]["resources"]
        # The provenance arm's audit names the refusing home, not the isolated
        # target (that guidance lives in the human-facing warning).
        assert str(scratch.resolve()) in denied[0]["error"]

    def test_rebuild_writes_nothing_from_a_scratch_home(self, monkeypatch, tmp_path):
        """End to end through ``rebuild_agent_config``: the shared dir is untouched."""
        from kiro_crew import agent

        _relocate_main_homes(monkeypatch, tmp_path)
        _capture_sel(monkeypatch, agent)
        scratch = tmp_path / "scratch-home"
        scratch.mkdir()
        monkeypatch.setenv("KIROCREW_HOME", str(scratch))
        monkeypatch.delenv("KIRO_HOME", raising=False)
        monkeypatch.delenv("KIROCREW_POD", raising=False)
        _durable_primary_checkout(monkeypatch, agent)
        shared = tmp_path / "user" / ".kiro" / "agents"
        _pretend_target_is_shared(monkeypatch, agent, shared)
        self._seed_default_writer_spec(shared)
        before = (shared / "kirocrew.json").read_bytes()

        returned = agent.rebuild_agent_config()

        assert returned == shared / agent.AGENT_FILENAME
        assert (
            shared / "kirocrew.json"
        ).read_bytes() == before, (
            "a non-default data home must not rewrite someone else's shared specs"
        )

    def test_the_refusal_names_the_remedy(self, monkeypatch, tmp_path, caplog):
        from kiro_crew import agent

        _relocate_main_homes(monkeypatch, tmp_path)
        _capture_sel(monkeypatch, agent)
        scratch = tmp_path / "scratch-home"
        scratch.mkdir()
        monkeypatch.setenv("KIROCREW_HOME", str(scratch))
        monkeypatch.delenv("KIRO_HOME", raising=False)
        monkeypatch.delenv("KIROCREW_POD", raising=False)
        _durable_primary_checkout(monkeypatch, agent)
        shared = tmp_path / "user" / ".kiro" / "agents"
        _pretend_target_is_shared(monkeypatch, agent, shared)
        self._seed_default_writer_spec(shared)

        with caplog.at_level("WARNING", logger="kiro_crew.agent"):
            agent._decline_shared_agent_home()

        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "non-default" in text
        assert "#9690" in text

    def test_adopted_kiro_home_makes_the_target_private(self, monkeypatch, tmp_path):
        """With the prologue's export in force the instance writes its OWN specs:
        the target is ``isolated_agents_dir(own home)`` and the guard stands aside.
        Being refused would not be harmless here -- it would hand this instance the
        shared spec, whose env pins the LIVE data home."""
        from kiro_crew import agent

        _relocate_main_homes(monkeypatch, tmp_path)
        scratch = tmp_path / "scratch-home"
        scratch.mkdir()
        monkeypatch.setenv("KIROCREW_HOME", str(scratch))
        monkeypatch.delenv("KIRO_HOME", raising=False)
        monkeypatch.delenv("KIROCREW_POD", raising=False)
        assert adopt_isolated_kiro_home() is not None
        _durable_primary_checkout(monkeypatch, agent)
        _pretend_target_is_shared(monkeypatch, agent, isolated_agents_dir(scratch.resolve()))

        assert agent._decline_shared_agent_home() is None

    @requires_pinned_walk
    @requires_symlinks
    def test_private_exemption_pins_write_across_parent_swap(self, monkeypatch, tmp_path):
        """A validated private target cannot be swapped onto the shared agents dir.

        The swap lands after the ownership decision and before publication. The
        rebuild must keep writing through the descriptor that decision inspected,
        rather than re-opening the now-hostile path.
        """
        from kiro_crew import agent

        _relocate_main_homes(monkeypatch, tmp_path)
        _capture_sel(monkeypatch, agent)
        scratch = tmp_path / "scratch-home"
        scratch.mkdir()
        private_kiro = isolated_kiro_home(scratch)
        private_agents = private_kiro / "agents"
        private_agents.mkdir(parents=True)
        shared_kiro = tmp_path / "user" / ".kiro"
        shared_agents = shared_kiro / "agents"
        self._seed_default_writer_spec(shared_agents)
        monkeypatch.setenv("KIROCREW_HOME", str(scratch))
        monkeypatch.setenv("KIRO_HOME", str(private_kiro))
        monkeypatch.delenv("KIROCREW_POD", raising=False)
        _durable_primary_checkout(monkeypatch, agent)
        _pretend_target_is_shared(monkeypatch, agent, private_agents)
        shared_before = {path.name: path.read_bytes() for path in shared_agents.iterdir()}
        held_kiro = scratch / "kiro-held"
        real_decline = agent._decline_shared_agent_home
        swapped = False

        def _swap_after_decision(*args, **kwargs):
            nonlocal swapped
            declined = real_decline(*args, **kwargs)
            if declined is None and not swapped:
                private_kiro.rename(held_kiro)
                private_kiro.symlink_to(shared_kiro, target_is_directory=True)
                swapped = True
            return declined

        monkeypatch.setattr(agent, "_decline_shared_agent_home", _swap_after_decision)

        returned = agent.rebuild_agent_config(refresh_forks=False)

        assert returned == private_agents / agent.AGENT_FILENAME
        assert swapped
        assert {path.name: path.read_bytes() for path in shared_agents.iterdir()} == shared_before
        assert (held_kiro / "agents" / agent.AGENT_FILENAME).is_file()

    def test_an_explicit_kiro_home_is_not_write_authorization(self, monkeypatch, tmp_path):
        """``KIRO_HOME=~/.kiro`` from a foreign data home makes kiro-cli READ the
        shared specs; it never makes this instance their writer. An environment
        variable is set by whoever launched the process -- an agent running
        ``KIRO_HOME=$HOME/.kiro kirocrew setup --agent-only`` included -- so it
        cannot be the thing that authorises rewriting the shared file with this
        home pinned into it. Ownership is the data home, nothing else."""
        from kiro_crew import agent

        _relocate_main_homes(monkeypatch, tmp_path)
        events = _capture_sel(monkeypatch, agent)
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path / "relocated-home"))
        shared_kiro = tmp_path / "user" / ".kiro"
        monkeypatch.setenv("KIRO_HOME", str(shared_kiro))
        monkeypatch.delenv("KIROCREW_POD", raising=False)
        _durable_primary_checkout(monkeypatch, agent)
        _pretend_target_is_shared(monkeypatch, agent, shared_kiro / "agents")
        self._seed_default_writer_spec(shared_kiro / "agents")

        assert agent._decline_shared_agent_home() is not None
        assert [e["outcome"] for e in events] == ["denied"]

    @requires_symlinks
    def test_a_planted_link_under_the_data_home_gets_no_exemption(self, monkeypatch, tmp_path):
        """An agent with write access to the data home plants
        ``<data home>/kiro -> ~/.kiro``. With ``KIRO_HOME=<data home>/kiro`` every
        path resolves onto the shared tree, so a RESOLVED comparison would read the
        machine-wide agents dir as this instance's private one. The exemption
        requires the isolated path to be link-free."""
        from kiro_crew import agent

        _relocate_main_homes(monkeypatch, tmp_path)
        events = _capture_sel(monkeypatch, agent)
        shared_kiro = tmp_path / "user" / ".kiro"
        (shared_kiro / "agents").mkdir(parents=True)
        scratch = tmp_path / "scratch-home"
        scratch.mkdir()
        isolated_kiro_home(scratch).symlink_to(shared_kiro, target_is_directory=True)
        monkeypatch.setenv("KIROCREW_HOME", str(scratch))
        monkeypatch.setenv("KIRO_HOME", str(isolated_kiro_home(scratch)))
        monkeypatch.delenv("KIROCREW_POD", raising=False)
        _durable_primary_checkout(monkeypatch, agent)
        _pretend_target_is_shared(monkeypatch, agent, shared_kiro / "agents")
        self._seed_default_writer_spec(shared_kiro / "agents")

        assert agent._decline_shared_agent_home() is not None
        assert [e["outcome"] for e in events] == ["denied"]

    def test_a_regular_file_at_the_isolated_home_gets_no_exemption(self, monkeypatch, tmp_path):
        """A stray regular file at ``<data home>/kiro``: ``resolve()`` does not
        notice, the exemption would pass, and the writer's ``mkdir(parents=True)``
        would then crash the boot with ``NotADirectoryError``. The exemption asks
        for directory shape explicitly, so the target stays UNEXEMPT -- whether a
        write then proceeds is the provenance arm's ownership call, not the
        exemption's."""
        from kiro_crew import agent

        _relocate_main_homes(monkeypatch, tmp_path)
        scratch = tmp_path / "scratch-home"
        scratch.mkdir()
        isolated_kiro_home(scratch).write_text("not a directory", encoding="utf-8")
        monkeypatch.setenv("KIROCREW_HOME", str(scratch))
        monkeypatch.setenv("KIRO_HOME", str(isolated_kiro_home(scratch)))
        monkeypatch.delenv("KIROCREW_POD", raising=False)
        _durable_primary_checkout(monkeypatch, agent)
        _pretend_target_is_shared(monkeypatch, agent, isolated_agents_dir(scratch))

        assert agent._private_isolated_agents_dir(scratch.resolve()) is None
        assert agent._unexempt_shared_target(isolated_agents_dir(scratch)) is not None

    def test_default_home_still_owns_the_shared_dir(self, monkeypatch, tmp_path):
        """The ordinary install is unchanged: no override, durable checkout -> writes."""
        from kiro_crew import agent

        _relocate_main_homes(monkeypatch, tmp_path)
        events = _capture_sel(monkeypatch, agent)
        monkeypatch.delenv("KIROCREW_HOME", raising=False)
        monkeypatch.delenv("KIRO_HOME", raising=False)
        monkeypatch.delenv("KIROCREW_POD", raising=False)
        _durable_primary_checkout(monkeypatch, agent)
        _pretend_target_is_shared(monkeypatch, agent, tmp_path / "user" / ".kiro" / "agents")

        assert agent._decline_shared_agent_home() is None
        assert [e["outcome"] for e in events] == ["allowed"]

    @pytest.mark.skipif(sys.platform == "win32", reason="/etc is a POSIX system dir")
    def test_an_invalid_kiro_home_is_not_an_opt_in(self, monkeypatch, tmp_path):
        """``KIRO_HOME=/etc`` is discarded by ``kiro_home()``, so the target is the
        shared dir after all; the raw variable being set must not read as the
        operator's consent to write it."""
        from kiro_crew import agent

        _relocate_main_homes(monkeypatch, tmp_path)
        events = _capture_sel(monkeypatch, agent)
        monkeypatch.setenv("KIROCREW_HOME", str(tmp_path / "scratch-home"))
        monkeypatch.setenv("KIRO_HOME", "/etc")
        monkeypatch.delenv("KIROCREW_POD", raising=False)
        _durable_primary_checkout(monkeypatch, agent)
        shared = tmp_path / "user" / ".kiro" / "agents"
        _pretend_target_is_shared(monkeypatch, agent, shared)
        self._seed_default_writer_spec(shared)

        assert agent._decline_shared_agent_home() is not None
        assert [e["outcome"] for e in events] == ["denied"]


class TestAppAgentWritersHonourOwnership:
    """``apps.bridges`` is the other writer of the agents dir. Pointed at the SHARED
    directory from a foreign home -- the ``KIRO_HOME=~/.kiro`` opt-out, or a
    prologue-bypassing caller -- it must neither materialise, prune nor remove app
    specs there: two instances with different app sets would otherwise fight over
    the default instance's files."""

    @staticmethod
    def _foreign_on_shared(monkeypatch, tmp_path):
        from kiro_crew import agent
        from kiro_crew.apps import bridges

        _relocate_main_homes(monkeypatch, tmp_path)
        scratch = tmp_path / "scratch-home"
        scratch.mkdir()
        monkeypatch.setenv("KIROCREW_HOME", str(scratch))
        shared = tmp_path / "user" / ".kiro" / "agents"
        shared.mkdir(parents=True)
        monkeypatch.setenv("KIRO_HOME", str(shared.parent))
        _pretend_target_is_shared(monkeypatch, agent, shared)
        monkeypatch.setattr(bridges, "KIRO_AGENTS_DIR", shared)
        return agent, bridges, shared, scratch

    def test_predicate_names_the_foreign_home_for_the_shared_dir(self, monkeypatch, tmp_path):
        agent, _, shared, scratch = self._foreign_on_shared(monkeypatch, tmp_path)
        assert agent.foreign_home_targets_shared_agents_dir(shared) == scratch.resolve()
        # A redirect of the caller's own is not the shared dir.
        assert agent.foreign_home_targets_shared_agents_dir(tmp_path / "elsewhere") is None
        # The instance's dedicated dir is its own to write.
        own = isolated_agents_dir(scratch.resolve())
        own.mkdir(parents=True)
        monkeypatch.setattr(agent, "ambient_agents_dir", lambda: own)
        assert agent.foreign_home_targets_shared_agents_dir(own) is None

    def test_predicate_is_silent_on_the_default_home(self, monkeypatch, tmp_path):
        from kiro_crew import agent

        _relocate_main_homes(monkeypatch, tmp_path)
        monkeypatch.delenv("KIROCREW_HOME", raising=False)
        shared = tmp_path / "user" / ".kiro" / "agents"
        _pretend_target_is_shared(monkeypatch, agent, shared)
        assert agent.foreign_home_targets_shared_agents_dir(shared) is None

    @staticmethod
    def _denials(events: list[dict]) -> list[dict]:
        return [e for e in events if e.get("outcome") == "denied"]

    def test_register_writes_nothing_into_the_shared_dir(self, monkeypatch, tmp_path, caplog):
        from types import SimpleNamespace

        _, bridges, shared, scratch = self._foreign_on_shared(monkeypatch, tmp_path)
        events = _capture_sel(monkeypatch, bridges)
        before = sorted(p.name for p in shared.iterdir())
        with caplog.at_level("WARNING", logger=bridges.logger.name):
            registered = bridges._register_agents(
                "some-app", SimpleNamespace(agents=["agents/a.json"]), tmp_path / "app"
            )
        assert registered == []
        assert sorted(p.name for p in shared.iterdir()) == before
        assert "not writing agent specs into the shared" in caplog.text
        # The refusal is a permission decision on the shared agent home, audited
        # like ``agent._decline_shared_agent_home``'s and not only logged.
        denied = self._denials(events)
        assert len(denied) == 1, events
        assert denied[0]["operation"] == "agent_home_write"
        assert denied[0]["source"] == "register_agents"
        assert denied[0]["resources"] == str(shared)
        assert str(scratch.resolve()) in denied[0]["error"]
        assert str(isolated_agents_dir(scratch.resolve())) in denied[0]["error"]

    def test_mcp_registration_writes_nothing_into_the_shared_dir(self, monkeypatch, tmp_path):
        from kiro_crew.apps.manifest import AppManifest

        _, bridges, shared, scratch = self._foreign_on_shared(monkeypatch, tmp_path)
        events = _capture_sel(monkeypatch, bridges)
        target = shared / "kirocrew.json"
        target.write_text('{"mcpServers": {"kept": {"command": "kept"}}}\n', encoding="utf-8")
        before = target.read_bytes()
        manifest = AppManifest(
            name="some-app",
            mcpServers={"tools": {"command": sys.executable, "args": ["-V"]}},
        )

        assert bridges._register_mcp_servers("some-app", manifest) == []
        assert target.read_bytes() == before
        denied = self._denials(events)
        assert len(denied) == 1
        assert denied[0]["source"] == "register_mcp_servers"
        assert denied[0]["resources"] == str(shared)
        assert str(scratch.resolve()) in denied[0]["error"]

    def test_deregister_and_prune_leave_the_shared_dir_alone(self, monkeypatch, tmp_path):
        from types import SimpleNamespace

        _, bridges, shared, _ = self._foreign_on_shared(monkeypatch, tmp_path)
        events = _capture_sel(monkeypatch, bridges)
        theirs = shared / (bridges._safe_link_name("some-app/agent") + ".json")
        theirs.write_text("{}", encoding="utf-8")
        mcp_spec = shared / "kirocrew.json"
        mcp_spec.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "some-app:old": {"command": "old"},
                        "kept": {"command": "kept"},
                    }
                }
            ),
            encoding="utf-8",
        )
        before = mcp_spec.read_bytes()

        assert bridges._deregister_agents("some-app") == 0
        assert bridges._deregister_mcp_servers("some-app") == 0
        bridges._prune_stale_app_resources(
            "some-app", SimpleNamespace(agents=[], skills=[], mcpServers={}), tmp_path / "app"
        )
        assert theirs.exists()
        assert mcp_spec.read_bytes() == before
        # Each writer that stood down left its own audit row, named by ``source``.
        denied = self._denials(events)
        assert [e["source"] for e in denied] == [
            "deregister_agents",
            "deregister_mcp_servers",
            "prune_stale_app_resources",
        ]
        assert {e["operation"] for e in denied} == {"agent_home_write"}
        assert {e["resources"] for e in denied} == {str(shared)}

    def test_own_agents_dir_is_written_without_an_audit_row(self, monkeypatch, tmp_path):
        """The audit is for refusals over the SHARED home. A foreign home writing its
        own ``isolated_agents_dir`` is not a decision about a shared resource, so it
        leaves no row -- the same asymmetry as the core writer's private-target
        return."""
        agent, bridges, _, scratch = self._foreign_on_shared(monkeypatch, tmp_path)
        events = _capture_sel(monkeypatch, bridges)
        own = isolated_agents_dir(scratch.resolve())
        own.mkdir(parents=True)
        monkeypatch.setattr(agent, "ambient_agents_dir", lambda: own)
        monkeypatch.setattr(bridges, "KIRO_AGENTS_DIR", own)
        theirs = own / (bridges._safe_link_name("some-app/agent") + ".json")
        theirs.write_text("{}", encoding="utf-8")

        assert bridges._deregister_agents("some-app") == 1
        assert not theirs.exists()
        assert self._denials(events) == []


# --------------------------------------------------------------------------
# kirocrew doctor: name the drifted pin
# --------------------------------------------------------------------------
def _write_spec(agents_dir: Path, name: str, servers: dict) -> Path:
    agents_dir.mkdir(parents=True, exist_ok=True)
    p = agents_dir / name
    p.write_text(json.dumps({"name": name[:-5], "mcpServers": servers}), encoding="utf-8")
    return p


class TestDoctorSpecHomeDrift:
    @pytest.fixture
    def own_home(self, monkeypatch, tmp_path: Path) -> Path:
        home = tmp_path / "gateway-home"
        home.mkdir()
        monkeypatch.setenv("KIROCREW_HOME", str(home))
        return home.resolve()

    def test_a_managed_spec_pinning_another_home_is_drift(self, own_home, tmp_path):
        from kiro_crew.doctor_spec_home import check_spec_home_drift

        agents = tmp_path / "agents"
        foreign = tmp_path / "scratch" / "runtime-3c0a7e8b" / "pfshot" / "home"
        _write_spec(
            agents,
            "kirocrew.json",
            {
                "kirocrew-core": {"command": "kirocrew", "env": {"KIROCREW_HOME": str(foreign)}},
                "kirocrew-cron": {"command": "kirocrew", "env": {"KIROCREW_HOME": str(foreign)}},
                "other-tool": {"command": "x"},
            },
        )

        report = check_spec_home_drift(agents_dir=agents)

        assert report.expected == str(own_home)
        assert report.scanned == 1
        assert sorted(d.server for d in report.managed) == ["kirocrew-core", "kirocrew-cron"]
        assert all(d.pinned == str(foreign) and d.managed for d in report.drift)
        assert report.foreign == []

    def test_a_matching_pin_and_no_pin_are_both_healthy(self, own_home, tmp_path):
        from kiro_crew.doctor_spec_home import check_spec_home_drift

        agents = tmp_path / "agents"
        _write_spec(
            agents,
            "kirocrew.json",
            {
                # Pinned to OUR home, spelled with a trailing slash.
                "kirocrew-core": {"command": "kirocrew", "env": {"KIROCREW_HOME": f"{own_home}/"}},
                # A default-home writer pins nothing; not a finding.
                "kirocrew-cron": {"command": "kirocrew"},
            },
        )
        report = check_spec_home_drift(agents_dir=agents)
        assert report.drift == []
        assert report.scanned == 1

    def test_a_foreign_spec_is_reported_apart_from_managed_ones(self, own_home, tmp_path):
        from kiro_crew.doctor_spec_home import check_spec_home_drift

        agents = tmp_path / "agents"
        _write_spec(
            agents,
            "some-aim-agent.json",
            {"kirocrew-core": {"command": "kirocrew", "env": {"KIROCREW_HOME": "/elsewhere"}}},
        )
        report = check_spec_home_drift(agents_dir=agents)
        assert report.managed == []
        assert [d.spec for d in report.foreign] == ["some-aim-agent.json"]

    def test_foreign_drift_is_named_but_never_an_issue(self, own_home, tmp_path, capsys):
        """A third-party spec is not this install's to repair and
        ``setup --agent-only`` does not touch it, so doctor prints the ⚠️ line and
        exits zero -- a nonzero exit the operator cannot clear would only teach
        them to ignore the section."""
        from kiro_crew.doctor_spec_home import doctor_spec_home_drift

        agents = tmp_path / "agents"
        _write_spec(
            agents,
            "some-aim-agent.json",
            {"kirocrew-core": {"command": "kirocrew", "env": {"KIROCREW_HOME": "/elsewhere"}}},
        )
        issues: list[str] = []
        doctor_spec_home_drift(issues, agents_dir=agents)
        out = capsys.readouterr().out
        assert "some-aim-agent.json" in out and "foreign spec" in out
        assert issues == []

    def test_a_custom_server_inside_an_owned_spec_is_not_managed(self, own_home, tmp_path):
        """``setup --agent-only`` rewrites only Kiro Crew's OWN server entries and
        preserves a user-added one, so a drifted pin on a custom server inside
        ``kirocrew.json`` would survive the remedy doctor names for managed
        drift. It is reported apart, as edit-by-hand."""
        from kiro_crew.doctor_spec_home import check_spec_home_drift

        agents = tmp_path / "agents"
        _write_spec(
            agents,
            "kirocrew.json",
            {
                "kirocrew-core": {"command": "kirocrew", "env": {"KIROCREW_HOME": "/elsewhere"}},
                "my-custom-srv": {"command": "srv", "env": {"KIROCREW_HOME": "/elsewhere"}},
            },
        )
        report = check_spec_home_drift(agents_dir=agents)
        assert [d.server for d in report.managed] == ["kirocrew-core"]
        assert [d.server for d in report.foreign] == ["my-custom-srv"]

    def test_a_pin_is_compared_lexically_and_never_touches_the_filesystem(
        self, own_home, tmp_path, monkeypatch
    ):
        """A spec pin is untrusted text. On Windows ``Path.resolve()`` OPENS a
        UNC path, so a pin of ``\\\\attacker\\share`` would authenticate to that
        host during a doctor walk. The comparison must be lexical: no stat, no
        resolve on the pin -- and the pin is still reported as drift."""
        import os as _os

        from kiro_crew import doctor_spec_home

        agents = tmp_path / "agents"
        unc = "\\\\attacker\\share\\home"
        _write_spec(agents, "kirocrew.json", {"kirocrew-core": {"env": {"KIROCREW_HOME": unc}}})

        touched: list[str] = []
        real_stat = _os.stat

        def _spy(path, *a, **k):
            touched.append(str(path))
            return real_stat(path, *a, **k)

        monkeypatch.setattr(_os, "stat", _spy)
        report = doctor_spec_home.check_spec_home_drift(agents_dir=agents)

        assert [d.pinned for d in report.drift] == [unc]
        assert not [p for p in touched if "attacker" in p], touched
        # The normaliser itself is filesystem-free by construction (code body,
        # after its docstring).
        src = Path(doctor_spec_home.__file__).read_text(encoding="utf-8")
        body = src.split("def _lexical(")[1].split("\ndef ")[0].split('"""')[-1]
        assert "resolve(" not in body and "stat(" not in body and "exists(" not in body

    def test_a_tilde_user_pin_never_consults_the_account_database(self, monkeypatch):
        """``~name/...`` makes ``os.path.expanduser`` look the name up in the
        account database -- an external probe on spec-supplied text. Only a bare
        ``~`` / ``~/`` prefix is expanded; anything else compares as written."""
        import os as _os

        from kiro_crew import doctor_spec_home

        def _boom(*a, **k):  # pragma: no cover - the assertion is that it is not called
            raise AssertionError("expanduser consulted for a ~name pin")

        monkeypatch.setattr(_os.path, "expanduser", _boom)
        assert doctor_spec_home._lexical("~attacker/crew") == _os.path.normcase(
            _os.path.normpath("~attacker/crew")
        )
        assert doctor_spec_home._lexical("/plain/home/") == _os.path.normcase("/plain/home")

    def test_a_bare_tilde_pin_still_expands(self, monkeypatch, tmp_path):
        from kiro_crew import doctor_spec_home

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        expected = doctor_spec_home._lexical(tmp_path / ".kiro" / "crew")
        assert doctor_spec_home._lexical("~/.kiro/crew") == expected

    def test_malformed_specs_are_skipped_not_fatal(self, own_home, tmp_path):
        from kiro_crew.doctor_spec_home import check_spec_home_drift

        agents = tmp_path / "agents"
        agents.mkdir()
        (agents / "broken.json").write_text("{not json", encoding="utf-8")
        (agents / "list.json").write_text("[1, 2]", encoding="utf-8")
        _write_spec(agents, "kirocrew.json", {"kirocrew-core": {"env": "not-an-object"}})
        report = check_spec_home_drift(agents_dir=agents)
        assert report.scanned == 1  # only kirocrew.json parsed as an object
        assert report.drift == []

    def test_missing_agents_dir_is_empty_not_an_error(self, own_home, tmp_path):
        from kiro_crew.doctor_spec_home import check_spec_home_drift

        report = check_spec_home_drift(agents_dir=tmp_path / "nope")
        assert report.scanned == 0 and report.drift == []

    def test_renderer_flags_managed_drift_with_the_remedy(self, own_home, tmp_path, capsys):
        from kiro_crew.doctor_spec_home import doctor_spec_home_drift

        agents = tmp_path / "agents"
        _write_spec(
            agents,
            "kirocrew.json",
            {"kirocrew-core": {"command": "kirocrew", "env": {"KIROCREW_HOME": "/scratch/x"}}},
        )
        issues: list[str] = []
        doctor_spec_home_drift(issues, agents_dir=agents)
        out = capsys.readouterr().out
        assert "Agent Spec Data Home" in out
        assert "kirocrew.json" in out and "KIROCREW_HOME=/scratch/x" in out
        assert "kirocrew setup --agent-only" in out
        assert issues == ["agent specs pin a different KIROCREW_HOME than this data home"]

    def test_renderer_is_green_and_silent_in_issues_when_healthy(self, own_home, tmp_path, capsys):
        from kiro_crew.doctor_spec_home import doctor_spec_home_drift

        agents = tmp_path / "agents"
        _write_spec(agents, "kirocrew.json", {"kirocrew-core": {"command": "kirocrew"}})
        issues: list[str] = []
        doctor_spec_home_drift(issues, agents_dir=agents)
        assert "✅" in capsys.readouterr().out
        assert issues == []

    def test_renderer_does_not_claim_success_when_no_managed_server_was_verified(
        self, own_home, tmp_path, capsys
    ):
        from kiro_crew.doctor_spec_home import doctor_spec_home_drift

        agents = tmp_path / "agents"
        _write_spec(agents, "kirocrew.json", {"custom": {"command": "custom"}})
        doctor_spec_home_drift([], agents_dir=agents)
        out = capsys.readouterr().out
        assert "no managed MCP servers found" in out
        assert "every managed MCP server resolves" not in out

    def test_drift_retention_bounds_count_and_each_string(self, own_home, tmp_path, capsys):
        from kiro_crew import doctor_spec_home

        agents = tmp_path / "agents"
        total = doctor_spec_home.DRIFT_REPORT_MAX_ITEMS + 3
        long_tail = "x" * (doctor_spec_home.DRIFT_FIELD_MAX_CHARS * 2)
        servers = {
            f"custom-{i}-{long_tail}": {"env": {"KIROCREW_HOME": f"/foreign/{i}/{long_tail}"}}
            for i in range(total)
        }
        _write_spec(agents, "foreign.json", servers)

        report = doctor_spec_home.check_spec_home_drift(agents_dir=agents)

        assert len(report.drift) == doctor_spec_home.DRIFT_REPORT_MAX_ITEMS
        assert report.managed_omitted == 0
        assert report.foreign_omitted == 3
        assert all(
            len(value) <= doctor_spec_home.DRIFT_FIELD_MAX_CHARS
            for drift in report.drift
            for value in (drift.spec, drift.server, drift.pinned)
        )
        doctor_spec_home.doctor_spec_home_drift([], agents_dir=agents)
        assert "+3 more drifted MCP server(s) omitted" in capsys.readouterr().out

    def test_managed_drift_is_admitted_before_foreign_rows_fill_the_cap(
        self, own_home, tmp_path, capsys
    ):
        """The rows the cap keeps are the actionable ones. Spec files are walked in
        name order, so five foreign rows (an ``app-*.json`` sorts before
        ``kirocrew.json``) would otherwise fill ``DRIFT_REPORT_MAX_ITEMS`` and push
        the managed drift into ``managed_omitted`` -- while the renderer printed
        the managed remedy under rows that name no managed server. Managed rows
        are admitted first, foreign rows take what is left, and the overflow
        line counts exactly what was dropped."""
        from kiro_crew import doctor_spec_home

        agents = tmp_path / "agents"
        cap = doctor_spec_home.DRIFT_REPORT_MAX_ITEMS
        _write_spec(
            agents,
            "app-third-party.json",
            {f"custom-{i}": {"env": {"KIROCREW_HOME": f"/foreign/{i}"}} for i in range(cap + 1)},
        )
        _write_spec(
            agents,
            "kirocrew.json",
            {
                "kirocrew-core": {"command": "kirocrew", "env": {"KIROCREW_HOME": "/stale"}},
                "kirocrew-cron": {"command": "kirocrew", "env": {"KIROCREW_HOME": "/stale"}},
            },
        )

        report = doctor_spec_home.check_spec_home_drift(agents_dir=agents)

        assert [d.server for d in report.managed] == ["kirocrew-core", "kirocrew-cron"]
        assert len(report.drift) == cap
        assert len(report.foreign) == cap - 2
        assert report.managed_omitted == 0
        assert report.foreign_omitted == 3  # (cap + 1) foreign rows, cap - 2 admitted
        issues: list[str] = []
        doctor_spec_home.doctor_spec_home_drift(issues, agents_dir=agents)
        out = capsys.readouterr().out
        assert "kirocrew-core pins" in out and "kirocrew-cron pins" in out
        assert "+3 more drifted MCP server(s) omitted" in out
        assert issues == ["agent specs pin a different KIROCREW_HOME than this data home"]

    def test_drift_scan_streams_and_bounds_the_managed_first_window(
        self, own_home, tmp_path, monkeypatch, capsys
    ):
        """Scanning and row retention stay bounded while preserving old output order.

        The scandir double refuses to yield entry N+1 until entry N was read. The
        old eager ``sorted(os.scandir(...))`` path therefore fails before it can
        read even one spec. The retention wrapper separately checks the live
        collection after every drift row, including replacements of earlier
        foreign rows by later managed rows.
        """
        from kiro_crew import doctor_spec_home

        agents = tmp_path / "agents"
        cap = doctor_spec_home.DRIFT_REPORT_MAX_ITEMS
        foreign_total = cap + 3
        for index in range(foreign_total):
            _write_spec(
                agents,
                f"app-{index:02d}.json",
                {f"custom-{index}": {"env": {"KIROCREW_HOME": f"/foreign/{index}"}}},
            )
        managed_servers = sorted(doctor_spec_home.MANAGED_MCP_SERVER_NAMES)[: cap + 2]
        _write_spec(
            agents,
            "kirocrew.json",
            {name: {"env": {"KIROCREW_HOME": f"/managed/{name}"}} for name in managed_servers},
        )

        paths = sorted(agents.glob("*.json"), key=lambda path: path.name)
        real_read = doctor_spec_home._read_agent_spec
        reads: list[str] = []

        def _counted_read(path, **kwargs):
            data = real_read(path, **kwargs)
            reads.append(path.name)
            return data

        class _Entry:
            def __init__(self, path: Path) -> None:
                self.name = path.name
                self.path = str(path)

        class _StreamingScandir:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def __iter__(self):
                start = len(reads)
                for index, path in enumerate(paths):
                    if index:
                        assert (
                            len(reads) == start + index
                        ), "the directory was materialized before specs were read"
                    yield _Entry(path)

        retained_sizes: list[int] = []
        real_retain = getattr(doctor_spec_home, "_retain_drift", None)

        def _guarded_retain(retained, *, rank, drift):
            assert len(retained) <= cap
            assert real_retain is not None
            real_retain(retained, rank=rank, drift=drift)
            retained_sizes.append(len(retained))
            assert len(retained) <= cap

        real_scandir = doctor_spec_home.os.scandir
        monkeypatch.setattr(doctor_spec_home, "_read_agent_spec", _counted_read)
        if real_retain is not None:
            monkeypatch.setattr(doctor_spec_home, "_retain_drift", _guarded_retain)

        # The scandir fake is scoped to this one call: doctor_spec_home.os IS
        # the global os module, and pytest's tmp_path teardown runs
        # shutil.rmtree, whose directory walk would otherwise meet the fake
        # (Windows rmtree drives os.scandir as an iterator and crashes the
        # teardown).
        with pytest.MonkeyPatch.context() as scandir_patch:
            scandir_patch.setattr(
                doctor_spec_home.os,
                "scandir",
                lambda directory: (
                    _StreamingScandir()
                    if isinstance(directory, (str, os.PathLike)) and Path(directory) == agents
                    else real_scandir(directory)
                ),
            )
            report = doctor_spec_home.check_spec_home_drift(agents_dir=agents)

        assert reads == [path.name for path in paths]
        assert len(retained_sizes) == foreign_total + len(managed_servers)
        assert max(retained_sizes) == cap
        assert [drift.server for drift in report.drift] == managed_servers[:cap]
        assert report.managed_omitted == len(managed_servers) - cap
        assert report.foreign_omitted == foreign_total
        doctor_spec_home.doctor_spec_home_drift([], agents_dir=agents)
        omitted = len(managed_servers) + foreign_total - cap
        assert f"+{omitted} more drifted MCP server(s) omitted" in capsys.readouterr().out

    def test_renderer_neutralizes_terminal_controls_in_a_pin(self, own_home, tmp_path, capsys):
        """A spec is untrusted input; an escape byte in its pin must not reach the
        terminal raw (same rule as the dead-path check)."""
        from kiro_crew.doctor_spec_home import doctor_spec_home_drift

        agents = tmp_path / "agents"
        _write_spec(
            agents,
            "kirocrew.json",
            {"kirocrew-core": {"env": {"KIROCREW_HOME": "/x\x1b]0;pwned\x07"}}},
        )
        doctor_spec_home_drift([], agents_dir=agents)
        out = capsys.readouterr().out
        assert "\x1b" not in out and "\\x1b" in out

    @requires_symlinks
    def test_specs_are_read_through_the_hardened_gate(self, own_home, tmp_path, monkeypatch):
        """Every spec read goes through ``agent_discovery._read_agent_spec`` -- the
        one reader that refuses a symlink whose RESOLVED target is sensitive and
        caps the size -- never a bare ``read_text``. Driven through the reader's
        own sensitive-path refusal, the way the hardened-read suite does."""
        from kiro_crew import agent_discovery
        from kiro_crew.doctor_spec_home import check_spec_home_drift

        agents = tmp_path / "agents"
        agents.mkdir()
        protected = tmp_path / "protected.json"
        protected.write_text(
            json.dumps({"mcpServers": {"kirocrew-core": {"env": {"KIROCREW_HOME": "/x"}}}}),
            encoding="utf-8",
        )
        (agents / "kirocrew.json").symlink_to(protected)
        monkeypatch.setattr(
            agent_discovery,
            "is_sensitive_canonical_path",
            lambda p: str(protected) in str(p),
        )

        report = check_spec_home_drift(agents_dir=agents)

        assert report.scanned == 0, "a link to a protected target must not be parsed"
        assert report.drift == []


# --------------------------------------------------------------------------
# Wiring ratchets
# --------------------------------------------------------------------------
def test_cli_prologue_adopts_the_kiro_home_after_the_data_home():
    """Every ``kirocrew`` verb shares one prologue; the adoption must sit in it,
    after ``ensure_data_home()`` (the override is validated there) and before any
    subcommand dispatch. Source-level so the ordering itself is what is pinned."""
    src = (REPO_ROOT / "src" / "kiro_crew" / "cli.py").read_text(encoding="utf-8")
    body = src.split("def main(", 1)[1]
    ensure_at = body.index("ensure_data_home()")
    adopt_at = body.index("adopt_isolated_kiro_home()")
    dispatch_at = body.index("if args.command is None:")
    assert ensure_at < adopt_at < dispatch_at


def test_gateway_startup_performs_no_bulk_transcript_migration():
    """Gateway boot must not scan or move a data-sized set of transcripts."""
    gateway = (REPO_ROOT / "src" / "kiro_crew" / "slack" / "gateway.py").read_text(encoding="utf-8")
    start_bg = gateway.split("async def _start_bg_session()", 1)[1].split(
        "asyncio.create_task(_start_bg_session())", 1
    )[0]

    assert "reclaim_adopted_transcripts" not in gateway
    assert "_migrate_adopted_transcript" not in start_bg
    assert "resolve_host_transcript" not in start_bg
    assert "resolve_resume_sid" not in start_bg


def test_the_allocation_path_migrates_through_the_off_loop_resolver():
    """The one path that hands a sid to kiro-cli must reach the transcript move
    through ``resolve_resume_sid`` -- ``get`` (cheap, guarded) then the copy on a
    worker thread -- never by calling the migration helper inline, and it must
    read the lookup's two facts apart: a WITHHELD mapping is never overwritten by
    the fresh session started in its place. Structural pin: the behaviour of the
    resolver and of the allocation guard are pinned in
    ``TestAdoptedHomeKeepsSessionMap``."""
    import inspect

    from kiro_crew import session_allocation, session_lifecycle, session_map

    src = inspect.getsource(session_allocation.SessionAllocationService._get_or_create_impl)
    assert "lookup = await resolve_resume_sid(owner._session_map, key)" in src
    assert "resume_sid, resume_withheld = lookup.sid, lookup.withheld" in src
    assert "resume_sid = owner._session_map.get(key)" not in src
    assert "_migrate_adopted_transcript" not in src
    # Every fresh-sid promotion in the allocation is gated on the withholding.
    assert "defer_sid_promotion = resume_withheld or (" in src
    assert "if sid and not resume_withheld:" in src
    assert "session.resume_withheld = resume_withheld" in src
    # ...and so is the shutdown persist, the other writer of a live sid.
    close_all = inspect.getsource(session_lifecycle.SessionLifecycleService.close_all)
    assert "if sess.resume_withheld:" in close_all

    resolver = inspect.getsource(session_map.resolve_resume_sid)
    assert "await asyncio.to_thread(resolve_host_transcript, sid)" in resolver
    # The resolver consumes only the verdict decided under the per-sid lock; it
    # never sees an outcome it could act on late.
    assert "MigrationOutcome" not in resolver
    assert "if served is None:" in resolver
    assert "return ResumeLookup(None, withheld=True)" in resolver
    decider = inspect.getsource(session_map.resolve_host_transcript)
    assert "lock = _migration_lock(sid)" in decider
    assert "with lock:" in decider
    assert "verdict = _serve_verdict(sid, _attempt_host_transcript_move(sid))" in decider
    assert "_release_migration_lock(sid, lock, terminal=terminal)" in decider
    # The verdict is ONE predicate on the disk the attempt left -- is the
    # complete host pair still unmoved -- never a property of a failure label.
    verdict = inspect.getsource(session_map._serve_verdict)
    assert "_host_pair_complete(source, sid)" in verdict
    assert "MigrationOutcome.FAILED" not in verdict
    assert "MigrationOutcome.MIGRATED" not in verdict
    assert not hasattr(session_map.MigrationOutcome, "withholds_resume")
    assert {o.name for o in session_map.MigrationOutcome} == {"MIGRATED", "NOT_PENDING", "FAILED"}
    # The guarded read only detects; the helper it defers to is not guarded.
    get_src = inspect.getsource(session_map.SessionMap.get)
    assert "host_transcript_pending(sid)" in get_src
    assert "resolve_host_transcript(" not in get_src
    assert "_migrate_adopted_transcript(" not in get_src
    assert "_copy_transcript_no_follow(" not in get_src
    assert not hasattr(session_map.resolve_host_transcript, "__wrapped__")
    # The outcome-returning twin had no production caller; the locked resolver
    # is the one off-loop entry.
    assert not hasattr(session_map, "migrate_host_transcript")


def test_doctor_runs_the_spec_home_check_beside_the_trust_root():
    src = (REPO_ROOT / "src" / "kiro_crew" / "cli_doctor.py").read_text(encoding="utf-8")
    assert re.search(
        r"_doctor_trust_root\(\)\s*\n(?:\s*#.*\n)*\s*doctor_spec_home_drift\(issues, agents_dir=_agents_dir\(\)\)",
        src,
    ), "doctor must run the spec-home drift check right after the trust-root check"


def test_adoption_reprimes_the_loaded_unc_agents_root(monkeypatch, tmp_path):
    """``hooks`` memoizes the agents dir (the UNC gate's trusted root) keyed on
    KIRO_HOME and primes it at import, which precedes the prologue. After the
    export the memo is stale and the FIRST gate check would resolve the path on
    whatever thread asked -- the event loop, on a UNC home an SMB round-trip. The
    re-prime is coupled INSIDE ``adopt_isolated_kiro_home`` so no entrypoint can
    adopt without it: after adoption the memo already answers the adopted dir."""
    from kiro_crew import hooks

    _relocate_main_homes(monkeypatch, tmp_path)
    scratch = tmp_path / "scratch-home"
    scratch.mkdir()
    monkeypatch.setenv("KIROCREW_HOME", str(scratch))
    monkeypatch.delenv("KIRO_HOME", raising=False)
    # The suite's agents-dir pin defers to a set KIRO_HOME, so the memo will
    # resolve the real (adopted) location once the export lands.
    hooks._unc_agents_root()  # primed under the PRE-adoption configuration
    primed_cache = hooks._unc_agents_root_cache
    assert primed_cache is not None
    stale_key = primed_cache[0]

    adopted = adopt_isolated_kiro_home()

    assert adopted is not None
    assert hooks._unc_agents_root_cache is not None
    assert hooks._unc_agents_root_cache[0] != stale_key, "memo not re-primed after adoption"
    assert hooks._unc_agents_root_cache[0][0] == str(adopted)
    assert hooks._unc_agents_root_cache[1] == isolated_agents_dir(scratch.resolve())


def test_reprime_seeds_the_unc_root_lexically_without_resolving(monkeypatch, tmp_path):
    """Outside the suite's agents-dir pin (production), the re-prime does not
    re-resolve the adopted home: the memo is seeded with the lexical
    ``<data home>/kiro/agents`` under the key the gate will look up, and no
    stat/lstat is issued -- the data home was resolved once by ``config_dir()``
    and a second round-trip on a UNC-backed home would sit on the boot path."""
    import os as _os

    from kiro_crew import hooks

    _relocate_main_homes(monkeypatch, tmp_path)
    scratch = tmp_path / "scratch-home"
    scratch.mkdir()
    monkeypatch.setenv("KIROCREW_HOME", str(scratch))
    monkeypatch.delenv("KIRO_HOME", raising=False)
    monkeypatch.setattr(paths, "_agents_dir_override", None)
    paths.config_dir()
    monkeypatch.setattr(paths.logger, "info", lambda *a, **k: None)
    hooks._unc_agents_root()  # primed under the PRE-adoption configuration
    expected_root = isolated_agents_dir(scratch.resolve())

    calls: list[str] = []
    for name in ("stat", "lstat", "readlink"):
        real = getattr(_os, name)

        def _spy(*a, _n=name, _real=real, **k):
            calls.append(_n)
            return _real(*a, **k)

        monkeypatch.setattr(_os, name, _spy)

    adopted = adopt_isolated_kiro_home()

    assert adopted is not None and calls == [], calls
    assert hooks._unc_agents_root_cache == (
        (str(adopted), paths.kiro_agents_dir, None),
        expected_root,
    )
    # And the seeded entry is what the gate reads back: a lookup is a key
    # comparison, still no filesystem.
    assert hooks._unc_agents_root() == expected_root and calls == []


class TestDoctorKiroHome:
    """The Data Home section names the kiro home this instance reads and, on an
    adopted (isolated) home, what the host ``~/.kiro`` still holds that it skips --
    the upgrade-transition signal for a permanently relocated install."""

    def test_shared_default_home_says_nothing(self, monkeypatch, tmp_path, capsys):
        from kiro_crew import cli_doctor

        _relocate_main_homes(monkeypatch, tmp_path)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "user"))
        monkeypatch.delenv("KIROCREW_HOME", raising=False)
        monkeypatch.delenv("KIRO_HOME", raising=False)
        cli_doctor._doctor_kiro_home(tmp_path / "user" / ".kiro" / "crew")
        assert capsys.readouterr().out == ""

    def test_a_read_error_on_the_shared_spec_is_silence_not_a_crash(
        self, monkeypatch, tmp_path, capsys
    ):
        """A read-time I/O failure (a dying disk, a misbehaving network mount)
        is not evidence either way; the read-only diagnostic must not abort."""
        import errno
        import os as os_mod

        from kiro_crew import cli_doctor

        _relocate_main_homes(monkeypatch, tmp_path)
        user = tmp_path / "user"
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: user))
        scratch = tmp_path / "relocated-home"
        scratch.mkdir()
        monkeypatch.setenv("KIROCREW_HOME", str(scratch))
        monkeypatch.setenv("KIRO_HOME", str(isolated_kiro_home(scratch)))
        agents = user / ".kiro" / "agents"
        agents.mkdir(parents=True)
        (agents / "kirocrew.json").write_text("{}", encoding="utf-8")

        real_fdopen = os_mod.fdopen

        class _ReadFails:
            def __init__(self, inner):
                self._inner = inner

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self._inner.close()
                return False

            def read(self, *args, **kwargs):
                raise OSError(errno.EIO, "Input/output error")

        monkeypatch.setattr(
            cli_doctor.os, "fdopen", lambda fd, *a, **k: _ReadFails(real_fdopen(fd, *a, **k))
        )
        # Completing without raising IS the assertion; the failed read must
        # also not be reported as a parsed spec.
        cli_doctor._doctor_kiro_home(scratch.resolve())
        assert "shared spec" not in capsys.readouterr().out

    def test_the_opt_out_is_named_as_the_shared_host_home(self, monkeypatch, tmp_path, capsys):
        """``KIRO_HOME=~/.kiro`` on a non-default data home is the one configuration
        where this instance reads the default instance's specs; the line says so
        rather than staying silent as it does on the default home."""
        from kiro_crew import cli_doctor

        _relocate_main_homes(monkeypatch, tmp_path)
        user = tmp_path / "user"
        (user / ".kiro").mkdir(parents=True)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: user))
        scratch = tmp_path / "relocated-home"
        scratch.mkdir()
        monkeypatch.setenv("KIROCREW_HOME", str(scratch))
        monkeypatch.setenv("KIRO_HOME", str(user / ".kiro"))

        cli_doctor._doctor_kiro_home(scratch.resolve())

        out = capsys.readouterr().out
        assert out.count("\n") == 1 and "kiro home:" in out
        assert "shared host home" in out and "never writes" in out
        assert "transcripts already moved into" in out

    def test_isolated_home_with_host_content_names_the_opt_out(self, monkeypatch, tmp_path, capsys):
        from kiro_crew import cli_doctor

        if not cli_doctor.pinned_fs.supports_pinned_tree_walk():
            pytest.skip("host-home child probes require descriptor-pinned listing")

        _relocate_main_homes(monkeypatch, tmp_path)
        user = tmp_path / "user"
        (user / ".kiro" / "sessions" / "cli").mkdir(parents=True)
        (user / ".kiro" / "sessions" / "cli" / "old.json").write_text("{}", encoding="utf-8")
        (user / ".kiro" / "steering").mkdir()
        (user / ".kiro" / "steering" / "team.md").write_text("x", encoding="utf-8")
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: user))
        scratch = tmp_path / "relocated-home"
        scratch.mkdir()
        monkeypatch.setenv("KIROCREW_HOME", str(scratch))
        monkeypatch.setenv("KIRO_HOME", str(isolated_kiro_home(scratch)))

        cli_doctor._doctor_kiro_home(scratch.resolve())

        out = capsys.readouterr().out
        assert "kiro home:" in out and "isolated" in out
        assert "sessions, steering" in out
        assert f"export KIRO_HOME={user / '.kiro'}" in out
        assert "transcripts already moved into" in out

    def test_isolated_home_with_empty_host_is_one_line(self, monkeypatch, tmp_path, capsys):
        from kiro_crew import cli_doctor

        _relocate_main_homes(monkeypatch, tmp_path)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "user"))
        scratch = tmp_path / "scratch-home"
        scratch.mkdir()
        monkeypatch.setenv("KIROCREW_HOME", str(scratch))
        monkeypatch.setenv("KIRO_HOME", str(isolated_kiro_home(scratch)))

        cli_doctor._doctor_kiro_home(scratch.resolve())

        out = capsys.readouterr().out
        assert out.count("\n") == 1 and "kiro home:" in out
        assert "host home" not in out

    def test_a_platform_without_pinned_walk_omits_all_host_child_probes(
        self, monkeypatch, tmp_path, capsys
    ):
        """Windows cannot prevent an ancestor junction traversal with stdlib.
        The safe diagnostic is therefore the isolated-home line alone, not a
        by-name fallback that can touch a UNC share."""
        from kiro_crew import cli_doctor

        _relocate_main_homes(monkeypatch, tmp_path)
        user = tmp_path / "user"
        (user / ".kiro" / "steering").mkdir(parents=True)
        (user / ".kiro" / "steering" / "team.md").write_text("x", encoding="utf-8")
        scratch = tmp_path / "relocated-home"
        scratch.mkdir()
        _write_spec(
            user / ".kiro" / "agents",
            "kirocrew.json",
            {
                "kirocrew-core": {
                    "command": "kirocrew",
                    "env": {"KIROCREW_HOME": str(scratch)},
                }
            },
        )
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: user))
        monkeypatch.setenv("KIROCREW_HOME", str(scratch))
        monkeypatch.setenv("KIRO_HOME", str(isolated_kiro_home(scratch)))
        monkeypatch.setattr(cli_doctor.pinned_fs, "supports_pinned_walk", lambda: False)
        monkeypatch.setattr(
            cli_doctor.pinned_fs,
            "open_dir_pinned",
            lambda *_args, **_kwargs: pytest.fail("unpinnable platform opened host home"),
        )

        cli_doctor._doctor_kiro_home(scratch.resolve())

        out = capsys.readouterr().out
        assert "kiro home:" in out and "isolated" in out
        assert "still holds" not in out and "shared spec:" not in out

    def test_an_unreadable_host_subtree_does_not_abort_doctor(self, monkeypatch, tmp_path, capsys):
        """A read-only diagnostic must survive a host subtree it cannot list.
        Simulated rather than chmod'ed so the case also runs as root and on
        Windows, where mode bits do not produce ``PermissionError``."""
        from kiro_crew import cli_doctor

        if not cli_doctor.pinned_fs.supports_pinned_tree_walk():
            pytest.skip("host-home child probes require descriptor-pinned listing")

        _relocate_main_homes(monkeypatch, tmp_path)
        user = tmp_path / "user"
        (user / ".kiro" / "skills").mkdir(parents=True)
        (user / ".kiro" / "steering").mkdir()
        (user / ".kiro" / "steering" / "team.md").write_text("x", encoding="utf-8")
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: user))
        scratch = tmp_path / "relocated-home"
        scratch.mkdir()
        monkeypatch.setenv("KIROCREW_HOME", str(scratch))
        monkeypatch.setenv("KIRO_HOME", str(isolated_kiro_home(scratch)))
        real_open = os.open

        def _open(path, flags, *args, **kwargs):
            if path == "skills" and kwargs.get("dir_fd") is not None:
                raise PermissionError(13, "Permission denied", str(path))
            return real_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(os, "open", _open)
        monkeypatch.setattr(cli_doctor.pinned_fs, "supports_pinned_walk", lambda: True)
        monkeypatch.setattr(cli_doctor.pinned_fs, "supports_pinned_tree_walk", lambda: True)

        cli_doctor._doctor_kiro_home(scratch.resolve())

        out = capsys.readouterr().out
        assert "still holds steering" in out
        assert "skills" not in out.split("still holds", 1)[1].split("\n", 1)[0]

    def test_a_shared_spec_still_pinning_this_home_is_named(self, monkeypatch, tmp_path, capsys):
        """The leftover of a relocated install that wrote the shared specs before it
        owned an isolated kiro home: ``~/.kiro/agents/kirocrew.json`` still pins
        THIS data home, the DEFAULT instance's sessions verify against it and fail,
        and nothing on this instance rewrites that file any more. Doctor names it
        with the remedy that runs on the other instance -- a ⚠️ line, not an issue."""
        from kiro_crew import cli_doctor

        if not cli_doctor.pinned_fs.supports_pinned_walk():
            pytest.skip("orphan-pin probe requires a descriptor-pinned read")

        _relocate_main_homes(monkeypatch, tmp_path)
        user = tmp_path / "user"
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: user))
        scratch = tmp_path / "relocated-home"
        scratch.mkdir()
        monkeypatch.setenv("KIROCREW_HOME", str(scratch))
        monkeypatch.setenv("KIRO_HOME", str(isolated_kiro_home(scratch)))
        _write_spec(
            user / ".kiro" / "agents",
            "kirocrew.json",
            {
                "kirocrew-core": {"command": "kirocrew", "env": {"KIROCREW_HOME": str(scratch)}},
                "kirocrew-cron": {"command": "kirocrew", "env": {"KIROCREW_HOME": "/other"}},
                # A user-added server the rebuild preserves: pinning this home there
                # is not something ``setup --agent-only`` would clear.
                "my-tool": {"command": "my-tool", "env": {"KIROCREW_HOME": str(scratch)}},
            },
        )

        cli_doctor._doctor_kiro_home(scratch.resolve())

        out = capsys.readouterr().out
        assert "shared spec:" in out and "still pins THIS data home" in out
        first = out.split("shared spec:", 1)[1].split("\n", 1)[0]
        assert "kirocrew-core" in first
        assert "kirocrew-cron" not in first and "my-tool" not in first
        assert "kirocrew setup --agent-only" in out

    def test_no_shared_spec_line_when_the_pin_is_someone_elses(self, monkeypatch, tmp_path, capsys):
        from kiro_crew import cli_doctor

        _relocate_main_homes(monkeypatch, tmp_path)
        user = tmp_path / "user"
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: user))
        scratch = tmp_path / "scratch-home"
        scratch.mkdir()
        monkeypatch.setenv("KIROCREW_HOME", str(scratch))
        monkeypatch.setenv("KIRO_HOME", str(isolated_kiro_home(scratch)))
        _write_spec(
            user / ".kiro" / "agents",
            "kirocrew.json",
            {"kirocrew-core": {"command": "kirocrew", "env": {"KIROCREW_HOME": "/other"}}},
        )

        cli_doctor._doctor_kiro_home(scratch.resolve())

        assert "shared spec:" not in capsys.readouterr().out

    @staticmethod
    def _spy_traversals(monkeypatch, link: Path) -> list[str]:
        """Record every ``os.stat`` that would traverse *link*.

        ``lstat`` is ``os.stat(follow_symlinks=False)`` and is the one probe that
        does NOT traverse the link it names; a following stat of the link, and
        any stat -- following or not -- of a path BENEATH it, resolves the link on
        the way. On Windows that resolution is the SMB connection.
        """
        traversed: list[str] = []
        real_stat = os.stat
        from kiro_crew import pinned_fs

        supports_walk = pinned_fs.supports_pinned_walk()
        supports_tree = pinned_fs.supports_pinned_tree_walk()

        def _spy_stat(path, *args, **kwargs):
            text = str(path)
            beneath = text.startswith(str(link) + os.sep)
            if beneath or (text == str(link) and kwargs.get("follow_symlinks", True)):
                traversed.append(text)
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr(os, "stat", _spy_stat)
        monkeypatch.setattr(pinned_fs, "supports_pinned_walk", lambda: supports_walk)
        monkeypatch.setattr(pinned_fs, "supports_pinned_tree_walk", lambda: supports_tree)
        return traversed

    @requires_symlinks
    def test_a_linked_host_subtree_is_neither_followed_nor_counted(
        self, monkeypatch, tmp_path, capsys
    ):
        """The host ``~/.kiro`` is user-writable and shared with other tools, so a
        linked subtree is not ours to follow.  The host root and child are opened
        once with no-follow descriptor operations; on Windows the probe is omitted
        because stdlib cannot provide that boundary."""
        from kiro_crew import cli_doctor

        if not cli_doctor.pinned_fs.supports_pinned_tree_walk():
            pytest.skip("host-home child probes require descriptor-pinned listing")

        _relocate_main_homes(monkeypatch, tmp_path)
        user = tmp_path / "user"
        (user / ".kiro" / "steering").mkdir(parents=True)
        (user / ".kiro" / "steering" / "team.md").write_text("x", encoding="utf-8")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "planted.md").write_text("x", encoding="utf-8")
        (user / ".kiro" / "skills").symlink_to(elsewhere, target_is_directory=True)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: user))
        scratch = tmp_path / "relocated-home"
        scratch.mkdir()
        monkeypatch.setenv("KIROCREW_HOME", str(scratch))
        monkeypatch.setenv("KIRO_HOME", str(isolated_kiro_home(scratch)))
        traversed = self._spy_traversals(monkeypatch, user / ".kiro" / "skills")

        cli_doctor._doctor_kiro_home(scratch.resolve())

        out = capsys.readouterr().out
        assert traversed == []
        assert "still holds steering" in out
        assert "skills" not in out.split("still holds", 1)[1].split("\n", 1)[0]

    @requires_symlinks
    @pytest.mark.parametrize("planted_at", ["kirocrew.json", "agents"])
    def test_a_linked_shared_spec_name_is_never_resolved_or_read(
        self, planted_at, monkeypatch, tmp_path, capsys
    ):
        """A planted link at either the spec leaf or ``agents`` directory is
        refused by a descriptor-relative no-follow open.  No existence probe and
        no general reader resolve the attacker-influenced spelling first."""
        from kiro_crew import cli_doctor

        _relocate_main_homes(monkeypatch, tmp_path)
        user = tmp_path / "user"
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: user))
        scratch = tmp_path / "relocated-home"
        scratch.mkdir()
        monkeypatch.setenv("KIROCREW_HOME", str(scratch))
        monkeypatch.setenv("KIRO_HOME", str(isolated_kiro_home(scratch)))
        real_spec = _write_spec(
            tmp_path / "elsewhere",
            "kirocrew.json",
            {"kirocrew-core": {"command": "kirocrew", "env": {"KIROCREW_HOME": str(scratch)}}},
        )
        if planted_at == "kirocrew.json":
            (user / ".kiro" / "agents").mkdir(parents=True)
            link = user / ".kiro" / "agents" / "kirocrew.json"
            link.symlink_to(real_spec)
        else:
            (user / ".kiro").mkdir(parents=True)
            link = user / ".kiro" / "agents"
            link.symlink_to(real_spec.parent, target_is_directory=True)
        traversed = self._spy_traversals(monkeypatch, link)
        reads: list[Path] = []
        real_read = cli_doctor._read_agent_spec

        def _spy_read(path, **kwargs):
            reads.append(path)
            return real_read(path, **kwargs)

        monkeypatch.setattr(cli_doctor, "_read_agent_spec", _spy_read)

        cli_doctor._doctor_kiro_home(scratch.resolve())

        assert traversed == [] and reads == []
        assert "shared spec:" not in capsys.readouterr().out

    @requires_symlinks
    @pytest.mark.parametrize("planted_at", ["kirocrew.json", "agents"])
    def test_a_shared_spec_swap_after_root_pinning_never_enters_the_by_name_reader(
        self, planted_at, monkeypatch, tmp_path, capsys
    ):
        """Replacing the leaf or its parent after the trusted host root is open
        cannot redirect the next step: every child is opened relative to that
        descriptor, and the general by-name reader is never entered."""
        from kiro_crew import cli_doctor

        if not cli_doctor.pinned_fs.supports_pinned_walk():
            pytest.skip("orphan-pin probe requires a descriptor-pinned read")

        _relocate_main_homes(monkeypatch, tmp_path)
        user = tmp_path / "user"
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: user))
        scratch = tmp_path / "relocated-home"
        scratch.mkdir()
        monkeypatch.setenv("KIROCREW_HOME", str(scratch))
        monkeypatch.setenv("KIRO_HOME", str(isolated_kiro_home(scratch)))
        agents = user / ".kiro" / "agents"
        shared_spec = _write_spec(
            agents,
            "kirocrew.json",
            {"kirocrew-core": {"command": "kirocrew", "env": {"KIROCREW_HOME": "/safe"}}},
        )
        planted_spec = _write_spec(
            tmp_path / "elsewhere",
            "kirocrew.json",
            {
                "kirocrew-core": {
                    "command": "kirocrew",
                    "env": {"KIROCREW_HOME": str(scratch)},
                }
            },
        )
        real_open_root = cli_doctor.pinned_fs.open_dir_pinned
        swapped = False

        def _swap_after_root_pin(path, **kwargs):
            nonlocal swapped
            fd = real_open_root(path, **kwargs)
            if not swapped and Path(path) == user / ".kiro":
                if planted_at == "kirocrew.json":
                    shared_spec.unlink()
                    shared_spec.symlink_to(planted_spec)
                else:
                    agents.rename(agents.with_name("agents-before-swap"))
                    agents.symlink_to(planted_spec.parent, target_is_directory=True)
                swapped = True
            return fd

        by_name_reads: list[Path] = []

        def _record_by_name_read(path, **_kwargs):
            by_name_reads.append(path)
            return None

        monkeypatch.setattr(cli_doctor.pinned_fs, "open_dir_pinned", _swap_after_root_pin)
        monkeypatch.setattr(cli_doctor, "_read_agent_spec", _record_by_name_read)

        cli_doctor._doctor_kiro_home(scratch.resolve())

        assert swapped is True
        assert by_name_reads == []
        assert "shared spec:" not in capsys.readouterr().out


class TestAdoptedHomeKeepsSessionMap:
    """An install that already ran on a non-default ``KIROCREW_HOME`` has its
    transcripts under the host ``~/.kiro/sessions/cli``. After the prologue gives
    it an isolated kiro home, each mapped transcript moves only when that session
    is OPENED -- ``resolve_resume_sid``, the path that hands the sid to kiro-cli;
    guarded ``get`` only detects -- and prune and failed opens continue to treat
    the host file as live."""

    @staticmethod
    def _relocated_install(monkeypatch, tmp_path):
        from kiro_crew import session_map as sm_mod

        _relocate_main_homes(monkeypatch, tmp_path)
        user = tmp_path / "user"
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: user))
        scratch = tmp_path / "relocated-home"
        scratch.mkdir()
        monkeypatch.setenv("KIROCREW_HOME", str(scratch))
        # What the prologue exports on this install after the upgrade.
        monkeypatch.setenv("KIRO_HOME", str(isolated_kiro_home(scratch)))
        monkeypatch.setattr(paths, "_sessions_dir_override", None)
        monkeypatch.setattr(sm_mod, "_KIRO_SESSIONS_DIR", None)
        sm_mod._reset_adopted_source_cache()
        host_sessions = user / ".kiro" / "sessions" / "cli"
        host_sessions.mkdir(parents=True)
        return sm_mod, scratch, host_sessions

    @pytest.mark.parametrize("spelling", ["trailing_sep", "tilde"])
    def test_equivalent_kiro_home_spelling_keeps_the_mapping_and_resumes(
        self, monkeypatch, tmp_path, spelling
    ):
        """An equivalent spelling of the adopted home is still the adopted home.

        The doctor and migration gate must canonicalize the same way; otherwise a
        trailing separator or a literal-``~`` spelling makes the cheap read
        discard a live host-side mapping before the resume path gets a chance to
        migrate it. Doctor applies ``expanduser`` then ``resolve``
        (``_explicit_kiro_home``); the gate must agree on both.
        """
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        adopted_home = isolated_kiro_home(scratch)
        if spelling == "trailing_sep":
            monkeypatch.setenv("KIRO_HOME", f"{adopted_home}{os.sep}")
        else:
            # ``Path.expanduser`` reads HOME (POSIX) / USERPROFILE (Windows),
            # not the patched ``Path.home``; point both at tmp so ``~`` expands
            # inside this test's tree.
            monkeypatch.setenv("HOME", str(tmp_path))
            monkeypatch.setenv("USERPROFILE", str(tmp_path))
            relative = adopted_home.relative_to(tmp_path)
            monkeypatch.setenv("KIRO_HOME", os.path.join("~", str(relative)))
        journal = b'{"role": "user", "content": "hi"}\n'
        (host_sessions / "sid-a.json").write_bytes(b"{}")
        (host_sessions / "sid-a.jsonl").write_bytes(journal)
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")

        assert sm.get("dashboard:one") == "sid-a"
        assert sm.prune() == 0
        assert self._open(sm_mod, sm, "dashboard:one") == "sid-a"

        adopted = isolated_kiro_home(scratch) / "sessions" / "cli"
        assert (adopted / "sid-a.jsonl").read_bytes() == journal
        assert not (host_sessions / "sid-a.json").exists()
        assert sm.mapped_sids_by_key() == {"dashboard:one": "sid-a"}

    @staticmethod
    def _open(sm_mod, sm, key: str) -> str | None:
        """What the allocation path does: the off-loop migrating resolve."""
        return asyncio.run(sm_mod.resolve_resume_sid(sm, key)).sid

    @staticmethod
    def _record_verdicts(sm_mod, monkeypatch) -> list:
        """Observe the outcome label each locked attempt hands the verdict.

        ``_serve_verdict`` is the one seam between the attempt and the verdict,
        evaluated under the per-sid lock; recording its argument is how a test
        drives the real ``resolve_host_transcript`` path and still reads how the
        attempt ended (:class:`MigrationOutcome`). The verdict itself is decided
        from the disk, not from that label. A sid that was not pending is
        answered before any attempt, so it records nothing.
        """
        seen: list = []
        real = sm_mod._serve_verdict

        def _spy(sid, outcome):
            seen.append(outcome)
            return real(sid, outcome)

        monkeypatch.setattr(sm_mod, "_serve_verdict", _spy)
        return seen

    def test_open_migrates_only_the_requested_session(self, monkeypatch, tmp_path):
        """A resume migrates one mapped session, never the rest of the map."""
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        journal = '{"role": "user", "content": "hi"}\n'
        for sid in ("sid-a", "sid-b"):
            (host_sessions / f"{sid}.json").write_text("{}", encoding="utf-8")
            (host_sessions / f"{sid}.jsonl").write_text(journal, encoding="utf-8")
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")
        sm.set("dashboard:two", "sid-b")

        assert self._open(sm_mod, sm, "dashboard:one") == "sid-a"

        adopted = isolated_kiro_home(scratch) / "sessions" / "cli"
        assert (adopted / "sid-a.json").is_file()
        assert (adopted / "sid-a.jsonl").read_text(encoding="utf-8") == journal
        assert not (host_sessions / "sid-a.json").exists()
        assert not (host_sessions / "sid-a.jsonl").exists()
        assert (host_sessions / "sid-b.json").is_file()
        assert (host_sessions / "sid-b.jsonl").is_file()
        assert not (adopted / "sid-b.json").exists()
        assert sm.mapped_sids_by_key() == {
            "dashboard:one": "sid-a",
            "dashboard:two": "sid-b",
        }

    def test_guarded_get_only_detects_and_never_copies(self, monkeypatch, tmp_path):
        """``get`` runs under the map lock, so it must stay bounded to stats: a
        host-side transcript is answered as live and left exactly where it is."""
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        (host_sessions / "sid-a.json").write_text("{}", encoding="utf-8")
        (host_sessions / "sid-a.jsonl").write_text(
            '{"role": "user", "content": "hi"}\n', encoding="utf-8"
        )
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")
        copies: list[Path] = []
        real_copy = sm_mod._copy_transcript_no_follow

        def _record(src, dst):
            copies.append(src)
            return real_copy(src, dst)

        monkeypatch.setattr(sm_mod, "_copy_transcript_no_follow", _record)

        assert sm.get("dashboard:one") == "sid-a"
        assert sm.get("dashboard:one") == "sid-a"

        assert copies == []
        assert (host_sessions / "sid-a.json").is_file()
        assert (host_sessions / "sid-a.jsonl").is_file()
        assert not (isolated_kiro_home(scratch) / "sessions").exists()
        assert sm_mod.host_transcript_pending("sid-a") is True

    def test_open_copies_off_the_loop_thread_and_outside_the_map_lock(self, monkeypatch, tmp_path):
        """The copy is unbounded I/O: it must run on a worker thread, and that
        thread must never take :data:`_MAP_LOCK`. The lock is replaced by a spy
        that records every acquiring thread; the copy seam records its own."""
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        journal = '{"role": "user", "content": "hi"}\n'
        (host_sessions / "sid-a.json").write_text("{}", encoding="utf-8")
        (host_sessions / "sid-a.jsonl").write_text(journal, encoding="utf-8")
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")

        class _SpyLock:
            def __init__(self):
                self._lock = threading.RLock()
                self.acquirers: set[int] = set()

            def __enter__(self):
                self._lock.acquire()
                self.acquirers.add(threading.get_ident())
                return self

            def __exit__(self, *exc):
                self._lock.release()
                return False

        spy = _SpyLock()
        monkeypatch.setattr(sm_mod, "_MAP_LOCK", spy)
        copy_threads: list[int] = []
        real_copy = sm_mod._copy_transcript_no_follow

        def _record(src, dst):
            copy_threads.append(threading.get_ident())
            return real_copy(src, dst)

        monkeypatch.setattr(sm_mod, "_copy_transcript_no_follow", _record)

        assert self._open(sm_mod, sm, "dashboard:one") == "sid-a"

        assert len(copy_threads) == 2  # the pair, one migration
        assert threading.get_ident() in spy.acquirers  # ``get`` ran guarded, on the loop
        assert set(copy_threads).isdisjoint({threading.get_ident()})
        assert set(copy_threads).isdisjoint(spy.acquirers)
        adopted = isolated_kiro_home(scratch) / "sessions" / "cli"
        assert (adopted / "sid-a.jsonl").read_text(encoding="utf-8") == journal
        assert not (host_sessions / "sid-a.json").exists()

    def test_two_concurrent_opens_of_one_sid_publish_the_pair_once(self, monkeypatch, tmp_path):
        """Per-sid coordination: the second open waits for the first, then finds
        nothing pending -- no double publish, no ``O_EXCL`` collision."""
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        journal = '{"role": "user", "content": "hi"}\n'
        (host_sessions / "sid-a.json").write_text("{}", encoding="utf-8")
        (host_sessions / "sid-a.jsonl").write_text(journal, encoding="utf-8")
        real_copy = sm_mod._copy_transcript_no_follow
        first_in_copy = threading.Event()
        release_first = threading.Event()
        json_copies: list[int] = []

        def _gated_copy(src, dst):
            if src.name == "sid-a.json":
                json_copies.append(threading.get_ident())
                if len(json_copies) == 1:
                    first_in_copy.set()
                    assert release_first.wait(10), "test gate never released"
            return real_copy(src, dst)

        monkeypatch.setattr(sm_mod, "_copy_transcript_no_follow", _gated_copy)
        verdicts = self._record_verdicts(sm_mod, monkeypatch)
        results: dict[str, object] = {}

        def _run(name):
            results[name] = sm_mod.resolve_host_transcript("sid-a")

        first = threading.Thread(target=_run, args=("first",))
        second = threading.Thread(target=_run, args=("second",))
        first.start()
        assert first_in_copy.wait(10)
        second.start()
        second.join(0.5)
        assert second.is_alive(), "the second open must wait on the per-sid lock"
        assert json_copies == [json_copies[0]]  # still only the first attempt
        release_first.set()
        first.join(10)
        second.join(10)
        assert not first.is_alive() and not second.is_alive()

        assert results == {"first": "sid-a", "second": "sid-a"}  # both opens serve it
        # ...the first by moving the pair, the second by finding nothing pending.
        assert verdicts == [
            sm_mod.MigrationOutcome.MIGRATED,
            sm_mod.MigrationOutcome.NOT_PENDING,
        ]
        assert len(json_copies) == 1
        adopted = isolated_kiro_home(scratch) / "sessions" / "cli"
        assert (adopted / "sid-a.json").read_text(encoding="utf-8") == "{}"
        assert (adopted / "sid-a.jsonl").read_text(encoding="utf-8") == journal
        assert not (host_sessions / "sid-a.json").exists()
        assert not (host_sessions / "sid-a.jsonl").exists()

    def test_a_durable_orphan_from_a_crash_is_recovered_on_the_next_open(
        self, monkeypatch, tmp_path
    ):
        """A durable one-file orphan at an adopted name -- here a lone state file
        with no journal, the shape an older build's crash left -- has nothing in
        memory to roll it back. A fresh process must not collide on that name
        forever: while the complete host pair exists it is a stale partial by
        construction, so it is cleared and the pair republished. Resume then
        works."""
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        journal = '{"role": "user", "content": "hi"}\n'
        (host_sessions / "sid-a.json").write_text('{"state": "host"}', encoding="utf-8")
        (host_sessions / "sid-a.jsonl").write_text(journal, encoding="utf-8")
        adopted = isolated_kiro_home(scratch) / "sessions" / "cli"
        adopted.mkdir(parents=True)
        # A complete, durable ``.json`` and no ``.jsonl``.
        (adopted / "sid-a.json").write_text('{"state": "host"}', encoding="utf-8")
        seeded = sm_mod.SessionMap()
        seeded.set("dashboard:one", "sid-a")
        seeded.flush()

        sm = sm_mod.SessionMap()  # the next process
        assert self._open(sm_mod, sm, "dashboard:one") == "sid-a"

        assert (adopted / "sid-a.json").read_text(encoding="utf-8") == '{"state": "host"}'
        assert (adopted / "sid-a.jsonl").read_text(encoding="utf-8") == journal
        assert not (host_sessions / "sid-a.json").exists()
        assert not (host_sessions / "sid-a.jsonl").exists()
        assert sm_mod.host_transcript_pending("sid-a") is False
        assert sm.get("dashboard:one") == "sid-a"

    def test_a_journal_only_orphan_from_a_crash_is_republished_on_the_next_open(
        self, monkeypatch, tmp_path
    ):
        """The journal is published FIRST and the state file last (so nothing can
        load a half-written journal), which makes a complete journal with no
        state file the shape a crash between the two publishes leaves. It equals
        the host journal -- every byte it holds is in the host pair -- so it is a
        stale partial: cleared, and the whole pair republished from the host,
        which is then retired. One rule for every residue; no promotion path."""
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        journal = '{"role": "user", "content": "hi"}\n'
        (host_sessions / "sid-a.json").write_text('{"state": "host"}', encoding="utf-8")
        (host_sessions / "sid-a.jsonl").write_text(journal, encoding="utf-8")
        adopted = isolated_kiro_home(scratch) / "sessions" / "cli"
        adopted.mkdir(parents=True)
        (adopted / "sid-a.jsonl").write_text(journal, encoding="utf-8")
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")
        copies: list[str] = []
        real_copy = sm_mod._copy_transcript_no_follow

        def _record(src, dst):
            copies.append(src.name)
            return real_copy(src, dst)

        monkeypatch.setattr(sm_mod, "_copy_transcript_no_follow", _record)
        unlinked: list[str] = []
        real_unlink = os.unlink

        def _record_unlink(path, *args, **kwargs):
            unlinked_path = _unlink_path(path, kwargs)
            unlinked.append(
                unlinked_path.name
                if unlinked_path.parent == adopted
                else f"host:{unlinked_path.name}"
            )
            return real_unlink(path, *args, **kwargs)

        monkeypatch.setattr(sm_mod.os, "unlink", _record_unlink)

        assert self._open(sm_mod, sm, "dashboard:one") == "sid-a"

        assert unlinked == ["sid-a.jsonl", "host:sid-a.json", "host:sid-a.jsonl"]
        assert copies == ["sid-a.jsonl", "sid-a.json"]  # the whole pair, journal first
        assert (adopted / "sid-a.json").read_text(encoding="utf-8") == '{"state": "host"}'
        assert (adopted / "sid-a.jsonl").read_text(encoding="utf-8") == journal
        assert not (host_sessions / "sid-a.json").exists()
        assert not (host_sessions / "sid-a.jsonl").exists()

    def test_a_residue_journal_longer_than_the_host_journal_is_withheld_and_kept(
        self, monkeypatch, tmp_path
    ):
        """The defensive cover. No session is served a sid while its complete
        host pair sits unmoved, so nothing appends to an adopted name in that
        state and a residue journal can only be a prefix of (or equal to) the
        host journal. A LONGER one holds bytes the host pair does not -- residue
        another build could have left -- and is not this code's to remove or to
        judge: nothing at either adopted name is touched, nothing is copied, the
        host pair stays, and the open is withheld until someone reconciles the
        two by hand. Withholding is the same whether the state file beside it is
        the host state, a prefix of it, or something else."""
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        journal = '{"role": "user", "content": "hi"}\n'
        live = journal + '{"role": "assistant", "content": "LIVE"}\n'
        (host_sessions / "sid-a.json").write_text('{"state": "host", "n": 1}', encoding="utf-8")
        (host_sessions / "sid-a.jsonl").write_text(journal, encoding="utf-8")
        adopted = isolated_kiro_home(scratch) / "sessions" / "cli"
        adopted.mkdir(parents=True)
        (adopted / "sid-a.jsonl").write_text(live, encoding="utf-8")
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")
        unlinked: list[str] = []
        real_unlink = os.unlink

        def _record_unlink(path, *args, **kwargs):
            unlinked.append(Path(path).name)
            return real_unlink(path, *args, **kwargs)

        monkeypatch.setattr(sm_mod.os, "unlink", _record_unlink)
        copies: list[str] = []
        real_copy = sm_mod._copy_transcript_no_follow

        def _record_copy(src, dst):
            copies.append(src.name)
            return real_copy(src, dst)

        monkeypatch.setattr(sm_mod, "_copy_transcript_no_follow", _record_copy)

        for state in (None, '{"state": "host"', '{"state": "host", "n": 1}', '{"state": "live"}'):
            if state is None:
                with contextlib.suppress(FileNotFoundError):
                    (adopted / "sid-a.json").unlink()
            else:
                (adopted / "sid-a.json").write_text(state, encoding="utf-8")
            unlinked.clear()

            assert self._open(sm_mod, sm, "dashboard:one") is None, state

            assert unlinked == [] and copies == []  # nothing removed, nothing copied
            assert (adopted / "sid-a.jsonl").read_text(encoding="utf-8") == live
            if state is not None:
                assert (adopted / "sid-a.json").read_text(encoding="utf-8") == state
            assert (host_sessions / "sid-a.json").read_text(encoding="utf-8") == (
                '{"state": "host", "n": 1}'
            )
            assert (host_sessions / "sid-a.jsonl").read_text(encoding="utf-8") == journal
            assert sm.mapped_sids_by_key()["dashboard:one"] == "sid-a"
            assert sm.get("dashboard:one") == "sid-a"  # the cheap read still names it

    def test_a_crash_during_the_state_copy_never_promotes_the_truncated_state(
        self, monkeypatch, tmp_path
    ):
        """The copier makes each adopted NAME visible at ``O_EXCL`` before its
        bytes land, so a process death during the second copy leaves a complete,
        equal journal beside a PARTIAL state file. That partial is never promoted:
        it is cleared and the state republished from the host, and the host pair
        -- the only complete state anywhere -- survives until the complete state
        is durable at the adopted name. The next open then serves valid state."""
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        journal = '{"role": "user", "content": "hi"}\n'
        state = '{"state": "host", "turns": 1}'
        (host_sessions / "sid-a.json").write_text(state, encoding="utf-8")
        (host_sessions / "sid-a.jsonl").write_text(journal, encoding="utf-8")
        adopted = isolated_kiro_home(scratch) / "sessions" / "cli"
        adopted.mkdir(parents=True)
        (adopted / "sid-a.jsonl").write_text(journal, encoding="utf-8")  # complete, equal
        (adopted / "sid-a.json").write_text(state[:9], encoding="utf-8")  # '{"state":' -- the crash
        seeded = sm_mod.SessionMap()
        seeded.set("dashboard:one", "sid-a")
        seeded.flush()
        real_unlink = os.unlink
        state_at_host_unlink: list[str | None] = []

        def _record(path, *args, **kwargs):
            if Path(path) == host_sessions / "sid-a.json":
                # The witness is about to go: what does the adopted name hold NOW?
                try:
                    state_at_host_unlink.append(
                        (adopted / "sid-a.json").read_text(encoding="utf-8")
                    )
                except FileNotFoundError:
                    state_at_host_unlink.append(None)
            return real_unlink(path, *args, **kwargs)

        monkeypatch.setattr(sm_mod.os, "unlink", _record)

        sm = sm_mod.SessionMap()  # the next process
        assert self._open(sm_mod, sm, "dashboard:one") == "sid-a"

        # The witness was asked to go exactly once, and only once a COMPLETE
        # state was at the adopted name.
        assert state_at_host_unlink == [state]
        assert (adopted / "sid-a.json").read_text(encoding="utf-8") == state
        assert (adopted / "sid-a.jsonl").read_text(encoding="utf-8") == journal
        assert not (host_sessions / "sid-a.json").exists()
        assert not (host_sessions / "sid-a.jsonl").exists()
        assert sm_mod.host_transcript_pending("sid-a") is False
        assert sm.get("dashboard:one") == "sid-a"

    def test_an_empty_state_file_from_a_crash_is_replaced_too(self, monkeypatch, tmp_path):
        """The earliest possible crash inside the state copy: the name exists and
        holds zero bytes. An empty file is a prefix of any state, so the same rule
        replaces it from the host."""
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        journal = '{"role": "user", "content": "hi"}\n'
        (host_sessions / "sid-a.json").write_text('{"state": "host"}', encoding="utf-8")
        (host_sessions / "sid-a.jsonl").write_text(journal, encoding="utf-8")
        adopted = isolated_kiro_home(scratch) / "sessions" / "cli"
        adopted.mkdir(parents=True)
        (adopted / "sid-a.jsonl").write_text(journal, encoding="utf-8")
        (adopted / "sid-a.json").write_bytes(b"")
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")

        assert self._open(sm_mod, sm, "dashboard:one") == "sid-a"
        assert (adopted / "sid-a.json").read_text(encoding="utf-8") == '{"state": "host"}'
        assert not (host_sessions / "sid-a.json").exists()

    def test_an_equal_journal_with_a_foreign_state_file_keeps_both_and_withholds(
        self, monkeypatch, tmp_path
    ):
        """One rule for both files: a state file that is neither the host state
        nor a prefix of it holds bytes the host pair does not, so it is not this
        code's to remove -- even beside a journal that equals the host journal.
        Nothing is touched, both pairs stay, and the open is withheld."""
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        journal = '{"role": "user", "content": "hi"}\n'
        (host_sessions / "sid-a.json").write_text('{"state": "host"}', encoding="utf-8")
        (host_sessions / "sid-a.jsonl").write_text(journal, encoding="utf-8")
        adopted = isolated_kiro_home(scratch) / "sessions" / "cli"
        adopted.mkdir(parents=True)
        (adopted / "sid-a.jsonl").write_text(journal, encoding="utf-8")
        (adopted / "sid-a.json").write_text('{"state": "somebody else"}', encoding="utf-8")
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")

        assert self._open(sm_mod, sm, "dashboard:one") is None

        assert (adopted / "sid-a.json").read_text(encoding="utf-8") == '{"state": "somebody else"}'
        assert (adopted / "sid-a.jsonl").read_text(encoding="utf-8") == journal
        assert (host_sessions / "sid-a.json").read_text(encoding="utf-8") == '{"state": "host"}'
        assert (host_sessions / "sid-a.jsonl").read_text(encoding="utf-8") == journal
        assert sm.mapped_sids_by_key()["dashboard:one"] == "sid-a"

    def test_a_longer_state_file_with_an_equal_journal_is_not_promoted_either(
        self, monkeypatch, tmp_path
    ):
        """A copy can never be longer than its source, so a state file that begins
        with the host state and keeps going is not a prefix copy of it: kept,
        untouched, and the open is withheld."""
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        journal = '{"role": "user", "content": "hi"}\n'
        (host_sessions / "sid-a.json").write_text('{"state": "host"}', encoding="utf-8")
        (host_sessions / "sid-a.jsonl").write_text(journal, encoding="utf-8")
        adopted = isolated_kiro_home(scratch) / "sessions" / "cli"
        adopted.mkdir(parents=True)
        (adopted / "sid-a.jsonl").write_text(journal, encoding="utf-8")
        (adopted / "sid-a.json").write_text('{"state": "host"} trailing', encoding="utf-8")
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")

        assert self._open(sm_mod, sm, "dashboard:one") is None
        assert (adopted / "sid-a.json").read_text(encoding="utf-8") == '{"state": "host"} trailing'
        assert (host_sessions / "sid-a.json").is_file()

    def test_recovery_refuses_a_divergent_residue_journal(self, monkeypatch, tmp_path):
        """A residue journal that differs from the host journal holds bytes the
        host pair does not, so it is not a stale partial and is never removed:
        no unlink, no copy, both pairs stay, the open is withheld, and a warning
        names both files for a human. Lossless over available."""
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        journal = '{"role": "user", "content": "hi"}\n'
        other = '{"role": "user", "content": "something else entirely"}\n'
        (host_sessions / "sid-a.json").write_text("{}", encoding="utf-8")
        (host_sessions / "sid-a.jsonl").write_text(journal, encoding="utf-8")
        adopted = isolated_kiro_home(scratch) / "sessions" / "cli"
        adopted.mkdir(parents=True)
        (adopted / "sid-a.json").write_text("{}", encoding="utf-8")
        (adopted / "sid-a.jsonl").write_text(other, encoding="utf-8")
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")
        unlinked: list[str] = []
        real_unlink = os.unlink

        def _record_unlink(path, *args, **kwargs):
            unlinked.append(Path(path).name)
            return real_unlink(path, *args, **kwargs)

        monkeypatch.setattr(sm_mod.os, "unlink", _record_unlink)
        copies: list[str] = []

        def _record_copy(src, dst):
            copies.append(src.name)
            raise AssertionError("nothing may be copied over a divergent residue")

        monkeypatch.setattr(sm_mod, "_copy_transcript_no_follow", _record_copy)

        assert self._open(sm_mod, sm, "dashboard:one") is None

        assert unlinked == [] and copies == []
        assert (adopted / "sid-a.jsonl").read_text(encoding="utf-8") == other
        assert (adopted / "sid-a.json").read_text(encoding="utf-8") == "{}"
        assert (host_sessions / "sid-a.jsonl").read_text(encoding="utf-8") == journal
        assert (host_sessions / "sid-a.json").is_file()
        assert sm.mapped_sids_by_key()["dashboard:one"] == "sid-a"

    def test_prefix_copy_admits_equal_and_shorter_and_refuses_longer_and_divergent(
        self, monkeypatch, tmp_path
    ):
        """The one classifier behind recovery: a residue is removable only when
        every byte it holds is also in the host file at the same offset."""
        from kiro_crew import session_map as sm_mod

        monkeypatch.setattr(sm_mod, "_COMPARE_CHUNK", 64)  # force the chunk loop to iterate
        host = tmp_path / "host.jsonl"
        body = b"".join(b'{"i": %d}\n' % i for i in range(200))
        host.write_bytes(body)
        equal, longer, shorter, divergent, empty = (
            tmp_path / n for n in ("equal", "longer", "shorter", "divergent", "empty")
        )
        equal.write_bytes(body)
        longer.write_bytes(body + b'{"live": 1}\n')
        shorter.write_bytes(body[: len(body) // 2])
        divergent.write_bytes(body[:100] + b"X" + body[101:])
        empty.write_bytes(b"")
        assert sm_mod._is_prefix_copy(host, equal) is True
        assert sm_mod._is_prefix_copy(host, shorter) is True
        assert sm_mod._is_prefix_copy(host, empty) is True
        assert sm_mod._is_prefix_copy(host, longer) is False
        assert sm_mod._is_prefix_copy(host, divergent) is False
        # A longer file that begins with every host byte is still refused: the
        # extra bytes are ones the host pair does not hold.
        assert longer.read_bytes().startswith(body)
        assert len(body) > 10 * sm_mod._COMPARE_CHUNK  # the loop really iterated
        assert not hasattr(sm_mod, "_journal_relation")
        assert not hasattr(sm_mod, "_promote_adopted_pair")

    def test_a_truncated_second_file_from_a_crash_is_republished_whole(self, monkeypatch, tmp_path):
        """A crash mid-copy of the ``.jsonl`` leaves BOTH names occupied, the
        journal truncated. Both are stale while the host pair exists: republished
        from the host, never read as a complete migration."""
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        journal = '{"role": "user", "content": "' + "x" * 2048 + '"}\n'
        (host_sessions / "sid-a.json").write_text("{}", encoding="utf-8")
        (host_sessions / "sid-a.jsonl").write_text(journal, encoding="utf-8")
        adopted = isolated_kiro_home(scratch) / "sessions" / "cli"
        adopted.mkdir(parents=True)
        (adopted / "sid-a.json").write_text("{}", encoding="utf-8")
        (adopted / "sid-a.jsonl").write_text(journal[:100], encoding="utf-8")
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")

        assert self._open(sm_mod, sm, "dashboard:one") == "sid-a"

        assert (adopted / "sid-a.jsonl").read_text(encoding="utf-8") == journal
        assert not (host_sessions / "sid-a.json").exists()
        assert not (host_sessions / "sid-a.jsonl").exists()

    def test_a_crash_before_the_host_cleanup_is_finished_on_the_next_open(
        self, monkeypatch, tmp_path
    ):
        """A crash after both publishes but before the host pair is removed
        leaves a complete adopted pair beside the host pair. The next open
        completes the move; the content is the host content either way."""
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        journal = '{"role": "user", "content": "hi"}\n'
        (host_sessions / "sid-a.json").write_text("{}", encoding="utf-8")
        (host_sessions / "sid-a.jsonl").write_text(journal, encoding="utf-8")
        adopted = isolated_kiro_home(scratch) / "sessions" / "cli"
        adopted.mkdir(parents=True)
        (adopted / "sid-a.json").write_text("{}", encoding="utf-8")
        (adopted / "sid-a.jsonl").write_text(journal, encoding="utf-8")
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")

        assert self._open(sm_mod, sm, "dashboard:one") == "sid-a"

        assert (adopted / "sid-a.json").read_text(encoding="utf-8") == "{}"
        assert (adopted / "sid-a.jsonl").read_text(encoding="utf-8") == journal
        assert not (host_sessions / "sid-a.json").exists()
        assert not (host_sessions / "sid-a.jsonl").exists()

    def test_no_flush_is_ever_asked_of_a_read_only_handle(self, monkeypatch, tmp_path):
        """Cross-OS pin. On Windows ``os.fsync`` is ``_commit`` ->
        ``FlushFileBuffers``, which needs a handle opened with WRITE access and
        reports a read-only one as ``EBADF``; POSIX flushes a read-only descriptor
        happily, so only an emulation catches the mismatch on Linux. Every adopted
        file the move makes durable is one the copier itself opened for writing,
        so an ``os.fsync`` that refuses any read-only descriptor, exactly as the
        Windows CRT does, must not turn a recoverable open (a complete unretired
        adopted pair beside the host pair) into a withheld one."""
        fcntl = pytest.importorskip("fcntl")  # Windows itself IS this test's fsync
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        journal = '{"role": "user", "content": "hi"}\n'
        (host_sessions / "sid-a.json").write_text("{}", encoding="utf-8")
        (host_sessions / "sid-a.jsonl").write_text(journal, encoding="utf-8")
        adopted = isolated_kiro_home(scratch) / "sessions" / "cli"
        adopted.mkdir(parents=True)
        (adopted / "sid-a.json").write_text("{}", encoding="utf-8")
        (adopted / "sid-a.jsonl").write_text(journal, encoding="utf-8")
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")
        real_fsync = os.fsync
        flushed_readonly: list[str] = []

        def _windows_commit(fd):
            # Directories are exempt: ``fsync_dir`` never reaches ``os.fsync`` on
            # Windows, and POSIX can only open a directory read-only.
            if stat.S_ISREG(os.fstat(fd).st_mode) and (
                fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY
            ):
                flushed_readonly.append(fd)
                raise OSError(errno.EBADF, "Bad file descriptor")
            return real_fsync(fd)

        monkeypatch.setattr(sm_mod.os, "fsync", _windows_commit)

        assert self._open(sm_mod, sm, "dashboard:one") == "sid-a"

        assert flushed_readonly == []  # no flush was ever asked of a read-only handle
        assert (adopted / "sid-a.json").read_text(encoding="utf-8") == "{}"
        assert not (host_sessions / "sid-a.json").exists()
        assert not (host_sessions / "sid-a.jsonl").exists()

    def test_the_recovery_rule_never_removes_a_destination_without_the_host_pair(
        self, monkeypatch, tmp_path
    ):
        """The guard that makes clearing a destination safe: only the complete
        host pair proves a destination stale. A lone host ``.json`` (its journal
        already gone) is not a pair -- the adopted files are left untouched and
        the mapping still resolves to them."""
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        live = '{"role": "assistant", "content": "kept"}\n'
        (host_sessions / "sid-a.json").write_text("{}", encoding="utf-8")  # no host .jsonl
        adopted = isolated_kiro_home(scratch) / "sessions" / "cli"
        adopted.mkdir(parents=True)
        (adopted / "sid-a.json").write_text('{"live": 1}', encoding="utf-8")
        (adopted / "sid-a.jsonl").write_text(live, encoding="utf-8")
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")

        assert self._open(sm_mod, sm, "dashboard:one") == "sid-a"

        assert (adopted / "sid-a.json").read_text(encoding="utf-8") == '{"live": 1}'
        assert (adopted / "sid-a.jsonl").read_text(encoding="utf-8") == live
        assert (host_sessions / "sid-a.json").is_file()

    def test_a_host_json_that_cannot_be_removed_rolls_the_publish_back(self, monkeypatch, tmp_path):
        """If the host ``.json`` -- the liveness witness -- cannot be removed after
        a complete publish, the adopted pair is rolled back rather than left
        beside a host pair as a duplicate a later open would have to reconcile.
        The complete host pair is then still unmoved, so the open is withheld:
        the mapping survives for the next attempt."""
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        journal = '{"role": "user", "content": "hi"}\n'
        (host_sessions / "sid-a.json").write_text("{}", encoding="utf-8")
        (host_sessions / "sid-a.jsonl").write_text(journal, encoding="utf-8")
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")
        real_unlink = os.unlink

        def _refuse_host_json(path, *args, **kwargs):
            if Path(path) == host_sessions / "sid-a.json":
                raise PermissionError(errno.EACCES, "simulated read-only host dir")
            return real_unlink(path, *args, **kwargs)

        monkeypatch.setattr(sm_mod.os, "unlink", _refuse_host_json)

        assert self._open(sm_mod, sm, "dashboard:one") is None

        adopted = isolated_kiro_home(scratch) / "sessions" / "cli"
        assert list(adopted.iterdir()) == []
        assert (host_sessions / "sid-a.json").is_file()
        assert (host_sessions / "sid-a.jsonl").read_text(encoding="utf-8") == journal
        assert sm.mapped_sids_by_key()["dashboard:one"] == "sid-a"

        monkeypatch.setattr(sm_mod.os, "unlink", real_unlink)
        assert self._open(sm_mod, sm, "dashboard:one") == "sid-a"
        assert (adopted / "sid-a.jsonl").read_text(encoding="utf-8") == journal
        assert not (host_sessions / "sid-a.json").exists()

    def test_a_rollback_that_leaves_residue_withholds_the_open_and_loses_nothing(
        self, monkeypatch, tmp_path
    ):
        """When the host witness cannot be removed AND the rollback cannot remove
        what it published, a complete adopted pair sits beside the complete host
        pair. The open serves NOTHING -- the host pair is unmoved, and a served
        sid would let the fresh session's sid replace the mapping. Nothing is
        lost by withholding: mapping and complete host pair are untouched, and
        once the filesystem recovers the next open reads the residue as a byte-
        equal copy, clears it and republishes the host bytes."""
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        journal = '{"role": "user", "content": "the only copy that matters"}\n'
        (host_sessions / "sid-a.json").write_text('{"state": "host"}', encoding="utf-8")
        (host_sessions / "sid-a.jsonl").write_bytes(journal.encode("utf-8"))
        adopted = isolated_kiro_home(scratch) / "sessions" / "cli"
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")
        real_unlink = os.unlink
        refused = {
            host_sessions / "sid-a.json",  # the host witness
            adopted / "sid-a.json",  # ...and both rollback unlinks
            adopted / "sid-a.jsonl",
        }

        def _refuse(path, *args, **kwargs):
            if _unlink_path(path, kwargs) in refused:
                raise PermissionError(errno.EACCES, "simulated unlink failure")
            return real_unlink(path, *args, **kwargs)

        monkeypatch.setattr(sm_mod.os, "unlink", _refuse)

        assert self._open(sm_mod, sm, "dashboard:one") is None  # nothing served

        # The unsound state is on disk -- residue beside the complete host pair --
        # but nothing was handed out over it, and nothing was lost.
        assert (adopted / "sid-a.json").is_file() and (adopted / "sid-a.jsonl").is_file()
        assert (host_sessions / "sid-a.json").read_text(encoding="utf-8") == '{"state": "host"}'
        assert (host_sessions / "sid-a.jsonl").read_text(encoding="utf-8") == journal
        assert sm.mapped_sids_by_key()["dashboard:one"] == "sid-a"
        assert sm.get("dashboard:one") == "sid-a"  # the cheap read still names the session

        monkeypatch.setattr(sm_mod.os, "unlink", real_unlink)
        assert self._open(sm_mod, sm_mod.SessionMap(), "dashboard:one") == "sid-a"

        assert (adopted / "sid-a.json").read_text(encoding="utf-8") == '{"state": "host"}'
        assert (adopted / "sid-a.jsonl").read_bytes() == journal.encode("utf-8")
        assert not (host_sessions / "sid-a.json").exists()
        assert not (host_sessions / "sid-a.jsonl").exists()
        assert sm_mod.host_transcript_pending("sid-a") is False

    def test_two_resolvers_one_sid_the_decision_is_atomic_with_the_attempt_and_lossless(
        self, monkeypatch, tmp_path
    ):
        """Two concurrent resolvers for one sid, and the two properties that hold.

        Resolver A (sid-a) fails before publishing anything and is paused between
        its attempt's outcome and its serve decision. Resolver B (same sid) is
        started meanwhile.

        1. ATOMIC: B must not be able to attempt -- let alone decide -- while A's
           decision is pending. A's verdict is decided against the disk as A's own
           attempt left it; no resolver ever consumes a verdict older than the
           last attempt on its sid.
        2. ONE PREDICATE: both attempts leave the complete host pair unmoved --
           A cleanly, B with a complete adopted copy beside it -- so BOTH are
           withheld and the mapping is never overwritten. Nothing was served, so
           nothing appends to B's residue: it is a byte-equal copy, and the next
           open clears it, republishes the host bytes and serves the sid.
        """
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        journal = '{"role": "user", "content": "hi"}\n'
        (host_sessions / "sid-a.json").write_text('{"state": "host"}', encoding="utf-8")
        (host_sessions / "sid-a.jsonl").write_text(journal, encoding="utf-8")
        adopted = isolated_kiro_home(scratch) / "sessions" / "cli"
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")

        # A fails before publishing; B copies normally.
        real_copy = sm_mod._copy_transcript_no_follow
        copies: list[str] = []

        def _first_copy_fails(src, dst):
            # The attempts are serialized on the per-sid lock, so the first copy
            # call is A's (executor threads carry no caller name).
            copies.append(src.name)
            if len(copies) == 1:
                raise OSError("simulated: A fails before publishing anything")
            return real_copy(src, dst)

        monkeypatch.setattr(sm_mod, "_copy_transcript_no_follow", _first_copy_fails)
        # B's host-witness unlink and both rollback unlinks fail.
        real_unlink = os.unlink
        refused = {host_sessions / "sid-a.json", adopted / "sid-a.json", adopted / "sid-a.jsonl"}

        def _refuse(path, *args, **kwargs):
            if _unlink_path(path, kwargs) in refused:
                raise PermissionError(errno.EACCES, "simulated unlink failure")
            return real_unlink(path, *args, **kwargs)

        monkeypatch.setattr(sm_mod.os, "unlink", _refuse)
        # Pause A between its attempt's outcome and its serve decision.
        a_deciding = threading.Event()
        release_a = threading.Event()
        real_verdict = sm_mod._serve_verdict
        verdicts: list[object] = []

        def _paused_verdict(sid, outcome):
            verdicts.append(outcome)
            if len(verdicts) == 1:  # A's: the first decision made for this sid
                a_deciding.set()
                assert release_a.wait(10), "test gate never released"
            return real_verdict(sid, outcome)

        monkeypatch.setattr(sm_mod, "_serve_verdict", _paused_verdict)
        results: dict[str, object] = {}

        def _resolve(name):
            results[name] = asyncio.run(sm_mod.resolve_resume_sid(sm, "dashboard:one"))

        a = threading.Thread(target=_resolve, args=("resolver-A",), name="resolver-A")
        b = threading.Thread(target=_resolve, args=("resolver-B",), name="resolver-B")
        a.start()
        assert a_deciding.wait(10)
        b.start()
        b.join(0.5)
        # (1) B is blocked behind A's pending decision: no attempt, no verdict.
        assert b.is_alive(), "B must wait for A's decision on the per-sid lock"
        assert "resolver-B" not in results
        assert len(copies) == 1 and len(verdicts) == 1  # only A has attempted or decided
        assert not (adopted / "sid-a.jsonl").exists()
        release_a.set()
        a.join(10)
        b.join(10)
        assert not a.is_alive() and not b.is_alive()

        # Both attempts ended with the complete host pair unmoved: both withhold
        # -- and say so, apart from "no mapping", so the callers keep the mapping.
        assert results == {
            "resolver-A": sm_mod.ResumeLookup(None, withheld=True),
            "resolver-B": sm_mod.ResumeLookup(None, withheld=True),
        }
        assert verdicts == [sm_mod.MigrationOutcome.FAILED, sm_mod.MigrationOutcome.FAILED]
        assert (adopted / "sid-a.json").is_file() and (adopted / "sid-a.jsonl").is_file()
        assert (host_sessions / "sid-a.jsonl").read_text(encoding="utf-8") == journal
        assert sm.mapped_sids_by_key()["dashboard:one"] == "sid-a"

        # (2) Nothing was served, so nothing appended to B's residue: it is a
        # byte-equal copy of the host pair. Once the filesystem recovers, the
        # next open clears it, republishes the host bytes and serves the sid.
        assert (adopted / "sid-a.jsonl").read_text(encoding="utf-8") == journal
        monkeypatch.setattr(sm_mod.os, "unlink", real_unlink)

        assert self._open(sm_mod, sm_mod.SessionMap(), "dashboard:one") == "sid-a"

        assert (adopted / "sid-a.jsonl").read_text(encoding="utf-8") == journal
        assert (adopted / "sid-a.json").read_text(encoding="utf-8") == '{"state": "host"}'
        assert not (host_sessions / "sid-a.json").exists()
        assert not (host_sessions / "sid-a.jsonl").exists()
        assert sm_mod.host_transcript_pending("sid-a") is False

    def test_a_rollback_never_removes_a_journal_that_grew_under_it(self, monkeypatch, tmp_path):
        """The tail inside one attempt: both files are published, and before the
        host witness fails to go, something writes into the adopted journal.
        Rollback must not unlink a journal that holds bytes this attempt did not
        write: it leaves the whole pair in place and withholds this open. The
        next open finds a journal that is not a prefix copy of the host journal
        and withholds too -- kept, never removed, for a hand reconciliation."""
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        journal = '{"role": "user", "content": "hi"}\n'
        (host_sessions / "sid-a.json").write_text("{}", encoding="utf-8")
        (host_sessions / "sid-a.jsonl").write_text(journal, encoding="utf-8")
        adopted = isolated_kiro_home(scratch) / "sessions" / "cli"
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")
        real_unlink = os.unlink
        appended = '{"role": "assistant", "content": "LIVE"}\n'

        def _append_then_refuse_host_json(path, *args, **kwargs):
            if Path(path) == host_sessions / "sid-a.json":
                # A session loaded the freshly published pair and appended.
                with open(adopted / "sid-a.jsonl", "a", encoding="utf-8") as live:
                    live.write(appended)
                raise PermissionError(errno.EACCES, "simulated read-only host dir")
            return real_unlink(path, *args, **kwargs)

        monkeypatch.setattr(sm_mod.os, "unlink", _append_then_refuse_host_json)

        assert self._open(sm_mod, sm, "dashboard:one") is None  # withheld, not rolled back

        assert (adopted / "sid-a.jsonl").read_text(encoding="utf-8") == journal + appended
        assert (adopted / "sid-a.json").is_file()
        assert (host_sessions / "sid-a.json").is_file()

        unlinked: list[str] = []

        def _record_unlink(path, *args, **kwargs):
            unlinked.append(Path(path).name)
            return real_unlink(path, *args, **kwargs)

        monkeypatch.setattr(sm_mod.os, "unlink", _record_unlink)
        assert self._open(sm_mod, sm, "dashboard:one") is None  # still withheld
        assert unlinked == []
        assert (adopted / "sid-a.jsonl").read_text(encoding="utf-8") == journal + appended
        assert (host_sessions / "sid-a.json").is_file()
        assert (host_sessions / "sid-a.jsonl").read_text(encoding="utf-8") == journal
        assert sm.mapped_sids_by_key()["dashboard:one"] == "sid-a"

    def test_a_stale_partial_that_cannot_be_cleared_withholds_the_open(self, monkeypatch, tmp_path):
        """Same premise, other entry: a crash orphan the recovery pre-pass cannot
        unlink is residue beside the complete host pair, so the open serves
        nothing rather than let kiro-cli write into a file the next successful
        pre-pass would delete. Once the unlink works, the next open recovers."""
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        journal = '{"role": "user", "content": "hi"}\n'
        (host_sessions / "sid-a.json").write_text("{}", encoding="utf-8")
        (host_sessions / "sid-a.jsonl").write_text(journal, encoding="utf-8")
        adopted = isolated_kiro_home(scratch) / "sessions" / "cli"
        adopted.mkdir(parents=True)
        (adopted / "sid-a.json").write_text("{}", encoding="utf-8")  # the crash orphan
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")
        real_unlink = os.unlink

        def _refuse_orphan(path, *args, **kwargs):
            if _unlink_path(path, kwargs) == adopted / "sid-a.json":
                raise PermissionError(errno.EACCES, "simulated unlink failure")
            return real_unlink(path, *args, **kwargs)

        monkeypatch.setattr(sm_mod.os, "unlink", _refuse_orphan)

        assert self._open(sm_mod, sm, "dashboard:one") is None
        assert (adopted / "sid-a.json").is_file()
        assert not (adopted / "sid-a.jsonl").exists()
        assert (host_sessions / "sid-a.json").is_file()
        assert (host_sessions / "sid-a.jsonl").read_text(encoding="utf-8") == journal
        assert sm.mapped_sids_by_key()["dashboard:one"] == "sid-a"

        monkeypatch.setattr(sm_mod.os, "unlink", real_unlink)
        assert self._open(sm_mod, sm, "dashboard:one") == "sid-a"
        assert (adopted / "sid-a.jsonl").read_text(encoding="utf-8") == journal
        assert not (host_sessions / "sid-a.json").exists()

    def test_the_verdict_is_read_from_the_host_pair_not_from_the_outcome_label(
        self, monkeypatch, tmp_path
    ):
        """Three outcome labels, one verdict rule. A failed attempt that leaves
        the complete host pair unmoved withholds whether it left residue (a
        partial the rollback could not clear) or nothing at all (a copy that
        failed before publishing); a migrated pair and a not-pending sid serve."""
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        outcome = sm_mod.MigrationOutcome
        assert {o.name for o in outcome} == {"MIGRATED", "NOT_PENDING", "FAILED"}

        journal = '{"role": "user", "content": "hi"}\n'
        (host_sessions / "sid-a.json").write_text("{}", encoding="utf-8")
        (host_sessions / "sid-a.jsonl").write_text(journal, encoding="utf-8")
        adopted = isolated_kiro_home(scratch) / "sessions" / "cli"
        verdicts = self._record_verdicts(sm_mod, monkeypatch)
        # Not pending: served before any attempt, so no verdict is ever decided.
        assert sm_mod.resolve_host_transcript("nope") == "nope" and verdicts == []

        real_copy = sm_mod._copy_transcript_no_follow

        def _fail_copy(src, dst):
            raise OSError(errno.ENOSPC, "simulated full disk")

        monkeypatch.setattr(sm_mod, "_copy_transcript_no_follow", _fail_copy)
        assert sm_mod.resolve_host_transcript("sid-a") is None  # clean failure: withheld
        assert verdicts[-1] is outcome.FAILED
        assert not adopted.exists() or list(adopted.iterdir()) == []  # nothing published
        monkeypatch.setattr(sm_mod, "_copy_transcript_no_follow", real_copy)

        adopted.mkdir(parents=True, exist_ok=True)
        (adopted / "sid-a.jsonl").write_text(journal[:5], encoding="utf-8")  # a true partial
        real_unlink = os.unlink

        def _refuse_residue(path, *args, **kwargs):
            if _unlink_path(path, kwargs) == adopted / "sid-a.jsonl":
                raise PermissionError(errno.EACCES, "simulated")
            return real_unlink(path, *args, **kwargs)

        monkeypatch.setattr(sm_mod.os, "unlink", _refuse_residue)
        assert sm_mod.resolve_host_transcript("sid-a") is None  # residue failure: withheld
        assert verdicts[-1] is outcome.FAILED
        monkeypatch.setattr(sm_mod.os, "unlink", real_unlink)

        assert sm_mod.resolve_host_transcript("sid-a") == "sid-a"
        assert verdicts[-1] is outcome.MIGRATED
        assert sm_mod.resolve_host_transcript("sid-a") == "sid-a" and len(verdicts) == 3
        assert (adopted / "sid-a.jsonl").read_text(encoding="utf-8") == journal

    def test_a_host_pair_that_cannot_be_inspected_withholds(self, monkeypatch, tmp_path):
        """The predicate fails closed. A host pair whose files cannot be stat'ed
        is not PROVEN absent or incomplete, so a failed attempt over it is
        withheld rather than served: serving would let a fresh sid replace the
        mapping of a conversation that may well be complete on disk."""
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        assert sm_mod._host_pair_complete(host_sessions, "absent") is False
        (host_sessions / "sid-a.json").write_text("{}", encoding="utf-8")
        assert sm_mod._host_pair_complete(host_sessions, "sid-a") is False  # no journal
        (host_sessions / "sid-a.jsonl").write_text("{}\n", encoding="utf-8")
        assert sm_mod._host_pair_complete(host_sessions, "sid-a") is False  # < 10 bytes
        (host_sessions / "sid-a.jsonl").write_text('{"role": "user"}\n', encoding="utf-8")
        assert sm_mod._host_pair_complete(host_sessions, "sid-a") is True
        assert sm_mod._host_pair_complete(host_sessions, "sid\x00a") is False
        real_lstat = Path.lstat

        def _refuse(self):
            if self.parent == host_sessions:
                raise PermissionError(errno.EACCES, "simulated")
            return real_lstat(self)

        monkeypatch.setattr(Path, "lstat", _refuse)
        assert sm_mod._host_pair_complete(host_sessions, "sid-a") is True
        assert sm_mod._host_pair_complete(host_sessions, "absent") is True  # cannot prove absent

    def test_a_tiny_or_incomplete_host_pair_has_nothing_to_move_and_serves(
        self, monkeypatch, tmp_path
    ):
        """Nothing to lose, so nothing to withhold: a host ``.json`` beside a
        journal under 10 bytes (or beside no journal at all) is not a resumable
        transcript. The sid is served, nothing is copied, and the host files are
        left where they are."""
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        (host_sessions / "sid-a.json").write_text("{}", encoding="utf-8")
        (host_sessions / "sid-a.jsonl").write_text("{}\n", encoding="utf-8")  # 3 bytes
        (host_sessions / "sid-b.json").write_text("{}", encoding="utf-8")  # no journal
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")
        sm.set("dashboard:two", "sid-b")
        verdicts = self._record_verdicts(sm_mod, monkeypatch)

        def _no_copy(src, dst):
            raise AssertionError("nothing to move, nothing may be copied")

        monkeypatch.setattr(sm_mod, "_copy_transcript_no_follow", _no_copy)

        assert self._open(sm_mod, sm, "dashboard:one") == "sid-a"
        assert self._open(sm_mod, sm, "dashboard:two") == "sid-b"

        assert verdicts == [sm_mod.MigrationOutcome.FAILED, sm_mod.MigrationOutcome.FAILED]
        assert (host_sessions / "sid-a.json").is_file() and (host_sessions / "sid-b.json").is_file()
        assert not (isolated_kiro_home(scratch) / "sessions").exists()

    @staticmethod
    def _acp_factory(fresh_sid: str):
        """A kiro-cli-shaped provider whose fresh sid the allocation would promote.

        ``set_resume_session_id`` records what the allocation hands kiro-cli to
        load; ``_session_id`` is what a fresh session reports afterwards.
        """
        from unittest.mock import AsyncMock, MagicMock

        from kiro_crew.acp.types import ACP_BACKEND_KIRO
        from kiro_crew.providers.acp import AcpProvider

        def factory(session_key=None, agent=None, channel_id=None, **kwargs):
            provider = object.__new__(AcpProvider)
            provider._client = MagicMock()
            provider._client._session_id = fresh_sid
            provider._client._work_dir = "/workspace"
            provider._client._pid = None
            provider._client.backend = ACP_BACKEND_KIRO
            provider._client.resumed = False
            provider._client.set_resume_session_id = MagicMock()
            provider._history_replay_needed = False
            provider._defer_replay_sid_promotion = False
            provider.start = AsyncMock()  # type: ignore[method-assign]
            provider.shutdown = AsyncMock()  # type: ignore[method-assign]
            provider.context_usage_pct = MagicMock(return_value=0.0)  # type: ignore[method-assign]
            return provider

        return factory

    def test_a_withheld_resume_never_overwrites_the_mapping_and_the_next_open_recovers_it(
        self, monkeypatch, tmp_path
    ):
        """Allocation side of the withholding. The lookup carries WITHHELD apart
        from ABSENT: a bare ``None`` reaching ``_get_or_create_impl`` would read
        as "no mapping", and the allocation would start a fresh session and
        promote its sid over the mapping -- the withheld conversation, still
        complete on disk, unreachable by key. With the distinction the fresh
        session is served (the user is not blocked) but its sid is never
        promoted -- not at registration, not at shutdown -- and the next open,
        once the residue is reconciled, resumes the mapped conversation."""
        from typing import cast
        from unittest.mock import MagicMock

        from kiro_crew.config import KiroCrewConfig
        from kiro_crew.providers.acp import AcpProvider
        from kiro_crew.session import SessionManager

        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        journal = '{"role": "user", "content": "the conversation that matters"}\n'
        (host_sessions / "sid-a.json").write_text('{"state": "host"}', encoding="utf-8")
        (host_sessions / "sid-a.jsonl").write_text(journal, encoding="utf-8")
        adopted = isolated_kiro_home(scratch) / "sessions" / "cli"
        adopted.mkdir(parents=True)
        # Residue nothing can classify as ours: an equal journal beside a state
        # file that is neither the host state nor a prefix of it -> withheld.
        (adopted / "sid-a.jsonl").write_text(journal, encoding="utf-8")
        (adopted / "sid-a.json").write_text('{"state": "somebody else"}', encoding="utf-8")
        cfg = KiroCrewConfig()

        async def _open_fresh_then_shut_down():
            mgr = SessionManager(cfg, provider_factory=self._acp_factory("fresh-sid"))
            mgr._session_map.set("dashboard:one", "sid-a")
            provider, is_new, resumed = await mgr.get_or_create("dashboard:one")
            assert isinstance(provider, AcpProvider)
            try:
                # Served fresh: nothing handed to kiro-cli to load...
                cast(MagicMock, provider.client.set_resume_session_id).assert_not_called()
                assert (is_new, resumed) == (True, False)
                # ...and the fresh sid promoted over NOTHING: the mapping still
                # names the withheld conversation.
                assert mgr._session_map.mapped_sids_by_key() == {"dashboard:one": "sid-a"}
                assert mgr.provider_switch_replay_pending("dashboard:one") is False
            finally:
                mgr.release("dashboard:one")
                await mgr.close_all()
            # Shutdown's persist is the other writer of a live sid.
            assert mgr._session_map.mapped_sids_by_key() == {"dashboard:one": "sid-a"}

        asyncio.run(_open_fresh_then_shut_down())
        # Nothing on disk was touched by serving the fresh session.
        assert (host_sessions / "sid-a.json").read_text(encoding="utf-8") == '{"state": "host"}'
        assert (host_sessions / "sid-a.jsonl").read_text(encoding="utf-8") == journal
        assert (adopted / "sid-a.json").read_text(encoding="utf-8") == '{"state": "somebody else"}'

        # Someone reconciles the residue (here: drops the foreign state file, so
        # recovery supplies it from the host). The next open resumes sid-a.
        (adopted / "sid-a.json").unlink()

        async def _reopen():
            mgr = SessionManager(cfg, provider_factory=self._acp_factory("sid-a"))
            provider, _is_new, _resumed = await mgr.get_or_create("dashboard:one")
            assert isinstance(provider, AcpProvider)
            try:
                cast(MagicMock, provider.client.set_resume_session_id).assert_called_once_with(
                    "sid-a"
                )
                assert mgr._session_map.mapped_sids_by_key() == {"dashboard:one": "sid-a"}
            finally:
                mgr.release("dashboard:one")
                await mgr.close_all()

        asyncio.run(_reopen())
        assert (adopted / "sid-a.json").read_text(encoding="utf-8") == '{"state": "host"}'
        assert (adopted / "sid-a.jsonl").read_text(encoding="utf-8") == journal
        assert not (host_sessions / "sid-a.json").exists()
        assert sm_mod.host_transcript_pending("sid-a") is False

    @pytest.mark.parametrize(
        "failure",
        ["copier_enospc", pytest.param("link_free_dir_refusal", marks=requires_symlinks)],
    )
    def test_a_clean_move_failure_withholds_and_the_fresh_sid_never_overwrites_the_mapping(
        self, monkeypatch, tmp_path, failure
    ):
        """The data-loss class, allocation side, for a move that leaves NOTHING at
        the adopted names: the copier fails (``ENOSPC``) and unlinks its own
        partial, or the destination directory is refused as a symlink before a
        byte is copied. kiro-cli reads the adopted directory, so a served sid
        would find nothing there, start fresh, and the allocation would promote
        the fresh sid over the mapping -- orphaning the complete host pair for
        good. The lookup is WITHHELD instead: a fresh session is served, its sid
        is never promoted (not at registration, not at shutdown), the host pair
        and mapping are untouched, and once the failure clears the next open
        moves the pair and resumes the mapped conversation."""
        from typing import cast
        from unittest.mock import MagicMock

        from kiro_crew.config import KiroCrewConfig
        from kiro_crew.providers.acp import AcpProvider
        from kiro_crew.session import SessionManager

        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        journal = '{"role": "user", "content": "the conversation that matters"}\n'
        (host_sessions / "sid-a.json").write_text('{"state": "host"}', encoding="utf-8")
        (host_sessions / "sid-a.jsonl").write_text(journal, encoding="utf-8")
        kiro_home = isolated_kiro_home(scratch)
        adopted = kiro_home / "sessions" / "cli"
        cfg = KiroCrewConfig()

        real_copyfileobj = pinned_fs.shutil.copyfileobj
        if failure == "copier_enospc":

            def _enospc(inp, out, *args, **kwargs):
                out.write(b"partial")
                raise OSError(errno.ENOSPC, "No space left on device")

            monkeypatch.setattr(pinned_fs.shutil, "copyfileobj", _enospc)
        else:
            outside = tmp_path / "elsewhere"
            outside.mkdir()
            kiro_home.mkdir(parents=True)
            (kiro_home / "sessions").symlink_to(outside, target_is_directory=True)

        async def _open_fresh_then_shut_down():
            mgr = SessionManager(cfg, provider_factory=self._acp_factory("fresh-sid"))
            mgr._session_map.set("dashboard:one", "sid-a")
            provider, is_new, resumed = await mgr.get_or_create("dashboard:one")
            assert isinstance(provider, AcpProvider)
            try:
                # The fresh sid promoted over NOTHING: the mapping still names
                # the withheld conversation...
                assert mgr._session_map.mapped_sids_by_key() == {"dashboard:one": "sid-a"}
                # ...and it was served fresh: nothing handed to kiro-cli to load.
                cast(MagicMock, provider.client.set_resume_session_id).assert_not_called()
                assert (is_new, resumed) == (True, False)
            finally:
                mgr.release("dashboard:one")
                await mgr.close_all()
            # Shutdown's persist is the other writer of a live sid.
            assert mgr._session_map.mapped_sids_by_key() == {"dashboard:one": "sid-a"}

        asyncio.run(_open_fresh_then_shut_down())
        # The host pair is exactly as it was, and nothing sits at an adopted name.
        assert (host_sessions / "sid-a.json").read_text(encoding="utf-8") == '{"state": "host"}'
        assert (host_sessions / "sid-a.jsonl").read_text(encoding="utf-8") == journal
        assert not (adopted / "sid-a.json").exists() and not (adopted / "sid-a.jsonl").exists()
        assert sm_mod.host_transcript_pending("sid-a") is True

        # The failure clears; the next open moves the pair and resumes sid-a.
        if failure == "copier_enospc":
            monkeypatch.setattr(pinned_fs.shutil, "copyfileobj", real_copyfileobj)
        else:
            (kiro_home / "sessions").unlink()

        async def _reopen():
            mgr = SessionManager(cfg, provider_factory=self._acp_factory("sid-a"))
            provider, _is_new, _resumed = await mgr.get_or_create("dashboard:one")
            assert isinstance(provider, AcpProvider)
            try:
                cast(MagicMock, provider.client.set_resume_session_id).assert_called_once_with(
                    "sid-a"
                )
                assert mgr._session_map.mapped_sids_by_key() == {"dashboard:one": "sid-a"}
            finally:
                mgr.release("dashboard:one")
                await mgr.close_all()

        asyncio.run(_reopen())
        assert (adopted / "sid-a.json").read_text(encoding="utf-8") == '{"state": "host"}'
        assert (adopted / "sid-a.jsonl").read_text(encoding="utf-8") == journal
        assert not (host_sessions / "sid-a.json").exists()
        assert sm_mod.host_transcript_pending("sid-a") is False

    def test_a_replay_commit_settles_without_promoting_over_a_withheld_mapping(
        self, monkeypatch, tmp_path
    ):
        """The third writer of a live sid. A replay lease armed on a session that
        was started over a withheld mapping (a later Tool Search fallback, say)
        must settle without the promotion it otherwise performs."""
        from kiro_crew.config import KiroCrewConfig
        from kiro_crew.session import SessionManager, _Session

        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        (host_sessions / "sid-a.json").write_text("{}", encoding="utf-8")
        (host_sessions / "sid-a.jsonl").write_text('{"role": "user"}\n', encoding="utf-8")
        mgr = SessionManager(KiroCrewConfig(), provider_factory=self._acp_factory("fresh-sid"))
        mgr._session_map.set("dashboard:one", "sid-a")
        mgr._sessions["dashboard:one"] = _Session(
            provider=self._acp_factory("fresh-sid")(),
            provider_switch_replay=True,
            resume_withheld=True,
        )

        assert mgr.commit_provider_switch_replay_sid("dashboard:one") is True

        assert mgr.provider_switch_replay_pending("dashboard:one") is False
        assert mgr._session_map.mapped_sids_by_key() == {"dashboard:one": "sid-a"}

    def test_lazy_move_failure_withholds_and_the_next_open_retries(self, monkeypatch, tmp_path):
        """A move that fails before publishing anything leaves the complete host
        pair unmoved: the open is withheld (nothing served, so no fresh sid can
        replace the mapping), the host pair and map are intact, and the next
        open completes the move and serves the sid."""
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        journal = '{"role": "user", "content": "hi"}\n'
        (host_sessions / "sid-a.json").write_text("{}", encoding="utf-8")
        (host_sessions / "sid-a.jsonl").write_text(journal, encoding="utf-8")
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")
        real_copy = sm_mod._copy_transcript_no_follow
        attempts: list[Path] = []

        def _fail_copy(src, dst):
            attempts.append(src)
            raise OSError("simulated full disk")

        monkeypatch.setattr(sm_mod, "_copy_transcript_no_follow", _fail_copy)
        assert self._open(sm_mod, sm, "dashboard:one") is None  # withheld
        assert attempts == [host_sessions / "sid-a.jsonl"]  # journal first, state file last
        assert sm.mapped_sids_by_key()["dashboard:one"] == "sid-a"
        assert sm.get("dashboard:one") == "sid-a"  # the cheap read still names it
        assert (host_sessions / "sid-a.json").is_file()
        assert (host_sessions / "sid-a.jsonl").is_file()

        monkeypatch.setattr(sm_mod, "_copy_transcript_no_follow", real_copy)
        assert self._open(sm_mod, sm, "dashboard:one") == "sid-a"
        adopted = isolated_kiro_home(scratch) / "sessions" / "cli"
        assert (adopted / "sid-a.json").is_file()
        assert (adopted / "sid-a.jsonl").read_text(encoding="utf-8") == journal
        assert not (host_sessions / "sid-a.json").exists()
        assert not (host_sessions / "sid-a.jsonl").exists()

    def test_prune_keeps_a_mapping_with_a_host_transcript(self, monkeypatch, tmp_path):
        """The shared two-location fence protects host-side sessions during prune."""
        sm_mod, _, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        (host_sessions / "sid-a.json").write_text("{}", encoding="utf-8")
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")

        assert sm.prune() == 0
        assert sm.mapped_sids_by_key()["dashboard:one"] == "sid-a"

    def test_get_keeps_a_mapping_with_a_host_transcript_when_move_fails(
        self, monkeypatch, tmp_path
    ):
        """The same two-location fence protects the request-driven resume path:
        the bare guarded read keeps naming the sid, and the migrating open --
        withheld while the pair stays host-side -- never drops the mapping."""
        sm_mod, _, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        (host_sessions / "sid-a.json").write_text("{}", encoding="utf-8")
        (host_sessions / "sid-a.jsonl").write_text(
            '{"role": "user", "content": "hi"}\n', encoding="utf-8"
        )
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")

        def _fail_copy(src, dst):
            raise OSError("simulated read-only destination")

        monkeypatch.setattr(sm_mod, "_copy_transcript_no_follow", _fail_copy)
        assert sm.get("dashboard:one") == "sid-a"
        assert self._open(sm_mod, sm, "dashboard:one") is None  # withheld, not served
        assert sm.get("dashboard:one") == "sid-a"
        assert sm.mapped_sids_by_key()["dashboard:one"] == "sid-a"
        assert (host_sessions / "sid-a.json").is_file()

    def test_corrupt_sid_never_aborts_prune(self, monkeypatch, tmp_path):
        """A map row whose sid embeds a NUL is dropped, not a crash of the whole pass."""
        sm_mod, _, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        (host_sessions / "sid-a.json").write_text("{}", encoding="utf-8")
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")
        sm.set("dashboard:bad", "sid\x00bad")

        assert sm.prune() == 1  # the corrupt row goes; nothing raises
        assert sm.mapped_sids_by_key()["dashboard:one"] == "sid-a"
        assert sm.get("dashboard:bad") is None

    def test_nothing_moves_under_the_opt_out(self, monkeypatch, tmp_path):
        """``KIRO_HOME=~/.kiro`` keeps reading the host directory, so there is
        nothing to reclaim and the host directory is not touched."""
        sm_mod, _, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        monkeypatch.setenv("KIRO_HOME", str(host_sessions.parent.parent))
        (host_sessions / "sid-a.json").write_text("{}", encoding="utf-8")
        (host_sessions / "sid-a.jsonl").write_text(
            '{"role": "user", "content": "hi"}\n', encoding="utf-8"
        )
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")

        assert self._open(sm_mod, sm, "dashboard:one") == "sid-a"
        assert (host_sessions / "sid-a.json").is_file()
        assert sm.prune() == 0

    def test_nothing_moves_on_the_default_home(self, monkeypatch, tmp_path):
        sm_mod, _, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        monkeypatch.delenv("KIROCREW_HOME", raising=False)
        monkeypatch.delenv("KIRO_HOME", raising=False)
        (host_sessions / "sid-a.json").write_text("{}", encoding="utf-8")
        (host_sessions / "sid-a.jsonl").write_text(
            '{"role": "user", "content": "hi"}\n', encoding="utf-8"
        )
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")

        assert self._open(sm_mod, sm, "dashboard:one") == "sid-a"
        assert (host_sessions / "sid-a.json").is_file()

    @requires_symlinks
    def test_a_kiro_home_linked_to_the_host_home_has_nothing_to_move_and_serves(
        self, monkeypatch, tmp_path
    ):
        """``<data home>/kiro`` pointing at the host ``~/.kiro`` makes the host
        directory the very directory kiro-cli reads: the transcript is already
        where it must be. Nothing to move, so the attempt reports NOT_PENDING and
        the sid is served -- the complete pair sitting there is not "unmoved",
        it is home. Nothing is copied or removed."""
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        (host_sessions / "sid-a.json").write_text("{}", encoding="utf-8")
        (host_sessions / "sid-a.jsonl").write_text(
            '{"role": "user", "content": "hi"}\n', encoding="utf-8"
        )
        kiro_home = isolated_kiro_home(scratch)
        kiro_home.parent.mkdir(parents=True, exist_ok=True)
        kiro_home.symlink_to(host_sessions.parent.parent, target_is_directory=True)
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")
        verdicts = self._record_verdicts(sm_mod, monkeypatch)

        def _no_copy(src, dst):
            raise AssertionError("nothing to move, nothing may be copied")

        monkeypatch.setattr(sm_mod, "_copy_transcript_no_follow", _no_copy)

        assert self._open(sm_mod, sm, "dashboard:one") == "sid-a"

        assert verdicts == [sm_mod.MigrationOutcome.NOT_PENDING]
        assert (host_sessions / "sid-a.json").is_file()
        assert (host_sessions / "sid-a.jsonl").is_file()
        assert sm.mapped_sids_by_key()["dashboard:one"] == "sid-a"

    def test_a_sid_that_is_not_a_bare_filename_is_skipped(self, monkeypatch, tmp_path):
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        (host_sessions / "escape.json").write_text("{}", encoding="utf-8")
        sm = sm_mod.SessionMap()
        sm._data["dashboard:x"] = {"sid": "../escape"}

        assert sm.get("dashboard:x") is None
        assert (host_sessions / "escape.json").is_file()
        assert not (isolated_kiro_home(scratch) / "sessions").exists()

    def test_failed_pair_copy_leaves_no_half_migration(self, monkeypatch, tmp_path):
        """A journal-copy failure rolls back the transcript copy before returning,
        and the open is withheld while the complete host pair stays unmoved."""
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        journal = '{"role": "user", "content": "hi"}\n'
        (host_sessions / "sid-a.json").write_text("{}", encoding="utf-8")
        (host_sessions / "sid-a.jsonl").write_text(journal, encoding="utf-8")
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")
        adopted = isolated_kiro_home(scratch) / "sessions" / "cli"
        real_copy = sm_mod._copy_transcript_no_follow

        def _die_on_journal(src, dst):
            if str(src).endswith(".jsonl"):
                raise OSError("simulated failure mid-migration")
            return real_copy(src, dst)

        monkeypatch.setattr(sm_mod, "_copy_transcript_no_follow", _die_on_journal)
        assert self._open(sm_mod, sm, "dashboard:one") is None  # withheld
        assert (host_sessions / "sid-a.json").is_file()
        assert (host_sessions / "sid-a.jsonl").is_file()
        assert not (adopted / "sid-a.json").exists()
        assert not (adopted / "sid-a.jsonl").exists()
        assert sm.prune() == 0
        assert sm.mapped_sids_by_key()["dashboard:one"] == "sid-a"

        monkeypatch.setattr(sm_mod, "_copy_transcript_no_follow", real_copy)
        assert self._open(sm_mod, sm, "dashboard:one") == "sid-a"
        assert (adopted / "sid-a.json").is_file()
        assert (adopted / "sid-a.jsonl").read_text(encoding="utf-8") == journal
        assert not (host_sessions / "sid-a.json").exists()
        assert not (host_sessions / "sid-a.jsonl").exists()

    @requires_symlinks
    def test_a_dangling_link_at_the_destination_is_not_followed(self, monkeypatch, tmp_path):
        """``dst.exists()`` says no for a dangling symlink, and a by-name write
        would land the transcript wherever the link points. The move is refused
        without following or replacing the link, nothing lands at the sibling
        name either, the source stays on the host side, and the open is
        withheld while it does."""
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        (host_sessions / "sid-a.json").write_text('{"secret": 1}', encoding="utf-8")
        (host_sessions / "sid-a.jsonl").write_text('{"role": "user"}\n', encoding="utf-8")
        adopted = isolated_kiro_home(scratch) / "sessions" / "cli"
        adopted.mkdir(parents=True)
        outside = tmp_path / "elsewhere" / "captured.json"
        outside.parent.mkdir()
        (adopted / "sid-a.json").symlink_to(outside)
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")

        assert self._open(sm_mod, sm, "dashboard:one") is None
        assert sm.mapped_sids_by_key()["dashboard:one"] == "sid-a"
        assert not outside.exists()
        assert (adopted / "sid-a.json").is_symlink()
        assert not (adopted / "sid-a.json").exists()
        assert not (adopted / "sid-a.jsonl").exists()
        assert (host_sessions / "sid-a.json").read_text(encoding="utf-8") == '{"secret": 1}'
        assert (host_sessions / "sid-a.jsonl").is_file()

    @requires_symlinks
    def test_a_symlinked_component_of_the_destination_refuses_the_migration(
        self, monkeypatch, tmp_path
    ):
        """A link planted at ``<kiro home>/sessions`` would carry every moved
        transcript out of the instance's kiro home. Nothing moves, nothing is
        created behind the link, the host directory is left intact, and the open
        is withheld -- the mapping survives for a later, link-free open."""
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        (host_sessions / "sid-a.json").write_text("{}", encoding="utf-8")
        (host_sessions / "sid-a.jsonl").write_text('{"role": "user"}\n', encoding="utf-8")
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        kiro_home = isolated_kiro_home(scratch)
        kiro_home.mkdir(parents=True)
        (kiro_home / "sessions").symlink_to(outside, target_is_directory=True)
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")

        assert self._open(sm_mod, sm, "dashboard:one") is None
        assert sm.mapped_sids_by_key()["dashboard:one"] == "sid-a"
        assert list(outside.iterdir()) == []
        assert (host_sessions / "sid-a.json").is_file() and (
            host_sessions / "sid-a.jsonl"
        ).is_file()

    @requires_symlinks
    def test_a_symlinked_kiro_home_root_refuses_the_migration(self, monkeypatch, tmp_path):
        """A link planted AT ``<data home>/kiro`` itself -- the root of the
        walk, which a component-only check never lstats -- would carry the
        whole adopted home, and every moved transcript with it, wherever the
        link points. The root is judged like any other component: nothing
        moves, nothing is created behind the link, the host directory is left
        intact, and the open is withheld."""
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        (host_sessions / "sid-a.json").write_text("{}", encoding="utf-8")
        (host_sessions / "sid-a.jsonl").write_text('{"role": "user"}\n', encoding="utf-8")
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        kiro_home = isolated_kiro_home(scratch)
        kiro_home.parent.mkdir(parents=True, exist_ok=True)
        kiro_home.symlink_to(outside, target_is_directory=True)
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")

        assert self._open(sm_mod, sm, "dashboard:one") is None
        assert sm.mapped_sids_by_key()["dashboard:one"] == "sid-a"
        assert list(outside.iterdir()) == []
        assert (host_sessions / "sid-a.json").is_file() and (
            host_sessions / "sid-a.jsonl"
        ).is_file()

    @requires_pinned_walk
    def test_transcript_publish_uses_one_pinned_destination_descriptor(self, monkeypatch, tmp_path):
        """Both names publish relative to the one descriptor opened for the
        destination directory; the parent path is never resolved again between
        validation and either exclusive create."""
        from kiro_crew import pinned_fs
        from kiro_crew import session_map as sm_mod

        _sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        journal = b'{"role": "user", "content": "hi"}\n'
        (host_sessions / "sid-a.json").write_bytes(b"{}")
        (host_sessions / "sid-a.jsonl").write_bytes(journal)
        target = isolated_kiro_home(scratch) / "sessions" / "cli"
        target.mkdir(parents=True)
        opened_target_fds: list[int] = []
        copy_calls: list[tuple[int | None, str | None]] = []
        real_open_dir = pinned_fs.create_and_open_dir_pinned
        real_copy = pinned_fs.copy_file_pinned

        def _record_open(path, **kwargs):
            fd = real_open_dir(path, **kwargs)
            if Path(path) == target:
                opened_target_fds.append(fd)
            return fd

        def _record_copy(*args, **kwargs):
            copy_calls.append((kwargs.get("dst_dir_fd"), kwargs.get("dst_name")))
            return real_copy(*args, **kwargs)

        monkeypatch.setattr(pinned_fs, "create_and_open_dir_pinned", _record_open)
        monkeypatch.setattr(pinned_fs, "copy_file_pinned", _record_copy)
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")

        assert self._open(sm_mod, sm, "dashboard:one") == "sid-a"

        assert len(opened_target_fds) == 1
        assert copy_calls == [
            (opened_target_fds[0], "sid-a.jsonl"),
            (opened_target_fds[0], "sid-a.json"),
        ]
        assert not hasattr(sm_mod, "_link_free_dir")
        assert (target / "sid-a.jsonl").read_bytes() == journal

    def test_a_home_without_hard_link_support_still_migrates(self, monkeypatch, tmp_path):
        """The publish must not depend on ``os.link``: on a filesystem without
        hard links (exFAT and kin) that call raises ``OSError``, the caller's
        skip-and-log branch swallows it, and the prune that follows in the same
        startup reads the un-migrated transcript as gone and drops the mapping.
        The descriptor publish (``O_CREAT | O_EXCL``) uses no link at all, so
        the move succeeds even where ``os.link`` always fails."""
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        journal = '{"role": "user", "content": "hi"}\n'
        (host_sessions / "sid-a.json").write_text("{}", encoding="utf-8")
        (host_sessions / "sid-a.jsonl").write_text(journal, encoding="utf-8")

        def _no_hard_links(*args, **kwargs):
            raise OSError("hard links not supported here (simulated exFAT)")

        monkeypatch.setattr(sm_mod.os, "link", _no_hard_links)
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")

        assert self._open(sm_mod, sm, "dashboard:one") == "sid-a"
        adopted = isolated_kiro_home(scratch) / "sessions" / "cli"
        assert (adopted / "sid-a.json").is_file()
        assert (adopted / "sid-a.jsonl").read_text(encoding="utf-8") == journal
        assert not (host_sessions / "sid-a.json").exists()
        assert not (host_sessions / "sid-a.jsonl").exists()
        assert sm.prune() == 0 and sm.get("dashboard:one") == "sid-a"

    @requires_symlinks
    def test_a_symlink_under_a_mapped_sid_on_the_host_side_is_not_read_through(
        self, monkeypatch, tmp_path
    ):
        """A link planted where this instance's transcript should be would, read
        by name, copy whatever it points at into the isolated sessions directory.
        It is refused before a byte is read: nothing lands in the adopted home,
        the link and its target are untouched, and the plain sibling still moves."""
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        secret = tmp_path / "elsewhere" / "credential"
        secret.parent.mkdir()
        secret.write_text("hunter2", encoding="utf-8")
        (host_sessions / "sid-a.json").symlink_to(secret)
        (host_sessions / "sid-a.jsonl").write_text('{"role": "user"}\n', encoding="utf-8")
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")

        assert self._open(sm_mod, sm, "dashboard:one") == "sid-a"
        adopted = isolated_kiro_home(scratch) / "sessions" / "cli"
        assert not adopted.exists()
        assert (host_sessions / "sid-a.json").is_symlink()
        assert (host_sessions / "sid-a.jsonl").is_file()
        assert secret.read_text(encoding="utf-8") == "hunter2"

    def test_a_source_that_is_not_a_plain_file_is_refused_by_the_opener(self, tmp_path):
        """The opener itself is the guard: a directory or a name that vanished
        answers ``_SourceRefused`` (an ``OSError``), so the caller's skip-and-log
        branch applies, and no descriptor is leaked on the refusal path."""
        from kiro_crew import session_map as sm_mod

        (tmp_path / "a-dir.json").mkdir()
        with pytest.raises(sm_mod._SourceRefused):
            sm_mod._open_regular_file_no_follow(tmp_path / "a-dir.json")
        with pytest.raises(sm_mod._SourceRefused):
            sm_mod._open_regular_file_no_follow(tmp_path / "gone.json")
        plain = tmp_path / "plain.json"
        plain.write_bytes(b"{}")
        fd = sm_mod._open_regular_file_no_follow(plain)
        try:
            assert os.read(fd, 8) == b"{}"
        finally:
            os.close(fd)

    def test_a_failure_while_copying_keeps_the_source_and_leaves_no_partial_file(
        self, monkeypatch, tmp_path
    ):
        """The destination name is created exclusively by this move, so a copy
        that dies leaves the source in place and nothing under the final name --
        the partial file is removed on the failure path. The complete host pair
        is unmoved, so the open is withheld, and the next open simply tries
        again."""
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        body = '{"role": "user", "content": "' + "x" * 4096 + '"}\n'
        (host_sessions / "sid-a.json").write_text("{}", encoding="utf-8")
        (host_sessions / "sid-a.jsonl").write_text(body, encoding="utf-8")
        adopted = isolated_kiro_home(scratch) / "sessions" / "cli"

        def _die(inp, out, *args, **kwargs):
            out.write(b"partial")
            raise OSError("simulated failure mid-copy")

        real_copy = pinned_fs.shutil.copyfileobj
        monkeypatch.setattr(pinned_fs.shutil, "copyfileobj", _die)
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")
        assert self._open(sm_mod, sm, "dashboard:one") is None  # withheld
        assert (host_sessions / "sid-a.json").is_file() and (
            host_sessions / "sid-a.jsonl"
        ).is_file()
        assert list(adopted.iterdir()) == []
        assert sm.mapped_sids_by_key()["dashboard:one"] == "sid-a"

        monkeypatch.setattr(pinned_fs.shutil, "copyfileobj", real_copy)
        assert self._open(sm_mod, sm, "dashboard:one") == "sid-a"
        assert (adopted / "sid-a.jsonl").read_text(encoding="utf-8") == body
        assert (adopted / "sid-a.json").read_text(encoding="utf-8") == "{}"
        assert not (host_sessions / "sid-a.json").exists()
        assert not (host_sessions / "sid-a.jsonl").exists()
        assert list(adopted.glob(".*.part")) == []
        assert sm.prune() == 0 and sm.get("dashboard:one") == "sid-a"

    def test_destination_directory_fsync_failure_keeps_source_and_removes_destination(
        self, monkeypatch, tmp_path
    ):
        from kiro_crew import session_map as sm_mod

        src = tmp_path / "source.json"
        src.write_text('{"session": 1}', encoding="utf-8")
        dst = tmp_path / "adopted" / "source.json"
        dst.parent.mkdir()

        def _fail_directory_fsync(path, dir_fd=None):
            raise OSError("simulated destination-directory fsync failure")

        monkeypatch.setattr(
            sm_mod, "_fsync_transcript_destination_dir", _fail_directory_fsync, raising=False
        )

        with pytest.raises(OSError, match="destination-directory fsync"):
            sm_mod._copy_transcript_no_follow(src, dst)

        assert src.read_text(encoding="utf-8") == '{"session": 1}'
        assert not dst.exists()

    def test_unsupported_directory_fsync_does_not_abort_the_migration(self, monkeypatch, tmp_path):
        """A filesystem that cannot fsync a directory (network mounts answer
        ``EINVAL``) still migrates: the move completes on the file fsync alone
        rather than failing every transcript and leaving the mapping host-side
        forever. Driven through the live open path."""
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        journal = '{"role": "user", "content": "hi"}\n'
        (host_sessions / "sid-a.json").write_text('{"session": 1}', encoding="utf-8")
        (host_sessions / "sid-a.jsonl").write_text(journal, encoding="utf-8")
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")

        real_fsync = os.fsync

        def _reject_directory_fsync(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError(errno.EINVAL, "Invalid argument")
            return real_fsync(fd)

        monkeypatch.setattr(os, "fsync", _reject_directory_fsync)

        assert self._open(sm_mod, sm, "dashboard:one") == "sid-a"

        adopted = isolated_kiro_home(scratch) / "sessions" / "cli"
        assert (adopted / "sid-a.json").read_text(encoding="utf-8") == '{"session": 1}'
        assert (adopted / "sid-a.jsonl").read_text(encoding="utf-8") == journal
        assert not (host_sessions / "sid-a.json").exists()
        assert not (host_sessions / "sid-a.jsonl").exists()
        assert sm.get("dashboard:one") == "sid-a"

    def test_pending_probe_resolves_adopted_homes_once_per_configuration(
        self, monkeypatch, tmp_path
    ):
        """Repeated cheap probes reuse the canonical adopted-home comparison."""
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        configured = Path(os.environ["KIRO_HOME"]).expanduser()
        adopted = isolated_kiro_home(scratch)
        other = tmp_path / "other-kiro-home"
        other.mkdir()
        real_resolve = Path.resolve
        resolved: list[Path] = []

        def _counting_resolve(path, *args, **kwargs):
            if path in {configured, adopted, other}:
                resolved.append(path)
            return real_resolve(path, *args, **kwargs)

        monkeypatch.setattr(Path, "resolve", _counting_resolve)

        for _ in range(20):
            assert sm_mod._adopted_transcript_source() == host_sessions
        assert resolved == [configured, adopted]

        monkeypatch.setenv("KIRO_HOME", str(other))
        assert sm_mod._adopted_transcript_source() is None
        assert resolved == [configured, adopted, other, adopted]

    def test_terminal_migration_evicts_its_per_sid_lock(self, monkeypatch, tmp_path):
        """A completed one-time move retains no process-lifetime sid entry."""
        sm_mod, _, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        sid = "sid-terminal-lock"
        (host_sessions / f"{sid}.json").write_text("{}", encoding="utf-8")
        (host_sessions / f"{sid}.jsonl").write_text(
            '{"role": "user", "content": "hi"}\n', encoding="utf-8"
        )

        assert sm_mod.resolve_host_transcript(sid) == sid

        with sm_mod._MIGRATION_LOCKS_GUARD:
            assert sid not in sm_mod._MIGRATION_LOCKS
            assert sid not in sm_mod._MIGRATION_LOCK_REFS

    def test_a_host_journal_append_after_copy_prevents_retirement(self, monkeypatch, tmp_path):
        """A foreign append in the copy window keeps the authoritative host pair."""
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        journal = '{"role": "user", "content": "hi"}\n'
        appended = '{"role": "assistant", "content": "late"}\n'
        host_json = host_sessions / "sid-a.json"
        host_jsonl = host_sessions / "sid-a.jsonl"
        host_json.write_text("{}", encoding="utf-8")
        host_jsonl.write_text(journal, encoding="utf-8")
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")
        real_copy = sm_mod._copy_transcript_no_follow
        real_unlink = os.unlink
        host_unlinks: list[Path] = []

        def _append_after_copy(src, destination):
            identity = real_copy(src, destination)
            if src == host_json:
                with host_jsonl.open("a", encoding="utf-8") as stream:
                    stream.write(appended)
            return identity

        def _record_unlink(path, *args, **kwargs):
            unlinked = _unlink_path(path, kwargs)
            if unlinked.parent == host_sessions:
                host_unlinks.append(unlinked)
            return real_unlink(path, *args, **kwargs)

        monkeypatch.setattr(sm_mod, "_copy_transcript_no_follow", _append_after_copy)
        monkeypatch.setattr(sm_mod.os, "unlink", _record_unlink)

        assert self._open(sm_mod, sm, "dashboard:one") is None

        adopted = isolated_kiro_home(scratch) / "sessions" / "cli"
        assert host_unlinks == []
        assert host_json.read_text(encoding="utf-8") == "{}"
        assert host_jsonl.read_text(encoding="utf-8") == journal + appended
        assert list(adopted.iterdir()) == []
        assert sm.mapped_sids_by_key()["dashboard:one"] == "sid-a"

    @requires_pinned_walk
    def test_recovery_unlinks_residue_through_the_pinned_descriptor(self, monkeypatch, tmp_path):
        """Recovery removes the inspected inode through its held parent descriptor."""
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        journal = b'{"role": "user", "content": "hi"}\n'
        (host_sessions / "sid-a.json").write_bytes(b"{}")
        (host_sessions / "sid-a.jsonl").write_bytes(journal)
        adopted = isolated_kiro_home(scratch) / "sessions" / "cli"
        adopted.mkdir(parents=True)
        residue = adopted / "sid-a.jsonl"
        residue.write_bytes(journal)
        residue_stat = residue.lstat()
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")
        verified: list[tuple[str, tuple[int, int]]] = []
        direct_adopted_unlinks: list[Path] = []
        real_verified = pinned_fs.unlink_verified
        real_unlink = os.unlink

        def _record_verified(dir_fd, name, expected):
            assert Path(pinned_fs.fd_real_path(dir_fd) or "") == adopted
            verified.append((name, expected))
            return real_verified(dir_fd, name, expected)

        def _record_unlink(path, *args, **kwargs):
            unlinked = _unlink_path(path, kwargs)
            if unlinked.parent == adopted and kwargs.get("dir_fd") is None:
                direct_adopted_unlinks.append(unlinked)
            return real_unlink(path, *args, **kwargs)

        monkeypatch.setattr(pinned_fs, "unlink_verified", _record_verified)
        monkeypatch.setattr(sm_mod.os, "unlink", _record_unlink)

        assert self._open(sm_mod, sm, "dashboard:one") == "sid-a"

        assert verified == [("sid-a.jsonl", (residue_stat.st_dev, residue_stat.st_ino))]
        assert direct_adopted_unlinks == []
        assert (adopted / "sid-a.jsonl").read_bytes() == journal

    def test_a_hardlink_snapshotted_host_pair_still_migrates(self, monkeypatch, tmp_path):
        """A host home kept by a hard-link snapshot tool migrates like any other.

        ``rsnapshot``, ``rsync --link-dest`` and ``cp -al`` give every file in the
        host ``~/.kiro`` a second name inside the snapshot tree, so both transcript
        files carry ``st_nlink == 2``. The source is read through a descriptor the
        migration has itself validated as a regular file, and the bytes are the
        same bytes whichever name reaches them: the move succeeds, the adopted pair
        matches the host bytes, and the snapshot's names keep theirs.
        """
        sm_mod, scratch, host_sessions = self._relocated_install(monkeypatch, tmp_path)
        journal = b'{"role": "user", "content": "hi"}\n'
        state = b'{"conversation_id": "sid-a"}'
        host_json = host_sessions / "sid-a.json"
        host_jsonl = host_sessions / "sid-a.jsonl"
        host_json.write_bytes(state)
        host_jsonl.write_bytes(journal)
        snapshot = tmp_path / "user" / ".snapshots" / "hourly.0" / ".kiro" / "sessions" / "cli"
        snapshot.mkdir(parents=True)
        try:
            os.link(host_json, snapshot / "sid-a.json")
            os.link(host_jsonl, snapshot / "sid-a.jsonl")
        except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
            pytest.skip("hard links are unavailable on this host")
        assert host_json.stat().st_nlink == 2 and host_jsonl.stat().st_nlink == 2
        sm = sm_mod.SessionMap()
        sm.set("dashboard:one", "sid-a")

        assert self._open(sm_mod, sm, "dashboard:one") == "sid-a"

        adopted = isolated_kiro_home(scratch) / "sessions" / "cli"
        assert (adopted / "sid-a.json").read_bytes() == state
        assert (adopted / "sid-a.jsonl").read_bytes() == journal
        assert not host_json.exists() and not host_jsonl.exists()
        assert (snapshot / "sid-a.json").read_bytes() == state
        assert (snapshot / "sid-a.jsonl").read_bytes() == journal
        assert sm.mapped_sids_by_key() == {"dashboard:one": "sid-a"}


def test_only_the_ownership_guard_consults_the_ambient_agents_dir():
    """The generalizable rule behind this fix: a writer of the SHARED agents dir
    must first ask whether this instance owns it (``foreign_data_home()``), and
    that question is asked in exactly one place -- ``agent._unexempt_shared_target``,
    behind both ``_decline_shared_agent_home`` (the managed spec) and
    ``foreign_home_targets_shared_agents_dir`` (app specs, via ``apps.bridges``).
    A new module reaching for ``ambient_agents_dir()`` directly would be a writer
    (or reader) that bypasses that ownership decision, which is how the shared
    specs got poisoned in the first place. Every other consumer goes through
    ``kiro_agents_dir()``, which the guard sits behind."""
    src_root = REPO_ROOT / "src" / "kiro_crew"
    callers: dict[str, int] = {}
    for path in src_root.rglob("*.py"):
        if path.name == "paths.py" and path.parent.name == "config":
            continue  # the resolver's own definition
        text = path.read_text(encoding="utf-8")
        # Prose mentions (``ambient_agents_dir()`` in a docstring, or a comment
        # line) are not calls.
        n = sum(
            1
            for line in text.splitlines()
            if "ambient_agents_dir()" in line
            and "``ambient_agents_dir()``" not in line
            and not line.lstrip().startswith("#")
        )
        if n:
            callers[path.relative_to(src_root).as_posix()] = n
    assert callers == {"agent.py": 1}, callers
