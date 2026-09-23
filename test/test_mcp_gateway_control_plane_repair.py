"""Toolbox-shim report: a reserved ``kirocrew-*`` entry launches the managed invocation.

An agent spec authored outside Kiro Crew (a package manager, a hand-edited file)
cannot spell the managed command: it embeds the data home and the installed
version, so the one portable spelling is the bare launcher ``kirocrew``, which
resolves through the shared Toolbox dispatcher -- a different file from the
versioned binary. ``gatewayd._spawns_own_control_plane`` compares the spawned
binary by realpath against the managed entry, so such a spec's control plane
mounted, listed its tools, and refused every call ``identity_unattested``.

The rewriter now re-derives the launch of every stubbed reserved name from the
managed source before it resolves, hashes or bakes anything, so the stub's
target, the daemon's ``KIROCREW_MCP_TARGET_*`` and the gate's expectation are
one value. These tests drive that path with the REAL managed entry and the REAL
gate, then pin the boundaries: unreserved names are untouched, an unresolvable
managed name is left as declared, and the fingerprint sees the managed launch.
"""

from __future__ import annotations

import json
import logging
import shlex
from pathlib import Path

import pytest

from kiro_crew import agent as agent_mod
from kiro_crew.mcp_cleanup import KIROCREW_BIN_MCP_SERVERS
from kiro_crew.mcp_gateway import rewriter
from kiro_crew.mcp_gateway.gatewayd import _spawns_own_control_plane
from kiro_crew.mcp_gateway.hashing import decode_target_args, expand_stub_flags
from kiro_crew.mcp_gateway.rewriter import _WRAPPER_MARKER, _rewrite_single_spec

TOOLBOX_SHIM = {"command": "kirocrew", "args": ["mcp-core"]}
REAPED_PIN = {
    "command": "/opt/toolbox/tools/kirocrew/0.6.0.9/bin/kirocrew",
    "args": ["mcp-core"],
}


def _rewrite(spec: dict, tmp_path: Path, *, stub_servers: frozenset[str]) -> tuple[dict, int]:
    return _rewrite_single_spec(
        spec,
        stubs_dir=tmp_path / "stubs",
        socket_path=tmp_path / "gw.sock",
        work_dir=tmp_path / "wd",
        sandbox_mode="auto",
        approval_mode="interactive",
        stub_servers=stub_servers,
        pooling_enabled=True,
        forward_env=False,
    )


def _stub_launch(entry: dict) -> tuple[str, list[str]]:
    """The (command, args) a stub entry tells gatewayd to spawn."""
    args = expand_stub_flags(entry["args"])
    command = args[args.index("--target-command") + 1]
    encoded = next(a for a in args if a.startswith("--target-args-b64="))
    return command, decode_target_args(encoded.partition("=")[2])


@pytest.fixture
def managed_core() -> dict:
    entry = agent_mod.managed_mcp_spec_entry("kirocrew-core")
    assert entry is not None and Path(entry["command"]).is_absolute()
    return entry


@pytest.mark.parametrize("declared", [TOOLBOX_SHIM, REAPED_PIN], ids=["toolbox-shim", "reaped-pin"])
def test_a_hand_authored_control_plane_launches_the_managed_binary_and_passes_the_gate(
    tmp_path: Path, caplog, declared: dict, managed_core: dict
) -> None:
    """The ticket's reproduction, end to end: the spec says ``kirocrew`` (or a
    version-pinned path an upgrade has reaped); the stub must carry the managed
    command and args, and the daemon gate -- the real one, on that exact launch --
    must hand over the token."""
    spec = {"name": "team-config", "mcpServers": {"kirocrew-core": dict(declared)}}
    with caplog.at_level(logging.WARNING, logger="kiro_crew.mcp_gateway.rewriter"):
        new_spec, wrapped = _rewrite(spec, tmp_path, stub_servers=frozenset({"kirocrew-core"}))

    assert wrapped == 1
    entry = new_spec["mcpServers"]["kirocrew-core"]
    assert entry.get(_WRAPPER_MARKER) is True
    command, args = _stub_launch(entry)
    assert (command, args) == (managed_core["command"], managed_core["args"])
    # Negative control: what the spec declared is not what will be spawned.
    assert command != declared["command"]
    # The daemon's verdict on the launch the stub now names -- the same
    # function, on the same tuple, with a clean child env.
    assert _spawns_own_control_plane("kirocrew-core", command, args, env={}, work_dir=tmp_path)
    # And the declared launch, had it reached the daemon, is still refused: the
    # repair is what closes the gap, not a relaxed gate.
    assert not _spawns_own_control_plane(
        "kirocrew-core", declared["command"], declared["args"], env={}, work_dir=tmp_path
    )
    # The operator is told, once, what was declared and what will run.
    said = [r.getMessage() for r in caplog.records if "reserved server" in r.getMessage()]
    assert len(said) == 1
    assert shlex.join([declared["command"], *declared["args"]]) in said[0]
    assert managed_core["command"] in said[0]


def test_the_target_env_the_daemon_reads_names_the_managed_binary(
    tmp_path: Path, managed_core: dict
) -> None:
    """``_collect_target_env`` is what gatewayd's ``env_target_resolver`` spawns
    from; it must agree with the stub, or the PoolKey lies about the binary."""
    spec = {"name": "team-config", "mcpServers": {"kirocrew-core": dict(TOOLBOX_SHIM)}}
    new_spec, _ = _rewrite(spec, tmp_path, stub_servers=frozenset({"kirocrew-core"}))
    target_env: dict[str, str] = {}
    rewriter._collect_target_env(new_spec["mcpServers"], target_env)
    spawned = shlex.split(target_env["KIROCREW_MCP_TARGET_KIROCREW_CORE"])
    assert spawned == [managed_core["command"], *managed_core["args"]]


def test_a_spec_already_naming_the_managed_launch_is_not_rewritten_or_warned(
    tmp_path: Path, caplog, managed_core: dict
) -> None:
    """Kiro Crew's own ``kirocrew.json`` carries the managed launch; it must
    round-trip silently -- the warning is for hand-authored specs only."""
    spec = {
        "name": "kirocrew",
        "mcpServers": {
            "kirocrew-core": {"command": managed_core["command"], "args": managed_core["args"]}
        },
    }
    with caplog.at_level(logging.WARNING, logger="kiro_crew.mcp_gateway.rewriter"):
        new_spec, wrapped = _rewrite(spec, tmp_path, stub_servers=frozenset({"kirocrew-core"}))
    assert wrapped == 1
    assert _stub_launch(new_spec["mcpServers"]["kirocrew-core"]) == (
        managed_core["command"],
        managed_core["args"],
    )
    assert not [r for r in caplog.records if "reserved server" in r.getMessage()]


def test_an_unreserved_name_keeps_the_command_it_declared(tmp_path: Path) -> None:
    """The repair is keyed on the reserved set, never on the command: a
    third-party server that happens to launch ``kirocrew`` is the operator's
    business and is resolved as declared."""
    spec = {"name": "agent-a", "mcpServers": {"my-tool": dict(TOOLBOX_SHIM)}}
    out = rewriter._repair_control_plane_entry("my-tool", spec["mcpServers"]["my-tool"], "agent-a")
    assert out == TOOLBOX_SHIM
    assert "my-tool" not in KIROCREW_BIN_MCP_SERVERS


def test_a_reserved_name_the_managed_source_cannot_resolve_is_left_as_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``None`` from the managed source (closed ``spec_gate``, unresolvable
    invocation) means there is nothing to repair TO. The entry stays as declared
    and the daemon's gate rules on it, as before -- the rewriter never invents a
    launch."""
    monkeypatch.setattr(agent_mod, "managed_mcp_spec_entry", lambda name, **_kw: None)
    declared = dict(TOOLBOX_SHIM)
    out = rewriter._repair_control_plane_entry("kirocrew-core", declared, "agent-a")
    assert out is declared


def test_the_repair_keeps_restrictions_and_ordinary_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restrictions and an ordinary declared variable are the spec's; command/args move."""
    managed = {"command": "/opt/crew/bin/kirocrew", "args": ["mcp-core"]}
    monkeypatch.setattr(agent_mod, "managed_mcp_spec_entry", lambda name, **_kw: dict(managed))
    declared = {
        **TOOLBOX_SHIM,
        "env": {"MY_FLAG": "1"},
        "autoApprove": ["cron_list"],
        "disabledTools": ["workflow_run"],
    }
    out = rewriter._repair_control_plane_entry("kirocrew-core", declared, "agent-a")
    assert out["command"] == managed["command"] and out["args"] == managed["args"]
    assert out["env"] == {"MY_FLAG": "1"}
    assert out["autoApprove"] == ["cron_list"]
    assert out["disabledTools"] == ["workflow_run"]
    assert declared["command"] == "kirocrew", "the caller's dict is not mutated"


@pytest.mark.parametrize("launch", ["shim", "managed"], ids=["repaired-launch", "exact-launch"])
def test_the_repaired_entry_holds_its_env_to_the_managed_ownership_rule(
    monkeypatch: pytest.MonkeyPatch, caplog, launch: str
) -> None:
    """The launch is ours, so its environment answers to the rule the disk writer
    (``agent._enforce_managed_mcp_ownership``) and the ACP element
    (``session_mcp._managed_element_env``) already apply: Kiro Crew's reserved
    namespace, the loader channels, the home-deriving and launcher-exec keys are
    dropped; an ordinary variable stays. Otherwise a spec's
    ``KIROCREW_APPROVAL_MODE=auto`` reaches a tokened control plane and its
    subagents skip approval. Applies whether the launch needed repair or the spec
    copied the managed command exactly."""
    managed = {"command": "/opt/crew/bin/kirocrew", "args": ["mcp-core"]}
    monkeypatch.setattr(agent_mod, "managed_mcp_spec_entry", lambda name, **_kw: dict(managed))
    base = dict(TOOLBOX_SHIM) if launch == "shim" else dict(managed)
    declared = {
        **base,
        "env": {
            "KIROCREW_APPROVAL_MODE": "auto",
            "KIROCREW_HOME": "/elsewhere",
            "LD_PRELOAD": "/tmp/evil.so",
            "PYTHONPATH": "/tmp/shadow",
            "HOME": "/tmp/otherhome",
            "PATH": "/tmp/bin",
            "NODE_OPTIONS": "--require /tmp/x.js",
            "MY_FLAG": "1",
        },
    }
    with caplog.at_level(logging.WARNING, logger="kiro_crew.mcp_gateway.rewriter"):
        out = rewriter._repair_control_plane_entry("kirocrew-core", declared, "agent-a")
    assert out["command"] == managed["command"] and out["args"] == managed["args"]
    assert out["env"] == {"MY_FLAG": "1"}
    assert declared["env"]["KIROCREW_APPROVAL_MODE"] == "auto", "caller's dict not mutated"
    dropped = [r.getMessage() for r in caplog.records if "dropping" in r.getMessage()]
    assert any("'HOME'" in m for m in dropped) and any("'PATH'" in m for m in dropped)


def test_a_reserved_entry_whose_env_is_entirely_withheld_loses_the_env_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    managed = {"command": "/opt/crew/bin/kirocrew", "args": ["mcp-core"]}
    monkeypatch.setattr(agent_mod, "managed_mcp_spec_entry", lambda name, **_kw: dict(managed))
    out = rewriter._repair_control_plane_entry(
        "kirocrew-core", {**managed, "env": {"KIROCREW_APPROVAL_MODE": "auto"}}, "agent-a"
    )
    assert "env" not in out
    # And an already-managed entry with no env is returned as-is (no churn).
    same = dict(managed)
    assert rewriter._repair_control_plane_entry("kirocrew-core", same, "agent-a") is same


def test_every_reserved_name_is_repaired_not_only_the_control_plane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The set is ``KIROCREW_BIN_MCP_SERVERS`` -- the same one the daemon hands a
    token to -- so a granted opt-in server such as ``kirocrew-dashboard`` gets the
    same repair; it is identity-bound for the same reason."""
    monkeypatch.setattr(
        agent_mod,
        "managed_mcp_spec_entry",
        lambda name, **_kw: {"command": "/opt/crew/bin/kirocrew", "args": [name]},
    )
    for name in KIROCREW_BIN_MCP_SERVERS:
        out = rewriter._repair_control_plane_entry(
            name, {"command": "kirocrew", "args": [name]}, "a"
        )
        assert out["command"] == "/opt/crew/bin/kirocrew", name


def test_the_fingerprint_sees_the_managed_launch_so_an_upgrade_regenerates_overlays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A kirocrew upgrade moves the managed binary while every spec file stays
    byte-identical. The kept-overlay path must not keep serving a stub that
    hashes last version's path, so the launch is a fingerprint input."""
    source = tmp_path / "agents"
    source.mkdir()
    (source / "a.json").write_text(json.dumps({"name": "a", "mcpServers": {}}), encoding="utf-8")
    kwargs = dict(
        source_dir=source,
        settings_path=tmp_path / "mcp.json",
        overlay_dir=tmp_path / "overlay",
        socket_path=tmp_path / "gw.sock",
        work_dir=tmp_path / "wd",
        sandbox_mode="auto",
        approval_mode="interactive",
        stub_set=frozenset({"kirocrew-core", "kirocrew-cron"}),
        pooling_enabled=True,
        forward_env=False,
        identity_keys=(),
    )
    monkeypatch.setattr(
        agent_mod,
        "managed_mcp_spec_entry",
        lambda name, **_kw: {"command": "/toolbox/kirocrew/0.7.0.8/bin/kirocrew", "args": [name]},
    )
    before = rewriter._rewrite_inputs_fingerprint(**kwargs)
    monkeypatch.setattr(
        agent_mod,
        "managed_mcp_spec_entry",
        lambda name, **_kw: {"command": "/toolbox/kirocrew/0.7.0.9/bin/kirocrew", "args": [name]},
    )
    after = rewriter._rewrite_inputs_fingerprint(**kwargs)

    assert before["managed_control_plane"] == {
        "kirocrew-core": ["/toolbox/kirocrew/0.7.0.8/bin/kirocrew", ["kirocrew-core"]],
        "kirocrew-cron": ["/toolbox/kirocrew/0.7.0.8/bin/kirocrew", ["kirocrew-cron"]],
    }
    assert before != after
    assert {k for k in before if before[k] != after[k]} == {"managed_control_plane"}
    assert rewriter._FINGERPRINT_SCHEMA >= 9


# --- The denial reason reaches the session --------------------------------------
#
# Diagnosis on the ticket took three wrong turns because the one accurate line
# ("spawned X is not the spec's Y") lived only in the daemon's stdout. It now
# rides on the frames forwarded to the denied backend and is quoted by that
# backend's ``identity_unattested`` refusal.


def test_the_gate_reports_why_it_denied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = tmp_path / "kirocrew"
    launcher.write_text("#!/bin/sh\n")
    other = tmp_path / "toolbox-exec"
    other.write_text("#!/bin/sh\n")
    monkeypatch.setattr(
        agent_mod,
        "managed_mcp_spec_entry",
        lambda name, **_kw: {"command": str(launcher), "args": ["mcp-core"]},
    )
    from kiro_crew.mcp_gateway import gatewayd as gw

    denial: list[str] = []
    assert not gw._spawns_own_control_plane(
        "kirocrew-core", str(other), ["mcp-core"], env={}, denial=denial
    )
    assert denial == [f"spawned {str(other)!r} is not the spec's {str(launcher)!r}"]
    accepted: list[str] = []
    assert gw._spawns_own_control_plane(
        "kirocrew-core", str(launcher), ["mcp-core"], env={}, denial=accepted
    )
    assert accepted == []
    # A name outside the reserved set is not a denial and records nothing.
    quiet: list[str] = []
    assert not gw._spawns_own_control_plane("my-tool", str(other), [], env={}, denial=quiet)
    assert quiet == []


def test_a_denied_reserved_backend_is_told_why_and_never_handed_the_token() -> None:
    from types import SimpleNamespace

    from kiro_crew.mcp_caller import CallerContext
    from kiro_crew.mcp_gateway import gatewayd as gw

    base = CallerContext(session_key="dashboard:chat-69")
    conn = SimpleNamespace(stub_session_token="tok")
    reason = "spawned '/opt/local/bin/kirocrew' is not the spec's '/tb/kirocrew'"
    denied = SimpleNamespace(control_plane=False, control_plane_denial=reason)
    theirs = SimpleNamespace(control_plane=False, control_plane_denial="")
    ours = SimpleNamespace(control_plane=True, control_plane_denial="")

    handed = gw._caller_for_backend(denied, base, conn)  # type: ignore[arg-type]
    assert handed is not None and handed.identity_denial == reason
    assert handed.session_token == "", "a denial never comes with the token"
    assert base.identity_denial == "" and base.session_token == "", "base never mutated"
    assert gw._caller_for_backend(theirs, base, conn) is base  # type: ignore[arg-type]
    accepted = gw._caller_for_backend(ours, base, conn)  # type: ignore[arg-type]
    assert accepted is not None and accepted.session_token == "tok"
    assert accepted.identity_denial == ""


def test_the_denial_survives_the_meta_round_trip_and_is_absent_when_empty() -> None:
    from kiro_crew.mcp_caller import CALLER_META_KEY, CallerContext, build_caller_meta

    plain = build_caller_meta(CallerContext(session_key="k"))
    assert "identityDenial" not in plain[CALLER_META_KEY]
    reason = "child environment carries non-empty LD_PRELOAD"
    meta = build_caller_meta(CallerContext(session_key="k", identity_denial=reason))
    assert meta[CALLER_META_KEY]["identityDenial"] == reason
    parsed = CallerContext.from_meta(meta)
    assert parsed is not None and parsed.identity_denial == reason
    assert parsed.session_token == ""
