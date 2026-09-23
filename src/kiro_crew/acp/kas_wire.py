"""Localized parsing for the second ACP backend's KAS-specific wire surfaces.

Two surfaces live here, and they share one property: the shapes are the second
backend's, not ordinary ACP, so this module is the ONE place Kiro Crew reads or
builds them and an adjustment is a single-file edit.

1. ``session/update`` display/telemetry frames, whose discriminant blob arrives
   in a different shape than ``kiro-cli``'s. The handlers that act on a parsed
   frame live in :mod:`kiro_crew.acp.session_handle`.
2. The **hooks** surface: the backend can delegate hook extraction to its ACP
   client, asking ``_kiro/hooks/list`` for the hooks matching a trigger and
   ``_kiro/hooks/sessionStart`` for results computed before the turn. Kiro Crew
   answers both from its own hook store, so the matcher stays on the side where
   the UI that authored the hook lives.

The literals below are only what Kiro Crew must match to route a frame or answer
a request; the backend's own internals are not documented here.

Read-only by construction. The third method of that surface,
``_kiro/hooks/executeHook``, is the one that spawns a command, and nothing here
implements it: an inbound request naming it is classified as an unknown server
request and answered ``-32601``. What this module does own is the record of
which hook ids Kiro Crew actually listed for a session
(:class:`ListedHookStore`), which is the fact an execute path has to consult
before it may run anything.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from kiro_crew.hooks import (
    HOOK_EVENT_AGENT_SPAWN,
    HOOK_EVENT_POST_TOOL_USE,
    HOOK_EVENT_PRE_TOOL_USE,
    HOOK_EVENT_STOP,
    HOOK_EVENT_USER_PROMPT_SUBMIT,
    _tool_matches,
)

# ── Envelope keys ──
# Where a frame's discriminant blob sits; both hops are guarded on read.
META_KEY = "_meta"
KIRO_KEY = "kiro"

# ── Frame discriminants Crew matches on ──
KIND_CONTEXT_USAGE = "context_usage"
KIND_TURN_COMPLETION = "turn_completion"
KIND_SUMMARIZATION_STARTED = "summarization_started"
KIND_SUMMARIZATION_COMPLETED = "summarization_completed"
KIND_SUMMARIZATION_FAILED = "summarization_failed"
KIND_STEERING_QUEUED = "steering_queued"
KIND_STEERING_INJECTED = "steering_injected"
KIND_STEERING_CLEARED = "steering_cleared"
KIND_AGENT_SUBTASK = "agent-subtask"  # hyphenated, not underscored

# Summarization maps to Crew's compaction status; steering to mid-turn steer.
SUMMARIZATION_KINDS = frozenset(
    {KIND_SUMMARIZATION_STARTED, KIND_SUMMARIZATION_COMPLETED, KIND_SUMMARIZATION_FAILED}
)
STEERING_KINDS = frozenset(
    {KIND_STEERING_QUEUED, KIND_STEERING_INJECTED, KIND_STEERING_CLEARED}
)

# ── Payload fields Crew reads ──
FIELD_KIND = "kind"
FIELD_USAGE_PERCENTAGE = "usagePercentage"
FIELD_CONVERSATION_SUMMARY = "conversationSummary"
FIELD_CONTENT = "content"
FIELD_PROMPT_TURN_SUMMARIES = "promptTurnSummaries"
FIELD_AGENT_SUBTASK_ID = "agentSubtaskId"
FIELD_PIPELINE = "pipeline"
FIELD_STAGES = "stages"
FIELD_UNIT = "unit"
FIELD_USAGE = "usage"
UNIT_CREDIT = "credit"


def kiro_meta(update: dict) -> dict[str, Any] | None:
    """Return the frame's discriminant blob, or ``None`` when absent.

    Both hops are ``isinstance``-guarded so a frame without it (a different
    backend, or a malformed one) falls through to the shared parser rather than
    raising. The single extraction step every handler shares — do not inline the
    two-step ``.get`` elsewhere.
    """
    meta = update.get(META_KEY)
    kiro = meta.get(KIRO_KEY) if isinstance(meta, dict) else None
    return kiro if isinstance(kiro, dict) else None


def turn_credits(kiro: dict) -> float | None:
    """Sum the per-turn credit cost from a ``turn_completion`` frame.

    Only ``credit``-unit entries contribute (the acp provider bills in credits).
    Returns the total to ASSIGN — the frame carries the whole turn's summary, so
    a replayed/duplicate frame reports the same total and must not accumulate —
    or ``None`` when there is no summaries list, so the caller leaves the prior
    value untouched rather than zeroing it.
    """
    summaries = kiro.get(FIELD_PROMPT_TURN_SUMMARIES)
    if not isinstance(summaries, list):
        return None
    # Deferred import: _token_count lives in the dispatch module, which imports
    # types; importing it at module scope would risk an import cycle once
    # session_handle imports this module.
    from kiro_crew.acp._dispatch import _token_count

    total = 0.0
    for entry in summaries:
        if not isinstance(entry, dict) or entry.get(FIELD_UNIT) != UNIT_CREDIT:
            continue
        value = _token_count(entry.get(FIELD_USAGE))
        if value is not None:
            total += float(value)
    return total


# ── The hooks surface ──
#
# Method names. ``executeHook`` is deliberately NOT named here: a constant for a
# method nothing serves reads as a promise, and the only correct answer to it in
# this build is the unknown-request one every unnamed method already gets.
METHOD_HOOKS_LIST = "_kiro/hooks/list"
METHOD_HOOKS_SESSION_START = "_kiro/hooks/sessionStart"

# The trigger spellings THIS surface uses. They are not the agent-profile
# aliases: ``promptSubmit`` / ``agentStop`` / ``sessionStart`` stand where a
# profile writes ``userPromptSubmit`` / ``stop`` / ``agentSpawn``. Both sets
# normalize on the backend's side, but only these seven are what it asks for
# over this channel, so these seven are what Crew emits.
ACP_TRIGGER_PRE_TOOL_USE = "preToolUse"
ACP_TRIGGER_POST_TOOL_USE = "postToolUse"
ACP_TRIGGER_PROMPT_SUBMIT = "promptSubmit"
ACP_TRIGGER_AGENT_STOP = "agentStop"
ACP_TRIGGER_PRE_TASK_EXECUTION = "preTaskExecution"
ACP_TRIGGER_POST_TASK_EXECUTION = "postTaskExecution"
ACP_TRIGGER_SESSION_START = "sessionStart"

#: Every trigger this surface defines, in the order the covenant lists them.
ACP_HOOK_TRIGGERS: tuple[str, ...] = (
    ACP_TRIGGER_PRE_TOOL_USE,
    ACP_TRIGGER_POST_TOOL_USE,
    ACP_TRIGGER_PROMPT_SUBMIT,
    ACP_TRIGGER_AGENT_STOP,
    ACP_TRIGGER_PRE_TASK_EXECUTION,
    ACP_TRIGGER_POST_TASK_EXECUTION,
    ACP_TRIGGER_SESSION_START,
)

#: Crew's own event spelling -> the trigger spelling this surface asks for.
#:
#: Both spellings of each Crew event are keys, because the event string on a
#: stored hook is whatever the surface that authored it wrote: the hook store
#: carries the PascalCase names, and the camelCase IDE names are the spelling
#: the six additional triggers are authored under.
#:
#: The file triggers and the manual one are absent on purpose. They are not in
#: ``ACP_HOOK_TRIGGERS``, so a hook authored for one of them answers no request
#: on this channel rather than being folded into a trigger the agent did ask
#: for.
_CREW_EVENT_TO_ACP_TRIGGER: dict[str, str] = {
    HOOK_EVENT_AGENT_SPAWN: ACP_TRIGGER_SESSION_START,
    "agentSpawn": ACP_TRIGGER_SESSION_START,
    HOOK_EVENT_USER_PROMPT_SUBMIT: ACP_TRIGGER_PROMPT_SUBMIT,
    "userPromptSubmit": ACP_TRIGGER_PROMPT_SUBMIT,
    HOOK_EVENT_PRE_TOOL_USE: ACP_TRIGGER_PRE_TOOL_USE,
    "preToolUse": ACP_TRIGGER_PRE_TOOL_USE,
    HOOK_EVENT_POST_TOOL_USE: ACP_TRIGGER_POST_TOOL_USE,
    "postToolUse": ACP_TRIGGER_POST_TOOL_USE,
    HOOK_EVENT_STOP: ACP_TRIGGER_AGENT_STOP,
    "stop": ACP_TRIGGER_AGENT_STOP,
    "PreTaskExecution": ACP_TRIGGER_PRE_TASK_EXECUTION,
    "preTaskExecution": ACP_TRIGGER_PRE_TASK_EXECUTION,
    "PostTaskExecution": ACP_TRIGGER_POST_TASK_EXECUTION,
    "postTaskExecution": ACP_TRIGGER_POST_TASK_EXECUTION,
}

#: The two triggers whose request carries a tool identity, so they are the two
#: where a hook's matcher is a TOOL matcher rather than a context one.
_TOOL_TRIGGERS = frozenset({ACP_TRIGGER_PRE_TOOL_USE, ACP_TRIGGER_POST_TOOL_USE})

#: Cap on one list response. The request is per trigger and per tool call, so an
#: unbounded answer would put the whole hook store on the wire inside a turn.
#: A store larger than this is already past the per-event caps the merge path
#: enforces, so the cap is a wire bound rather than a policy.
HOOKS_LIST_MAX = 200

#: Id prefix, so an id Crew minted is distinguishable from one the backend
#: derived from a file path in its own loader. Nothing parses the id back: the
#: record in :class:`ListedHookStore` is what an execute path reads.
_ID_PREFIX = "crew"


@dataclass(frozen=True)
class NormalizedHook:
    """One hook, in the shape this surface's reader consumes.

    The reader deliberately does not know how a hook was WRITTEN. Crew's stored
    hooks and an agent spec's hooks are different shapes, and a spec's own shape
    is itself widening; every one of them reaches selection and projection as
    this dataclass, so a new source is one adapter and no change to the three
    steps below it.

    ``event`` keeps Crew's own spelling and :attr:`acp_trigger` resolves it, so
    a hook whose event maps to no trigger on this channel is visibly unserved
    rather than silently renamed.
    """

    id: str
    name: str
    event: str
    command: str = ""
    matcher: str = ""
    timeout: int | None = None
    enabled: bool = True

    @property
    def acp_trigger(self) -> str | None:
        """The trigger spelling this hook answers on, or ``None`` for neither."""
        return _CREW_EVENT_TO_ACP_TRIGGER.get(self.event)


def wire_hook_id(source: str, native_id: str) -> str:
    """The id Crew puts on the wire for one hook."""
    return f"{_ID_PREFIX}:{source}:{native_id}"


def normalize_script_hook(hook: Any) -> NormalizedHook | None:
    """Adapt one stored script hook, or ``None`` when it cannot be served.

    ``None`` covers the two cases that are not errors: a hook whose event has no
    trigger on this channel, and a hook with no command. A commandless hook is
    an injection of skill text that the store's own fire path composes, so
    listing it here would promise an action this surface has no shape for.
    """
    event = getattr(hook, "event", "")
    command = getattr(hook, "command", "")
    native_id = getattr(hook, "id", "")
    if not isinstance(event, str) or not isinstance(command, str) or not isinstance(native_id, str):
        return None
    if not native_id or not command.strip():
        return None
    if event not in _CREW_EVENT_TO_ACP_TRIGGER:
        return None
    timeout = getattr(hook, "timeout", None)
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        timeout = None
    matcher = getattr(hook, "matcher", "")
    name = getattr(hook, "name", "")
    return NormalizedHook(
        id=wire_hook_id("script", native_id),
        name=name if isinstance(name, str) and name else native_id,
        event=event,
        command=command,
        matcher=matcher if isinstance(matcher, str) else "",
        timeout=timeout,
        enabled=bool(getattr(hook, "enabled", True)),
    )


def crew_hooks() -> list[NormalizedHook]:
    """Every Crew hook this surface can serve, normalized.

    The store is the one the dashboard's hook UI writes. When it has not been
    initialized there are no hooks to serve, which answers as an empty list
    rather than an error: an unanswered request would park the agent's turn.
    """
    # Deferred import: the store getter reaches the module this one already
    # imports names from, and the getter is the only part that must be read at
    # CALL time — a module-scope binding would freeze whichever store existed at
    # import time, which for a gateway that builds its store during boot is
    # none.
    from kiro_crew.hooks import get_global_hook_store

    store = get_global_hook_store()
    if store is None:
        return []
    out: list[NormalizedHook] = []
    for hook in store.list_all():
        normalized = normalize_script_hook(hook)
        if normalized is not None:
            out.append(normalized)
    return out


def select_hooks(
    hooks: Iterable[NormalizedHook],
    *,
    trigger: str,
    tool_id: str = "",
    include_disabled: bool = False,
) -> list[NormalizedHook]:
    """The hooks that answer one request, in store order.

    Four filters, in the order a caller can reason about:

    * an unrecognized trigger selects nothing. A trigger Crew does not model is
      one it cannot claim a hook matches;
    * the hook's event must resolve to exactly that trigger;
    * a disabled hook is excluded unless the request asked for it. The default
      is the execution-facing one, so a disabled hook is never returned to a
      caller that would act on it;
    * on the two tool triggers, a hook with a matcher must match the named tool
      through Crew's own tool matcher. A request that names no tool skips this
      filter rather than dropping every matcher'd hook, because the agent asked
      for the trigger's whole set.
    """
    if trigger not in ACP_HOOK_TRIGGERS:
        return []
    out: list[NormalizedHook] = []
    for hook in hooks:
        if hook.acp_trigger != trigger:
            continue
        if not hook.enabled and not include_disabled:
            continue
        if trigger in _TOOL_TRIGGERS and hook.matcher and tool_id:
            if not _tool_matches(hook.matcher, tool_id):
                continue
        out.append(hook)
        if len(out) >= HOOKS_LIST_MAX:
            break
    return out


def project_hook(hook: NormalizedHook) -> dict[str, Any]:
    """One hook in the wire shape: ``{id, name, action}``.

    ``approved`` is never sent. It is the field that tells the agent this
    command already carries an approval, and Crew has no approval to report for
    a hook it has not been asked to run.
    """
    action: dict[str, Any] = {"type": "runCommand", "command": hook.command}
    if hook.timeout is not None:
        action["timeout"] = hook.timeout
    return {"id": hook.id, "name": hook.name, "action": action}


def _str_param(params: Mapping[str, Any], key: str) -> str:
    """One string field of a host-supplied params object, or ``""``."""
    value = params.get(key)
    return value if isinstance(value, str) else ""


def hooks_list_response(
    params: Mapping[str, Any],
    hooks: Iterable[NormalizedHook] | None = None,
) -> dict[str, Any]:
    """Answer ``_kiro/hooks/list``.

    Every field of ``params`` is host-supplied, so each is read through an
    isinstance guard and a shape that is not the expected one degrades to
    absent. ``toolTags`` and ``workspacePaths`` are read and deliberately
    ignored: Crew's hooks carry no tag matcher, and they are not scoped to a
    workspace root, so narrowing by either would drop hooks their author expects
    to fire.

    Records what it answered against the session, which is what makes an id
    executable later: an id that was never listed for a session was never minted
    by Crew for it.
    """
    trigger = _str_param(params, "trigger")
    session_id = _str_param(params, "sessionId")
    tool_id = _str_param(params, "toolId")
    include_disabled = params.get("includeDisabled") is True
    selected = select_hooks(
        crew_hooks() if hooks is None else hooks,
        trigger=trigger,
        tool_id=tool_id,
        include_disabled=include_disabled,
    )
    if session_id:
        listed_hooks().record(session_id, selected)
    return {"hooks": [project_hook(hook) for hook in selected]}


def hooks_session_start_response(params: Mapping[str, Any]) -> dict[str, Any]:
    """Answer ``_kiro/hooks/sessionStart`` with the results buffered for a turn.

    Empty for every trigger, which is the honest answer for this build rather
    than a stub: a precomputed result is the OUTPUT of a hook, every Crew hook
    this surface can serve carries a command as its payload, and no command runs
    from here. The request is still ANSWERED, because an unanswered one parks
    the agent's turn waiting for a response that never arrives, and the agent's
    own policy for this surface is to treat a failure as "no hooks" — which is
    indistinguishable from a refusal and is the reason a refusal must never be
    expressed that way.

    ``params`` is accepted and not read: there is no field whose value could
    make the answer non-empty.
    """
    return {"results": []}


class ListedHookStore:
    """Which hook ids Crew listed, per session.

    The point of the record is the direction it can answer in the negative. An
    execute path may only run a hook Crew itself put on the wire for THAT
    session, and the id the agent sends back is host-supplied, so "is this id
    one of mine" has to be a lookup rather than a parse of the id's shape. The
    stored hook is Crew's own, which is also what makes the command
    authoritative: an execute path reads the command from here, never from the
    request.

    Bounded on both axes, because both are driven by the host: sessions are
    evicted oldest-first, and ids within a session likewise. Eviction can only
    ever narrow what is executable, so the bound fails in the safe direction.
    """

    def __init__(self, *, max_sessions: int = 64, max_per_session: int = 512) -> None:
        self._max_sessions = max_sessions
        self._max_per_session = max_per_session
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, OrderedDict[str, NormalizedHook]] = OrderedDict()

    def record(self, session_id: str, hooks: Sequence[NormalizedHook]) -> None:
        """Record the hooks one list response returned for ``session_id``."""
        if not session_id:
            return
        with self._lock:
            listed = self._sessions.pop(session_id, None)
            if listed is None:
                listed = OrderedDict()
            for hook in hooks:
                listed.pop(hook.id, None)
                listed[hook.id] = hook
                while len(listed) > self._max_per_session:
                    listed.popitem(last=False)
            self._sessions[session_id] = listed
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)

    def lookup(self, session_id: str, hook_id: str) -> NormalizedHook | None:
        """The hook Crew listed under ``hook_id``, or ``None`` if it listed none."""
        with self._lock:
            listed = self._sessions.get(session_id)
            return listed.get(hook_id) if listed is not None else None

    def forget(self, session_id: str) -> None:
        """Drop one session's record, on the session going away."""
        with self._lock:
            self._sessions.pop(session_id, None)


_LISTED_HOOKS = ListedHookStore()


def listed_hooks() -> ListedHookStore:
    """The process-wide record of what each session was told."""
    return _LISTED_HOOKS
