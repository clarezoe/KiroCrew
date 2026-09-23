"""The read-only ACP hooks surface: what Kiro Crew answers, and what it withholds.

Pins four things the surface is only correct if it keeps doing:

* the trigger spellings on the wire are the ACP seven, not the agent-profile
  aliases;
* a list answer is filtered by trigger and by tool identity through Crew's own
  matcher, and a disabled hook is withheld unless the request asked for it;
* the handshake does not announce the capability, so the backend never asks;
* the method that would spawn a command is not served at all.

No server is stood up: the store is driven directly against a tmp directory and
the handlers are called as functions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiro_crew.acp import kas_wire
from kiro_crew.acp._dispatch import classify_notification
from kiro_crew.acp.harness.kas import KasHarness
from kiro_crew.acp.types import JSONRPC_METHOD_NOT_FOUND, KAS_CLIENT_CAPABILITIES, JsonRpcMessage
from kiro_crew.hooks import ScriptHookStore, set_global_hook_store

#: The method Crew must not serve. Spelled here rather than in the product, so
#: the product carries no constant for a method it refuses.
METHOD_HOOKS_EXECUTE = "_kiro/hooks/executeHook"


@pytest.fixture(autouse=True)
def _no_global_store():
    """Leave the process-wide store as the suite found it."""
    yield
    set_global_hook_store(None)  # type: ignore[arg-type]


@pytest.fixture
def store(tmp_path: Path) -> ScriptHookStore:
    return ScriptHookStore(tmp_path)


def _hook(**over) -> kas_wire.NormalizedHook:
    fields: dict = {
        "id": kas_wire.wire_hook_id("script", "abcd1234"),
        "name": "audit",
        "event": "PreToolUse",
        "command": "/bin/true",
    }
    fields.update(over)
    return kas_wire.NormalizedHook(**fields)


class TestTriggerSpellings:
    """The seven this channel uses, and the aliases it does not accept."""

    def test_the_seven_are_exactly_the_acp_set(self):
        assert kas_wire.ACP_HOOK_TRIGGERS == (
            "preToolUse",
            "postToolUse",
            "promptSubmit",
            "agentStop",
            "preTaskExecution",
            "postTaskExecution",
            "sessionStart",
        )

    @pytest.mark.parametrize(
        ("event", "trigger"),
        [
            ("AgentSpawn", "sessionStart"),
            ("UserPromptSubmit", "promptSubmit"),
            ("PreToolUse", "preToolUse"),
            ("PostToolUse", "postToolUse"),
            ("Stop", "agentStop"),
            ("PreTaskExecution", "preTaskExecution"),
            ("PostTaskExecution", "postTaskExecution"),
        ],
    )
    def test_crew_event_resolves_to_the_acp_spelling(self, event: str, trigger: str):
        assert _hook(event=event).acp_trigger == trigger

    def test_camel_case_authoring_spelling_resolves_too(self):
        # The six added triggers are authored in the IDE's camelCase form, and a
        # stored hook carries whichever spelling its author wrote.
        assert _hook(event="userPromptSubmit").acp_trigger == "promptSubmit"
        assert _hook(event="stop").acp_trigger == "agentStop"

    @pytest.mark.parametrize("alias", ["userPromptSubmit", "stop", "agentSpawn"])
    def test_profile_alias_is_not_a_trigger_on_this_channel(self, alias: str):
        # The alias is a valid event to author a hook UNDER; it is not a value
        # the agent can ask for here, so a request naming it selects nothing.
        assert alias not in kas_wire.ACP_HOOK_TRIGGERS
        assert kas_wire.select_hooks([_hook(event="UserPromptSubmit")], trigger=alias) == []

    @pytest.mark.parametrize("event", ["fileCreated", "fileEdited", "fileDeleted", "userTriggered"])
    def test_a_trigger_absent_from_the_seven_serves_nothing(self, event: str):
        assert _hook(event=event).acp_trigger is None
        assert kas_wire.select_hooks([_hook(event=event)], trigger="preToolUse") == []


class TestSelection:
    def test_returns_only_the_requested_trigger(self):
        pre = _hook(id="a", event="PreToolUse")
        post = _hook(id="b", event="PostToolUse")
        assert kas_wire.select_hooks([pre, post], trigger="postToolUse") == [post]

    def test_tool_id_filters_through_crews_matcher(self):
        bash = _hook(id="a", matcher="Bash*")
        read = _hook(id="b", matcher="Read")
        star = _hook(id="c", matcher="*")
        unmatched = _hook(id="d", matcher="")
        pool = [bash, read, star, unmatched]
        selected = kas_wire.select_hooks(pool, trigger="preToolUse", tool_id="Bash")
        assert selected == [bash, star, unmatched]

    def test_a_matcher_is_not_applied_when_no_tool_is_named(self):
        # The request's tool identity is optional, and the agent asking without
        # one is asking for the trigger's whole set.
        bash = _hook(matcher="Bash*")
        assert kas_wire.select_hooks([bash], trigger="preToolUse") == [bash]

    def test_a_matcher_is_not_a_tool_filter_off_the_tool_triggers(self):
        # promptSubmit carries no tool identity, so a matcher there is a context
        # matcher the agent never supplies the subject for.
        hook = _hook(event="UserPromptSubmit", matcher="Bash*")
        assert kas_wire.select_hooks([hook], trigger="promptSubmit", tool_id="Read") == [hook]

    def test_disabled_is_excluded_by_default(self):
        on = _hook(id="a")
        off = _hook(id="b", enabled=False)
        assert kas_wire.select_hooks([on, off], trigger="preToolUse") == [on]

    def test_disabled_is_included_when_asked_for(self):
        on = _hook(id="a")
        off = _hook(id="b", enabled=False)
        selected = kas_wire.select_hooks([on, off], trigger="preToolUse", include_disabled=True)
        assert selected == [on, off]

    def test_an_unknown_trigger_selects_nothing(self):
        assert kas_wire.select_hooks([_hook()], trigger="somethingNewer") == []
        assert kas_wire.select_hooks([_hook()], trigger="") == []

    def test_the_answer_is_bounded(self):
        pool = [_hook(id=str(i)) for i in range(kas_wire.HOOKS_LIST_MAX + 25)]
        assert len(kas_wire.select_hooks(pool, trigger="preToolUse")) == kas_wire.HOOKS_LIST_MAX


class TestProjection:
    def test_the_wire_shape_is_id_name_action(self):
        assert kas_wire.project_hook(_hook(timeout=45)) == {
            "id": "crew:script:abcd1234",
            "name": "audit",
            "action": {"type": "runCommand", "command": "/bin/true", "timeout": 45},
        }

    def test_no_timeout_means_the_field_is_absent(self):
        assert "timeout" not in kas_wire.project_hook(_hook())["action"]

    def test_approved_is_never_sent(self):
        # Sending it would tell the agent the command already carries an
        # approval, which is the one claim this surface must not make.
        assert "approved" not in kas_wire.project_hook(_hook(timeout=30))


class TestNormalizeScriptHook:
    def test_a_stored_hook_becomes_a_wire_hook(self, store: ScriptHookStore):
        created = store.create(
            {
                "name": "audit",
                "event": "PreToolUse",
                "matcher": "Bash*",
                "command": "echo hi",
                "timeout": 20,
            }
        )
        normalized = kas_wire.normalize_script_hook(created)
        assert normalized is not None
        assert normalized.id == f"crew:script:{created.id}"
        assert normalized.name == "audit"
        assert normalized.event == "PreToolUse"
        assert normalized.command == "echo hi"
        assert normalized.matcher == "Bash*"
        assert normalized.timeout == 20
        assert normalized.enabled is True

    def test_a_commandless_hook_is_not_served(self, store: ScriptHookStore):
        created = store.create(
            {
                "name": "skills only",
                "event": "UserPromptSubmit",
                "command": "",
                "skills": ["kirocrew-dev/prepare-pr"],
            }
        )
        assert kas_wire.normalize_script_hook(created) is None

    def test_an_id_less_hook_is_not_served(self):
        class _Bare:
            id = ""
            name = "x"
            event = "PreToolUse"
            command = "echo hi"
            matcher = ""
            timeout = 30
            enabled = True

        assert kas_wire.normalize_script_hook(_Bare()) is None

    def test_an_unmapped_event_is_not_served(self):
        class _Filed:
            id = "abcd1234"
            name = "x"
            event = "fileCreated"  # real hook event, absent from the ACP seven
            command = "echo hi"
            matcher = ""
            timeout = 30
            enabled = True

        assert kas_wire.normalize_script_hook(_Filed()) is None

    def test_a_non_string_field_is_not_served(self):
        class _Wrong:
            id = 7
            name = "x"
            event = "PreToolUse"
            command = "echo hi"
            matcher = ""
            timeout = 30
            enabled = True

        assert kas_wire.normalize_script_hook(_Wrong()) is None

    def test_a_non_int_timeout_degrades_to_absent(self):
        class _Odd:
            id = "abcd1234"
            name = "x"
            event = "PreToolUse"
            command = "echo hi"
            matcher = ""
            timeout = True  # a bool is an int subclass and is not a duration
            enabled = True

        normalized = kas_wire.normalize_script_hook(_Odd())
        assert normalized is not None
        assert normalized.timeout is None

    def test_the_store_is_read_through_the_global_accessor(self, store: ScriptHookStore):
        store.create({"name": "audit", "event": "PreToolUse", "command": "echo hi"})
        assert kas_wire.crew_hooks() == []  # no global store registered yet
        set_global_hook_store(store)
        assert [h.name for h in kas_wire.crew_hooks()] == ["audit"]


class TestListResponse:
    def test_lists_crews_hooks_for_a_trigger_filtered_by_tool_id(self, store: ScriptHookStore):
        bash = store.create(
            {"name": "on bash", "event": "PreToolUse", "matcher": "Bash*", "command": "echo a"}
        )
        store.create(
            {"name": "on read", "event": "PreToolUse", "matcher": "Read", "command": "echo b"}
        )
        store.create({"name": "on stop", "event": "Stop", "command": "echo c"})
        set_global_hook_store(store)

        result = kas_wire.hooks_list_response(
            {"trigger": "preToolUse", "sessionId": "s1", "toolId": "Bash"}
        )
        assert [h["name"] for h in result["hooks"]] == ["on bash"]
        assert result["hooks"][0]["id"] == f"crew:script:{bash.id}"

    def test_disabled_excluded_by_default_and_included_on_request(self, store: ScriptHookStore):
        store.create({"name": "on", "event": "PreToolUse", "command": "echo a"})
        off = store.create({"name": "off", "event": "PreToolUse", "command": "echo b"})
        store.toggle(off.id)
        set_global_hook_store(store)

        default = kas_wire.hooks_list_response({"trigger": "preToolUse", "sessionId": "s1"})
        assert [h["name"] for h in default["hooks"]] == ["on"]

        widened = kas_wire.hooks_list_response(
            {"trigger": "preToolUse", "sessionId": "s1", "includeDisabled": True}
        )
        assert [h["name"] for h in widened["hooks"]] == ["on", "off"]

    def test_include_disabled_must_be_the_boolean(self, store: ScriptHookStore):
        off = store.create({"name": "off", "event": "PreToolUse", "command": "echo b"})
        store.toggle(off.id)
        set_global_hook_store(store)
        for value in ["true", 1, {}, None]:
            result = kas_wire.hooks_list_response(
                {"trigger": "preToolUse", "sessionId": "s1", "includeDisabled": value}
            )
            assert result["hooks"] == []

    def test_host_supplied_shapes_degrade_to_absent(self):
        # Every field is host data. A wrong shape answers empty rather than
        # raising inside the turn's dispatch.
        assert kas_wire.hooks_list_response({}, hooks=[_hook()]) == {"hooks": []}
        assert kas_wire.hooks_list_response({"trigger": 7}, hooks=[_hook()]) == {"hooks": []}
        listed = kas_wire.hooks_list_response(
            {"trigger": "preToolUse", "sessionId": None, "toolId": []}, hooks=[_hook()]
        )
        assert [h["id"] for h in listed["hooks"]] == ["crew:script:abcd1234"]

    def test_tool_tags_and_workspace_paths_do_not_narrow(self):
        hook = _hook(matcher="Bash*")
        result = kas_wire.hooks_list_response(
            {
                "trigger": "preToolUse",
                "sessionId": "s1",
                "toolTags": ["destructive"],
                "workspacePaths": ["/nowhere"],
            },
            hooks=[hook],
        )
        assert [h["id"] for h in result["hooks"]] == [hook.id]

    def test_no_store_answers_empty_rather_than_raising(self):
        assert kas_wire.hooks_list_response({"trigger": "preToolUse", "sessionId": "s1"}) == {
            "hooks": []
        }


class TestSessionStartResponse:
    def test_the_buffer_is_empty(self):
        assert kas_wire.hooks_session_start_response({"trigger": "sessionStart"}) == {"results": []}

    def test_every_trigger_answers_the_same_way(self):
        assert kas_wire.hooks_session_start_response({"trigger": "preToolUse"}) == {"results": []}
        assert kas_wire.hooks_session_start_response({}) == {"results": []}


class TestListedHookStore:
    def test_a_listed_id_is_recorded_against_its_session(self):
        recorder = kas_wire.ListedHookStore()
        hook = _hook()
        recorder.record("s1", [hook])
        assert recorder.lookup("s1", hook.id) == hook

    def test_an_id_listed_for_another_session_is_not_found(self):
        recorder = kas_wire.ListedHookStore()
        hook = _hook()
        recorder.record("s1", [hook])
        assert recorder.lookup("s2", hook.id) is None

    def test_an_unlisted_id_is_not_found(self):
        recorder = kas_wire.ListedHookStore()
        recorder.record("s1", [_hook()])
        assert recorder.lookup("s1", "crew:script:forged") is None

    def test_the_recorded_hook_carries_the_command_crew_listed(self):
        recorder = kas_wire.ListedHookStore()
        recorder.record("s1", [_hook(command="/bin/true")])
        found = recorder.lookup("s1", "crew:script:abcd1234")
        assert found is not None
        assert found.command == "/bin/true"

    def test_forget_drops_a_session(self):
        recorder = kas_wire.ListedHookStore()
        hook = _hook()
        recorder.record("s1", [hook])
        recorder.forget("s1")
        assert recorder.lookup("s1", hook.id) is None

    def test_an_empty_session_id_records_nothing(self):
        recorder = kas_wire.ListedHookStore()
        recorder.record("", [_hook()])
        assert recorder.lookup("", "crew:script:abcd1234") is None

    def test_sessions_are_bounded_oldest_first(self):
        recorder = kas_wire.ListedHookStore(max_sessions=2)
        for session in ("s1", "s2", "s3"):
            recorder.record(session, [_hook(id=f"crew:script:{session}")])
        assert recorder.lookup("s1", "crew:script:s1") is None
        assert recorder.lookup("s3", "crew:script:s3") is not None

    def test_ids_within_a_session_are_bounded_oldest_first(self):
        recorder = kas_wire.ListedHookStore(max_per_session=2)
        for index in range(3):
            recorder.record("s1", [_hook(id=f"crew:script:{index}")])
        assert recorder.lookup("s1", "crew:script:0") is None
        assert recorder.lookup("s1", "crew:script:2") is not None

    def test_re_listing_refreshes_rather_than_duplicates(self):
        recorder = kas_wire.ListedHookStore(max_per_session=2)
        first = _hook(id="crew:script:a", command="/bin/true")
        second = _hook(id="crew:script:b")
        recorder.record("s1", [first, second])
        recorder.record("s1", [first])
        recorder.record("s1", [_hook(id="crew:script:c")])
        # The re-listed id was refreshed, so the untouched one is the evicted one.
        assert recorder.lookup("s1", "crew:script:a") is not None
        assert recorder.lookup("s1", "crew:script:b") is None

    def test_a_list_answer_records_what_it_returned(self, store: ScriptHookStore):
        created = store.create({"name": "audit", "event": "PreToolUse", "command": "echo hi"})
        set_global_hook_store(store)
        kas_wire.hooks_list_response({"trigger": "preToolUse", "sessionId": "sess-record"})
        found = kas_wire.listed_hooks().lookup("sess-record", f"crew:script:{created.id}")
        assert found is not None
        assert found.command == "echo hi"
        kas_wire.listed_hooks().forget("sess-record")

    def test_a_withheld_hook_is_not_recorded(self, store: ScriptHookStore):
        off = store.create({"name": "off", "event": "PreToolUse", "command": "echo b"})
        store.toggle(off.id)
        set_global_hook_store(store)
        kas_wire.hooks_list_response({"trigger": "preToolUse", "sessionId": "sess-withheld"})
        assert kas_wire.listed_hooks().lookup("sess-withheld", f"crew:script:{off.id}") is None


class TestTheCapabilityIsNotAnnounced:
    """The surface exists and the backend is not told, which is the whole gate.

    A hooks provider is chosen at handshake: announcing the capability is what
    makes the backend route hook extraction to its client. Until an execute path
    exists, announcing it would hand the agent a surface whose third method Crew
    answers with an error.
    """

    def test_the_handshake_meta_does_not_carry_hooks(self):
        kiro_meta = KAS_CLIENT_CAPABILITIES["_meta"]["kiro"]
        assert "hooks" not in kiro_meta

    def test_the_harness_declares_the_same_meta(self):
        kiro_meta = KasHarness().client_capabilities["_meta"]["kiro"]
        assert "hooks" not in kiro_meta


class TestExecuteHookIsNotServed:
    def test_it_classifies_as_an_unknown_server_request(self):
        msg = JsonRpcMessage(
            id=11,
            method=METHOD_HOOKS_EXECUTE,
            params={"hookId": "crew:script:abcd1234", "command": "rm -rf /", "sessionId": "s1"},
        )
        assert classify_notification(msg) == "server_request_unknown"

    def test_the_unknown_answer_is_method_not_found(self):
        assert JSONRPC_METHOD_NOT_FOUND == -32601

    def test_the_product_names_no_execute_method(self):
        # A constant for it would read as a promise to serve it.
        named = [value for value in vars(kas_wire).values() if isinstance(value, str)]
        assert METHOD_HOOKS_EXECUTE not in named

    def test_the_two_read_only_methods_are_named(self):
        assert kas_wire.METHOD_HOOKS_LIST == "_kiro/hooks/list"
        assert kas_wire.METHOD_HOOKS_SESSION_START == "_kiro/hooks/sessionStart"
