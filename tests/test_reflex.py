"""Unit tests for gbrain-retrieval-reflex.

Stock CPython: no GBrain, no Hermes runtime, no network, no sockets. The HTTP
edge is patched, so these tests also prove the plugin fails soft when GBrain
is unreachable.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def _load_plugin():
    """Load the plugin package from its directory (the name has dashes)."""
    name = "_gbrain_reflex_under_test"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


reflex = _load_plugin()


class StatePaths(unittest.TestCase):
    """Where the plugin looks for tokens, sockets and audit output."""

    def test_gbrain_home_wins_and_order_is_stable(self):
        env = {"GBRAIN_HOME": "/opt/gb", "HOME": "/h", "HERMES_HOME": "/hh"}
        with patch.dict(os.environ, env, clear=False):
            dirs = reflex._gbrain_dirs()
        self.assertEqual(
            [str(d) for d in dirs],
            ["/opt/gb", "/h/.gbrain", "/hh/.gbrain"],
        )

    def test_gbrain_home_duplicate_is_deduped(self):
        env = {"GBRAIN_HOME": "/h/.gbrain", "HOME": "/h", "HERMES_HOME": "/hh"}
        with patch.dict(os.environ, env, clear=False):
            dirs = reflex._gbrain_dirs()
        self.assertEqual([str(d) for d in dirs], ["/h/.gbrain", "/hh/.gbrain"])

    def test_token_and_env_paths_derive_from_the_same_bases(self):
        env = {"GBRAIN_HOME": "/opt/gb", "HOME": "/h", "HERMES_HOME": "/hh"}
        with patch.dict(os.environ, env, clear=False):
            tokens = [str(p) for p in reflex._token_paths()]
            envs = [str(p) for p in reflex._env_paths()]
        self.assertEqual(
            tokens,
            ["/opt/gb/hermes-mcp.token", "/h/.gbrain/hermes-mcp.token",
             "/hh/.gbrain/hermes-mcp.token"],
        )
        self.assertEqual(envs, ["/hh/.env", "/h/.hermes/.env"])

    def test_socket_candidates_override_first_then_bases(self):
        env = {
            "GBRAIN_RESOLVE_SOCKET": "/tmp/explicit.sock",
            "GBRAIN_HOME": "/opt/gb",
            "HOME": "/h",
            "HERMES_HOME": "/hh",
        }
        with patch.dict(os.environ, env, clear=False):
            cands = [str(p) for p in reflex._socket_candidates()]
        self.assertEqual(cands[0], "/tmp/explicit.sock")
        self.assertEqual(cands[1], "/opt/gb/brain.pglite/.gbrain-resolve.sock")
        self.assertEqual(cands[2], "/opt/gb/.gbrain-resolve.sock")
        self.assertEqual(cands[3], "/h/.gbrain/brain.pglite/.gbrain-resolve.sock")
        self.assertEqual(len(cands), len(set(cands)), "candidates must be unique")

    def test_no_user_specific_paths_are_baked_in(self):
        source = (ROOT / "__init__.py").read_text()
        for leaked in ("/home/", "/var/lib/hermes", "/data/.hermes"):
            self.assertNotIn(leaked, source, f"{leaked} must come from env")


class MessageNormalization(unittest.TestCase):
    def test_string_dict_and_list_forms(self):
        self.assertEqual(reflex._normalize_user_message("hello there"), "hello there")
        self.assertEqual(
            reflex._normalize_user_message([{"text": "a"}, {"content": "b"}]),
            "a\nb",
        )
        self.assertEqual(
            reflex._normalize_user_message({"content": "from a dict"}),
            "from a dict",
        )

    def test_strip_injected_blocks_keeps_the_user_text(self):
        text = (
            "what changed?\n\n"
            "## Brain pages (ambient push)\n- ops/thing: stale\n"
        )
        self.assertEqual(reflex._strip_injected_blocks(text), "what changed?")

    def test_strip_injected_blocks_handles_memory_budget(self):
        text = "real question\n\n## MEMORY budget (1/2)\nnote"
        self.assertEqual(reflex._strip_injected_blocks(text), "real question")


class VolunteerWindow(unittest.TestCase):
    def test_current_turn_is_last_and_roles_filtered(self):
        history = [
            {"role": "system", "content": "ignore me"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
        ]
        window = reflex._build_volunteer_window("now", history)
        lines = window.splitlines()
        self.assertEqual(lines[0], "user: first")
        self.assertEqual(lines[1], "assistant: reply")
        self.assertTrue(lines[-1].startswith("user: now"))

    def test_injected_blocks_never_enter_the_window(self):
        history = [
            {
                "role": "user",
                "content": "earlier\n\n## Brain pages (ambient push)\n- old/page: noise",
            }
        ]
        window = reflex._build_volunteer_window("now", history)
        self.assertNotIn("old/page", window)


class MergeRank(unittest.TestCase):
    def test_volunteer_first_then_query_and_dedupe(self):
        volunteered = [{"slug": "a/one", "confidence": 0.9}]
        queried = [
            {"slug": "a/one", "confidence": 0.95},
            {"slug": "b/two", "confidence": 0.7},
        ]
        pages, source = reflex._merge_rank_pages(volunteered, queried)
        self.assertEqual([p["slug"] for p in pages], ["a/one", "b/two"])
        self.assertEqual(source, "volunteer+query")
        self.assertEqual(pages[0]["source"], "volunteer")
        self.assertEqual(pages[1]["source"], "query")

    def test_query_only_is_sorted_by_score(self):
        queried = [
            {"slug": "low", "confidence": 0.2},
            {"slug": "high", "confidence": 0.8},
        ]
        pages, source = reflex._merge_rank_pages([], queried)
        self.assertEqual([p["slug"] for p in pages], ["high", "low"])
        self.assertEqual(source, "query")

    def test_non_dicts_and_slugless_entries_are_dropped(self):
        pages, source = reflex._merge_rank_pages(
            ["nonsense", {"confidence": 0.9}], [None, {"slug": "ok"}]
        )
        self.assertEqual([p["slug"] for p in pages], ["ok"])
        self.assertEqual(source, "query")

    def test_cap_is_respected(self):
        volunteered = [{"slug": f"v/{i}"} for i in range(reflex._MAX_POINTERS + 3)]
        queried = [{"slug": f"q/{i}"} for i in range(reflex._MAX_POINTERS + 3)]
        pages, _ = reflex._merge_rank_pages(volunteered, queried)
        self.assertEqual(len(pages), reflex._MAX_POINTERS)


class HookGate(unittest.TestCase):
    """pre_llm_call must be cheap, silent, and never raise."""

    def _boom(self, *a, **k):
        raise AssertionError("HTTP must not be attempted for this message")

    def test_trivial_and_short_messages_skip_http(self):
        with patch.object(reflex, "_volunteer_via_http", self._boom), patch.object(
            reflex, "_query_via_http", self._boom
        ):
            for text in ("ok", "thanks!", "hi", "a"):
                self.assertIsNone(reflex.on_pre_llm_call(user_message=text))

    def test_pages_are_injected_and_audited(self):
        seen: list[dict] = []
        with patch.object(
            reflex, "_volunteer_via_http", lambda window: [{"slug": "ops/thing"}]
        ), patch.object(reflex, "_query_via_http", lambda text: []), patch.object(
            reflex, "_audit", lambda payload: seen.append(payload)
        ), patch.object(reflex, "_resolve_socket_path", lambda: None):
            out = reflex.on_pre_llm_call(user_message="tell me about ops/thing")
        self.assertIsNotNone(out)
        self.assertEqual(out["target"], "user_message")
        self.assertIn("ops/thing", out["context"])
        self.assertEqual(seen[0]["slugs"], ["ops/thing"])

    def test_no_pages_no_context(self):
        with patch.object(reflex, "_volunteer_via_http", lambda window: []), patch.object(
            reflex, "_query_via_http", lambda text: []
        ), patch.object(reflex, "_resolve_socket_path", lambda: None):
            self.assertIsNone(reflex.on_pre_llm_call(user_message="nothing here"))


class Tolerance(unittest.TestCase):
    """GBrain down, or no credentials: silent no-op, never an exception."""

    def test_http_error_returns_none(self):
        def boom(*a, **k):
            raise urllib.error.URLError("connection refused")

        with patch("urllib.request.urlopen", boom), patch.object(
            reflex, "_read_bearer_token", lambda: "t"
        ):
            self.assertIsNone(reflex._mcp_http_call("tools/call", {}))

    def test_missing_token_is_empty_not_an_error(self):
        with patch.dict(os.environ, {"GBRAIN_TOKEN": ""}, clear=False), patch.object(
            reflex, "_TOKEN_PATHS", ()
        ), patch.object(reflex, "_ENV_PATHS", ()):
            self.assertEqual(reflex._read_bearer_token(), "")

    def test_register_only_adds_the_pre_llm_hook(self):
        hooks: list[tuple] = []
        ctx = types.SimpleNamespace(register_hook=lambda *a: hooks.append(a))
        reflex.register(ctx)
        self.assertEqual([h[0] for h in hooks], ["pre_llm_call"])


if __name__ == "__main__":
    unittest.main()
