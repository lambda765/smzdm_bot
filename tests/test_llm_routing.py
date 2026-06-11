from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import patch

from smzdm_notice.llm import routing
from smzdm_notice.llm.routing import LLMRoutingError


class LlmRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        routing._clear_routing_state()

    def tearDown(self) -> None:
        routing._clear_routing_state()

    def test_agent_inherits_default_connection_and_model_with_request_override(self) -> None:
        with self._configured_routing(
            {
                "connections": {
                    "deepseek": {
                        "provider": "openai_compatible",
                        "label": "DeepSeek",
                        "base_url": "https://api.deepseek.com/v1",
                        "api_key_env": "LLM_DEEPSEEK_API_KEY",
                    }
                },
                "defaults": {
                    "connection": "deepseek",
                    "model_id": "deepseek-chat",
                    "timeout_seconds": 300,
                    "max_retries": 2,
                    "request": {
                        "response_format": {"type": "json_object"},
                        "extra_body": {"thinking": {"type": "disabled"}},
                    },
                },
                "agents": {
                    "filter": {
                        "request": {
                            "temperature": 0.3,
                            "extra_body": {"top_k": 1},
                        }
                    },
                    "arbiter": {"request": {"temperature": 0.0}},
                    "draft": {"request": {"temperature": 0.0}},
                },
            },
            {"LLM_DEEPSEEK_API_KEY": "key"},
        ):
            resolved = routing.resolve("filter")

        self.assertEqual(resolved.connection, "deepseek")
        self.assertEqual(resolved.model_id, "deepseek-chat")
        self.assertEqual(resolved.temperature, 0.3)
        self.assertEqual(resolved.extra_body["thinking"], {"type": "disabled"})
        self.assertEqual(resolved.extra_body["top_k"], 1)

    def test_agent_connection_and_model_override_only_affects_that_agent(self) -> None:
        with self._configured_routing(
            {
                "connections": {
                    "deepseek": {
                        "provider": "openai_compatible",
                        "label": "DeepSeek",
                        "base_url": "https://api.deepseek.com/v1",
                        "api_key_env": "LLM_DEEPSEEK_API_KEY",
                    },
                    "glm": {
                        "provider": "openai_compatible",
                        "label": "GLM",
                        "base_url": "https://open.bigmodel.cn/api/paas/v4",
                        "api_key_env": "LLM_GLM_API_KEY",
                    },
                },
                "defaults": {"connection": "deepseek", "model_id": "deepseek-chat"},
                "agents": {
                    "filter": {"request": {"temperature": 0.3}},
                    "arbiter": {"connection": "glm", "model_id": "glm-4-flash"},
                    "draft": {},
                },
            },
            {"LLM_DEEPSEEK_API_KEY": "deepseek-key", "LLM_GLM_API_KEY": "glm-key"},
        ):
            filter_config = routing.resolve("filter")
            arbiter_config = routing.resolve("arbiter")
            draft_config = routing.resolve("draft")

        self.assertEqual(filter_config.connection, "deepseek")
        self.assertEqual(arbiter_config.connection, "glm")
        self.assertEqual(arbiter_config.model_id, "glm-4-flash")
        self.assertEqual(draft_config.connection, "deepseek")
        self.assertEqual(draft_config.model_id, "deepseek-chat")

    def test_hot_switch_default_model_only_changes_inheriting_agents(self) -> None:
        with self._configured_routing(
            {
                "connections": {
                    "deepseek": {
                        "provider": "openai_compatible",
                        "label": "DeepSeek",
                        "base_url": "https://api.deepseek.com/v1",
                        "api_key_env": "LLM_DEEPSEEK_API_KEY",
                    }
                },
                "defaults": {"connection": "deepseek", "model_id": "deepseek-chat"},
                "agents": {
                    "filter": {"model_id": "deepseek-chat"},
                    "arbiter": {},
                    "draft": {},
                },
            },
            {"LLM_DEEPSEEK_API_KEY": "key"},
        ) as path:
            snapshot = routing.use_default_model("deepseek-reasoner")
            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(saved["defaults"]["model_id"], "deepseek-reasoner")
        self.assertEqual(snapshot.resolve("filter").model_id, "deepseek-chat")
        self.assertEqual(snapshot.resolve("draft").model_id, "deepseek-reasoner")

    def test_default_connection_model_update_affects_inheriting_agents(self) -> None:
        with self._configured_routing(
            {
                "connections": {
                    "deepseek": {
                        "provider": "openai_compatible",
                        "label": "DeepSeek",
                        "base_url": "https://api.deepseek.com/v1",
                        "api_key_env": "LLM_DEEPSEEK_API_KEY",
                    },
                    "glm": {
                        "provider": "openai_compatible",
                        "label": "GLM",
                        "base_url": "https://open.bigmodel.cn/api/paas/v4",
                        "api_key_env": "LLM_GLM_API_KEY",
                    },
                },
                "defaults": {"connection": "deepseek", "model_id": "deepseek-chat"},
                "agents": {"filter": {}, "arbiter": {}, "draft": {}},
            },
            {"LLM_DEEPSEEK_API_KEY": "deepseek-key", "LLM_GLM_API_KEY": "glm-key"},
        ) as path:
            snapshot = routing.use_default_connection_model("glm", "glm-4-flash")
            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(saved["defaults"]["connection"], "glm")
        self.assertEqual(saved["defaults"]["model_id"], "glm-4-flash")
        self.assertEqual(snapshot.resolve("filter").connection, "glm")
        self.assertEqual(snapshot.resolve("filter").model_id, "glm-4-flash")

    def test_default_temperature_update_writes_defaults_request(self) -> None:
        with self._configured_routing(
            {
                "connections": {
                    "deepseek": {
                        "provider": "openai_compatible",
                        "label": "DeepSeek",
                        "base_url": "https://api.deepseek.com/v1",
                        "api_key_env": "LLM_DEEPSEEK_API_KEY",
                    }
                },
                "defaults": {"connection": "deepseek", "model_id": "deepseek-chat"},
                "agents": {"filter": {}, "arbiter": {}, "draft": {}},
            },
            {"LLM_DEEPSEEK_API_KEY": "key"},
        ) as path:
            snapshot = routing.set_default_temperature(0.4)
            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(saved["defaults"]["request"]["temperature"], 0.4)
        self.assertEqual(snapshot.resolve("filter").temperature, 0.4)

    def test_reset_agent_removes_connection_and_model_override_but_keeps_request(self) -> None:
        with self._configured_routing(
            {
                "connections": {
                    "deepseek": {
                        "provider": "openai_compatible",
                        "label": "DeepSeek",
                        "base_url": "https://api.deepseek.com/v1",
                        "api_key_env": "LLM_DEEPSEEK_API_KEY",
                    },
                    "glm": {
                        "provider": "openai_compatible",
                        "label": "GLM",
                        "base_url": "https://open.bigmodel.cn/api/paas/v4",
                        "api_key_env": "LLM_GLM_API_KEY",
                    },
                },
                "defaults": {"connection": "deepseek", "model_id": "deepseek-chat"},
                "agents": {
                    "filter": {},
                    "arbiter": {
                        "connection": "glm",
                        "model_id": "glm-4-flash",
                        "request": {"temperature": 0.0},
                    },
                    "draft": {},
                },
            },
            {"LLM_DEEPSEEK_API_KEY": "deepseek-key", "LLM_GLM_API_KEY": "glm-key"},
        ):
            snapshot = routing.reset_agent("arbiter")

        resolved = snapshot.resolve("arbiter")
        self.assertEqual(resolved.connection, "deepseek")
        self.assertEqual(resolved.model_id, "deepseek-chat")
        self.assertEqual(resolved.temperature, 0.0)

    def test_missing_api_key_env_is_configuration_error(self) -> None:
        with self.assertRaises(LLMRoutingError), self._configured_routing(
            {
                "connections": {
                    "deepseek": {
                        "provider": "openai_compatible",
                        "label": "DeepSeek",
                        "base_url": "https://api.deepseek.com/v1",
                        "api_key_env": "LLM_DEEPSEEK_API_KEY",
                    }
                },
                "defaults": {"connection": "deepseek", "model_id": "deepseek-chat"},
                "agents": {"filter": {}, "arbiter": {}, "draft": {}},
            },
            {},
        ):
            pass

    def test_missing_draft_agent_inherits_defaults(self) -> None:
        with self._configured_routing(
            {
                "connections": {
                    "deepseek": {
                        "provider": "openai_compatible",
                        "label": "DeepSeek",
                        "base_url": "https://api.deepseek.com/v1",
                        "api_key_env": "LLM_DEEPSEEK_API_KEY",
                    }
                },
                "defaults": {"connection": "deepseek", "model_id": "deepseek-chat"},
                "agents": {
                    "filter": {"request": {"temperature": 0.3}},
                    "arbiter": {"request": {"temperature": 0.0}},
                },
            },
            {"LLM_DEEPSEEK_API_KEY": "key"},
        ):
            draft_config = routing.resolve("draft")

        self.assertEqual(draft_config.connection, "deepseek")
        self.assertEqual(draft_config.model_id, "deepseek-chat")

    def test_draft_override_with_missing_key_is_configuration_error(self) -> None:
        with self.assertRaises(LLMRoutingError), self._configured_routing(
            {
                "connections": {
                    "deepseek": {
                        "provider": "openai_compatible",
                        "label": "DeepSeek",
                        "base_url": "https://api.deepseek.com/v1",
                        "api_key_env": "LLM_DEEPSEEK_API_KEY",
                    },
                    "draft": {
                        "provider": "openai_compatible",
                        "label": "Draft",
                        "base_url": "https://draft.example.com/v1",
                        "api_key_env": "LLM_DRAFT_API_KEY",
                    },
                },
                "defaults": {"connection": "deepseek", "model_id": "deepseek-chat"},
                "agents": {"filter": {}, "arbiter": {}, "draft": {"connection": "draft"}},
            },
            {"LLM_DEEPSEEK_API_KEY": "key"},
        ):
            pass

    def test_null_request_field_in_agent_inherits_default(self) -> None:
        with self._configured_routing(
            {
                "connections": {
                    "deepseek": {
                        "provider": "openai_compatible",
                        "label": "DeepSeek",
                        "base_url": "https://api.deepseek.com/v1",
                        "api_key_env": "LLM_DEEPSEEK_API_KEY",
                    }
                },
                "defaults": {
                    "connection": "deepseek",
                    "model_id": "deepseek-chat",
                    "request": {
                        "response_format": {"type": "json_object"},
                        "extra_body": {"provider": "default"},
                    },
                },
                "agents": {
                    "filter": {"request": {"response_format": None, "extra_body": None}},
                    "arbiter": {},
                    "draft": {},
                },
            },
            {"LLM_DEEPSEEK_API_KEY": "key"},
        ):
            filter_config = routing.resolve("filter")

        self.assertEqual(filter_config.response_format, {"type": "json_object"})
        self.assertEqual(filter_config.extra_body, {"provider": "default"})

    def test_builtin_response_format_is_used_when_config_omits_it(self) -> None:
        with self._configured_routing(
            {
                "connections": {
                    "deepseek": {
                        "provider": "openai_compatible",
                        "label": "DeepSeek",
                        "base_url": "https://api.deepseek.com/v1",
                        "api_key_env": "LLM_DEEPSEEK_API_KEY",
                    }
                },
                "defaults": {"connection": "deepseek", "model_id": "deepseek-chat"},
                "agents": {"filter": {}, "arbiter": {}, "draft": {}},
            },
            {"LLM_DEEPSEEK_API_KEY": "key"},
        ):
            config = routing.resolve("filter")
            kwargs = routing.build_chat_completion_kwargs(config, [{"role": "user", "content": "ping"}])

        self.assertEqual(config.response_format, {"type": "json_object"})
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})

    def test_response_format_overrides_builtin_default_by_layer(self) -> None:
        with self._configured_routing(
            {
                "connections": {
                    "deepseek": {
                        "provider": "openai_compatible",
                        "label": "DeepSeek",
                        "base_url": "https://api.deepseek.com/v1",
                        "api_key_env": "LLM_DEEPSEEK_API_KEY",
                    }
                },
                "defaults": {
                    "connection": "deepseek",
                    "model_id": "deepseek-chat",
                    "request": {"response_format": {"type": "default_json"}},
                },
                "agents": {
                    "filter": {"request": {"response_format": {"type": "agent_json"}}},
                    "arbiter": {"request": {"response_format": None}},
                    "draft": {},
                },
            },
            {"LLM_DEEPSEEK_API_KEY": "key"},
        ):
            filter_config = routing.resolve("filter")
            arbiter_config = routing.resolve("arbiter")

        self.assertEqual(filter_config.response_format, {"type": "agent_json"})
        self.assertEqual(arbiter_config.response_format, {"type": "default_json"})

    def test_resolved_request_dicts_do_not_mutate_snapshot(self) -> None:
        with self._configured_routing(
            {
                "connections": {
                    "deepseek": {
                        "provider": "openai_compatible",
                        "label": "DeepSeek",
                        "base_url": "https://api.deepseek.com/v1",
                        "api_key_env": "LLM_DEEPSEEK_API_KEY",
                    }
                },
                "defaults": {
                    "connection": "deepseek",
                    "model_id": "deepseek-chat",
                    "request": {
                        "response_format": {"type": "json_object"},
                        "extra_body": {"thinking": {"type": "disabled"}},
                    },
                },
                "agents": {"filter": {}, "arbiter": {}, "draft": {}},
            },
            {"LLM_DEEPSEEK_API_KEY": "key"},
        ):
            snapshot = routing.get_snapshot()
            config = snapshot.resolve("filter")
            assert config.response_format is not None
            config.response_format["type"] = "text"
            config.extra_body["thinking"]["type"] = "enabled"
            resolved_again = snapshot.resolve("filter")

        self.assertEqual(resolved_again.response_format, {"type": "json_object"})
        self.assertEqual(resolved_again.extra_body, {"thinking": {"type": "disabled"}})

    def test_chat_completion_kwargs_copy_request_dicts(self) -> None:
        with self._configured_routing(
            {
                "connections": {
                    "deepseek": {
                        "provider": "openai_compatible",
                        "label": "DeepSeek",
                        "base_url": "https://api.deepseek.com/v1",
                        "api_key_env": "LLM_DEEPSEEK_API_KEY",
                    }
                },
                "defaults": {
                    "connection": "deepseek",
                    "model_id": "deepseek-chat",
                    "request": {
                        "response_format": {"type": "json_object"},
                        "extra_body": {"thinking": {"type": "disabled"}},
                    },
                },
                "agents": {"filter": {}, "arbiter": {}, "draft": {}},
            },
            {"LLM_DEEPSEEK_API_KEY": "key"},
        ):
            config = routing.resolve("filter")
            kwargs = routing.build_chat_completion_kwargs(config, [{"role": "user", "content": "ping"}])
            kwargs["response_format"]["type"] = "text"
            kwargs["extra_body"]["thinking"]["type"] = "enabled"

        self.assertEqual(config.response_format, {"type": "json_object"})
        self.assertEqual(config.extra_body, {"thinking": {"type": "disabled"}})

    def test_temperature_accepts_five_and_rejects_non_finite_values(self) -> None:
        with self._configured_routing(
            {
                "connections": {
                    "deepseek": {
                        "provider": "openai_compatible",
                        "label": "DeepSeek",
                        "base_url": "https://api.deepseek.com/v1",
                        "api_key_env": "LLM_DEEPSEEK_API_KEY",
                    }
                },
                "defaults": {
                    "connection": "deepseek",
                    "model_id": "deepseek-chat",
                    "request": {"temperature": 5.0},
                },
                "agents": {"filter": {}, "arbiter": {}, "draft": {}},
            },
            {"LLM_DEEPSEEK_API_KEY": "key"},
        ):
            self.assertEqual(routing.resolve("filter").temperature, 5.0)

        for value in (5.1, -0.1, math.nan, math.inf):
            with self.assertRaisesRegex(LLMRoutingError, "temperature 必须在 0 到 5 之间"):
                routing.validate_raw(
                    {
                        "connections": {
                            "deepseek": {
                                "provider": "openai_compatible",
                                "label": "DeepSeek",
                                "base_url": "https://api.deepseek.com/v1",
                                "api_key_env": "LLM_DEEPSEEK_API_KEY",
                            }
                        },
                        "defaults": {
                            "connection": "deepseek",
                            "model_id": "deepseek-chat",
                            "request": {"temperature": value},
                        },
                        "agents": {"filter": {}, "arbiter": {}, "draft": {}},
                    },
                    env={"LLM_DEEPSEEK_API_KEY": "key"},
                )

    def test_connection_timeout_and_retries_are_used_before_defaults(self) -> None:
        with self._configured_routing(
            {
                "connections": {
                    "slow": {
                        "provider": "openai_compatible",
                        "label": "Slow",
                        "base_url": "https://slow.example.com/v1",
                        "api_key_env": "LLM_SLOW_API_KEY",
                        "timeout_seconds": 600,
                        "max_retries": 5,
                    }
                },
                "defaults": {
                    "connection": "slow",
                    "model_id": "slow-model",
                    "timeout_seconds": 300,
                    "max_retries": 2,
                },
                "agents": {
                    "filter": {},
                    "arbiter": {"timeout_seconds": 60, "max_retries": 1},
                    "draft": {},
                },
            },
            {"LLM_SLOW_API_KEY": "key"},
        ):
            filter_config = routing.resolve("filter")
            arbiter_config = routing.resolve("arbiter")

        self.assertEqual(filter_config.timeout_seconds, 600.0)
        self.assertEqual(filter_config.max_retries, 5)
        self.assertEqual(arbiter_config.timeout_seconds, 60.0)
        self.assertEqual(arbiter_config.max_retries, 1)

    def test_model_card_state_does_not_expose_api_key_values(self) -> None:
        with self._configured_routing(
            {
                "connections": {
                    "deepseek": {
                        "provider": "openai_compatible",
                        "label": "DeepSeek",
                        "base_url": "https://api.deepseek.com/v1",
                        "api_key_env": "LLM_DEEPSEEK_API_KEY",
                    }
                },
                "defaults": {
                    "connection": "deepseek",
                    "model_id": "deepseek-chat",
                    "request": {"temperature": 0.3},
                },
                "agents": {"filter": {}, "arbiter": {}, "draft": {}},
            },
            {"LLM_DEEPSEEK_API_KEY": "secret-key"},
        ):
            state = routing.model_card_state()

        self.assertNotIn("secret-key", json.dumps(state, ensure_ascii=False))
        self.assertEqual(state["defaults"]["temperature"], 0.3)
        self.assertEqual(state["agents"][0]["connection"], "deepseek")
        self.assertTrue(state["connections"][0]["key_configured"])

    def test_status_and_card_state_share_inheritance_flags(self) -> None:
        with self._configured_routing(
            {
                "connections": {
                    "deepseek": {
                        "provider": "openai_compatible",
                        "label": "DeepSeek",
                        "base_url": "https://api.deepseek.com/v1",
                        "api_key_env": "LLM_DEEPSEEK_API_KEY",
                    }
                },
                "defaults": {"connection": "deepseek", "model_id": "deepseek-chat"},
                "agents": {
                    "filter": {"model_id": "deepseek-reasoner"},
                    "arbiter": {},
                    "draft": {},
                },
            },
            {"LLM_DEEPSEEK_API_KEY": "key"},
        ):
            status = routing.format_status()
            card_state = routing.model_card_state()

        filter_state = next(agent for agent in card_state["agents"] if agent["name"] == "filter")
        self.assertTrue(filter_state["inherits_connection"])
        self.assertFalse(filter_state["inherits_model"])
        self.assertIn("- filter: deepseek/deepseek-reasoner", status)
        self.assertIn("继承 connection", status)

    def test_write_json_atomic_uses_unique_temp_filename(self) -> None:
        with self._configured_routing(
            {
                "connections": {
                    "deepseek": {
                        "provider": "openai_compatible",
                        "label": "DeepSeek",
                        "base_url": "https://api.deepseek.com/v1",
                        "api_key_env": "LLM_DEEPSEEK_API_KEY",
                    }
                },
                "defaults": {"connection": "deepseek", "model_id": "deepseek-chat"},
                "agents": {"filter": {}, "arbiter": {}, "draft": {}},
            },
            {"LLM_DEEPSEEK_API_KEY": "key"},
        ) as path:
            routing.use_default_model("deepseek-reasoner")
            fixed_tmp = path.with_name(path.name + ".tmp")

            self.assertFalse(fixed_tmp.exists())

    def test_concurrent_same_process_updates_do_not_lose_changes(self) -> None:
        with self._configured_routing(
            {
                "connections": {
                    "deepseek": {
                        "provider": "openai_compatible",
                        "label": "DeepSeek",
                        "base_url": "https://api.deepseek.com/v1",
                        "api_key_env": "LLM_DEEPSEEK_API_KEY",
                    }
                },
                "defaults": {"connection": "deepseek", "model_id": "deepseek-chat"},
                "agents": {"filter": {}, "arbiter": {}, "draft": {}},
            },
            {"LLM_DEEPSEEK_API_KEY": "key"},
        ) as path:
            original_write = routing._write_json_atomic
            first_write_started = Event()
            release_first_write = Event()
            writes = {"count": 0}

            def delayed_first_write(write_path, raw):
                writes["count"] += 1
                if writes["count"] == 1:
                    first_write_started.set()
                    self.assertTrue(release_first_write.wait(timeout=2))
                original_write(write_path, raw)

            with (
                patch("smzdm_notice.llm.routing._write_json_atomic", side_effect=delayed_first_write),
                ThreadPoolExecutor(max_workers=2) as executor,
            ):
                future_filter = executor.submit(routing.use_agent_model, "filter", "model-a")
                self.assertTrue(first_write_started.wait(timeout=2))
                future_arbiter = executor.submit(routing.use_agent_model, "arbiter", "model-b")
                release_first_write.set()
                future_filter.result(timeout=2)
                future_arbiter.result(timeout=2)

            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(saved["agents"]["filter"]["model_id"], "model-a")
        self.assertEqual(saved["agents"]["arbiter"]["model_id"], "model-b")

    def test_force_initialize_does_not_get_overwritten_by_in_flight_update(self) -> None:
        initial = {
            "connections": {
                "deepseek": {
                    "provider": "openai_compatible",
                    "label": "DeepSeek",
                    "base_url": "https://api.deepseek.com/v1",
                    "api_key_env": "LLM_DEEPSEEK_API_KEY",
                }
            },
            "defaults": {"connection": "deepseek", "model_id": "deepseek-chat"},
            "agents": {"filter": {}, "arbiter": {}, "draft": {}},
        }
        external = {
            **initial,
            "agents": {"filter": {"model_id": "external-model"}, "arbiter": {}, "draft": {}},
        }
        with self._configured_routing(initial, {"LLM_DEEPSEEK_API_KEY": "key"}) as path:
            original_write = routing._write_json_atomic
            write_finished = Event()
            release_update = Event()

            def delayed_write(write_path, raw):
                original_write(write_path, raw)
                write_finished.set()
                self.assertTrue(release_update.wait(timeout=2))

            with (
                patch("smzdm_notice.llm.routing._write_json_atomic", side_effect=delayed_write),
                ThreadPoolExecutor(max_workers=2) as executor,
            ):
                update_future = executor.submit(routing.use_agent_model, "filter", "model-a")
                self.assertTrue(write_finished.wait(timeout=2))
                path.write_text(json.dumps(external), encoding="utf-8")
                initialize_future = executor.submit(routing.initialize, True)
                release_update.set()
                update_future.result(timeout=2)
                initialize_future.result(timeout=2)
                current = routing.get_snapshot()

        self.assertEqual(current.resolve("filter").model_id, "external-model")

    def test_missing_llm_models_file_does_not_fallback_to_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "llm_models.json"
            with (
                patch("smzdm_notice.llm.routing.config.LLM_MODELS_FILE", str(missing)),
                patch.dict(os.environ, {"LLM_API_KEY": "legacy-key"}, clear=True),
                self.assertRaisesRegex(LLMRoutingError, "llm_models.json missing"),
            ):
                routing.initialize(force=True)

    def _configured_routing(self, raw: dict, env: dict[str, str]):
        return _RoutingContext(raw, env)


class _RoutingContext:
    def __init__(self, raw: dict, env: dict[str, str]) -> None:
        self.raw = raw
        self.env = env
        self.tmp: tempfile.TemporaryDirectory[str] | None = None
        self.path: Path | None = None
        self.patchers = []

    def __enter__(self) -> Path:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "llm_models.json"
        self.path.write_text(json.dumps(self.raw), encoding="utf-8")
        self.patchers = [
            patch("smzdm_notice.llm.routing.config.LLM_MODELS_FILE", str(self.path)),
            patch.dict(os.environ, self.env, clear=True),
        ]
        try:
            for patcher in self.patchers:
                patcher.__enter__()
            routing.initialize(force=True)
        except Exception:
            for patcher in reversed(self.patchers):
                patcher.__exit__(None, None, None)
            routing._clear_routing_state()
            if self.tmp:
                self.tmp.cleanup()
            raise
        return self.path

    def __exit__(self, exc_type, exc, tb) -> None:
        for patcher in reversed(self.patchers):
            patcher.__exit__(exc_type, exc, tb)
        routing._clear_routing_state()
        if self.tmp:
            self.tmp.cleanup()
