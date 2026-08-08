from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from openai import BadRequestError

from smzdm_notice.llm.routing import ResolvedLLMConfig
from smzdm_notice.preferences.builder import (
    _call_llm_for_draft,
    _deal_action_message,
    _draft_with_llm,
    _parse_llm_draft_content,
    build_arbitration_candidate_draft_outcome,
    build_deal_action_draft,
    build_deal_action_draft_outcome,
    build_memory_rule_draft_outcome,
    build_message_draft,
    build_message_draft_outcome,
    build_rebase_draft_outcome,
    build_revision_draft,
    build_revision_draft_outcome,
)
from smzdm_notice.preferences.models import ConfigDraft
from smzdm_notice.preferences.prompts import draft_rules_prompt, file_context_block
from smzdm_notice.preferences.store import DraftStore
from smzdm_notice.preferences.validation import markdown_outline, validate_draft_data


def _openai_response(status_code: int) -> httpx.Response:
    request = httpx.Request("POST", "https://llm.example.com/v1/chat/completions")
    return httpx.Response(status_code, request=request)


def _draft_llm_config(api_key: str = "draft-key") -> ResolvedLLMConfig:
    return ResolvedLLMConfig(
        agent="draft",
        connection="draft",
        connection_label="Draft",
        provider="openai_compatible",
        base_url="https://draft.example.com",
        api_key_env="LLM_DRAFT_TEST_API_KEY",
        api_key=api_key,
        model_id="draft-model",
        timeout_seconds=300.0,
        max_retries=2,
        temperature=0.0,
        response_format={"type": "json_object"},
        extra_body={},
    )


def _stream_chunk(
    *,
    content: str | None = None,
    reasoning_content: str | None = None,
    reasoning_details: list[dict] | None = None,
    with_choices: bool = True,
):
    if not with_choices:
        return SimpleNamespace(choices=[])
    delta = SimpleNamespace(
        content=content,
        reasoning_content=reasoning_content,
        reasoning_details=reasoning_details,
    )
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class PreferenceEditorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "preference.md").write_text("# preference\n", encoding="utf-8")
        (self.root / "inventory.md").write_text("# inventory\n", encoding="utf-8")
        self.store = DraftStore(
            draft_file=self.root / "drafts.json",
            backup_dir=self.root / "backups",
            audit_file=self.root / "audit.jsonl",
            root=self.root,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_preference_template_uses_quality_signal_language(self) -> None:
        template = (Path(__file__).resolve().parents[1] / "preference.md.template").read_text(encoding="utf-8")

        self.assertIn("质量信号评估", template)
        self.assertIn("开放推荐评估标准与加分项", template)
        self.assertIn("专项品类评估要求", template)
        self.assertNotIn("质量门槛（硬性要求，不满足则不推荐）", template)
        self.assertNotIn("可填写", template)
        self.assertNotIn("示例数字默认", template)
        self.assertNotIn("开放推荐通常", template)
        self.assertNotIn("专项品类可以", template)

    def test_draft_prompt_treats_numbers_as_reference_by_default(self) -> None:
        prompt = draft_rules_prompt()

        self.assertIn("质量信号", prompt)
        self.assertIn("专项品类评估要求", prompt)
        self.assertIn("数字阈值默认写成参考线或质量信号", prompt)
        self.assertIn('"必须"', prompt)
        self.assertIn("由你负责语义去重", prompt)
        self.assertIn("完整标题路径", prompt)

    def test_file_context_keeps_content_beyond_4000_chars(self) -> None:
        tail_rule = "- **水果类商品**：已有规则不得重复新增"
        (self.root / "preference.md").write_text("# preference\n" + ("x" * 4500) + "\n" + tail_rule, encoding="utf-8")

        context = file_context_block(self.root)

        self.assertIn(tail_rule, context)
        self.assertNotIn("内容过长已截断", context)

    def test_dynamic_outline_does_not_require_fixed_section_names(self) -> None:
        outline = markdown_outline("# 偏好\n\n## 任意自定义章节\n- 规则\n")

        self.assertIn("H2: 任意自定义章节", outline)

    def test_append_remains_valid_when_it_is_not_duplicate(self) -> None:
        original = "# preference\n\n## 当前规则\n- 已有规则\n"

        result = validate_draft_data(
            {"target_file": "preference.md", "edit_mode": "append", "append_text": "## 新章节\n- 新规则"},
            original,
        )

        self.assertTrue(result.ok, result.error)

    def test_append_preserves_existing_trailing_whitespace_and_blank_lines(self) -> None:
        original = "# preference\n\n## 当前规则\n- 已有规则  \n\n\n\n"

        result = validate_draft_data(
            {"target_file": "preference.md", "edit_mode": "append", "append_text": "- 新规则"},
            original,
        )

        self.assertTrue(result.ok, result.error)
        self.assertTrue(result.new_content.startswith(original))
        self.assertEqual(result.new_content, original + "- 新规则\n")

    def test_exact_duplicate_rule_is_left_to_llm_semantic_validation(self) -> None:
        original = "# preference\n\n## 当前规则\n- 这是一条足够长的已有偏好规则\n"

        result = validate_draft_data(
            {
                "target_file": "preference.md",
                "edit_mode": "append",
                "append_text": "- 这是一条足够长的已有偏好规则",
            },
            original,
        )

        self.assertTrue(result.ok, result.error)

    def test_exact_rule_text_is_allowed_under_different_sections(self) -> None:
        original = "# preference\n\n## 食品\n- 需要明确的历史低价信号\n"

        result = validate_draft_data(
            {
                "target_file": "preference.md",
                "edit_mode": "append",
                "append_text": "## 日用\n- 需要明确的历史低价信号",
            },
            original,
        )

        self.assertTrue(result.ok, result.error)

    def test_same_child_heading_is_allowed_under_different_parents(self) -> None:
        original = "# 偏好\n\n## 食品\n\n### 品牌\n- 食品品牌规则\n"

        result = validate_draft_data(
            {
                "target_file": "preference.md",
                "edit_mode": "append",
                "append_text": "## 日用\n\n### 品牌\n- 日用品牌规则",
            },
            original,
        )

        self.assertTrue(result.ok, result.error)

    def test_new_duplicate_heading_path_is_rejected(self) -> None:
        original = "# 偏好\n\n## 食品\n\n### 品牌\n- 食品品牌规则\n"

        result = validate_draft_data(
            {
                "target_file": "preference.md",
                "edit_mode": "append",
                "append_text": "## 食品\n\n### 品牌\n- 另一条品牌规则",
            },
            original,
        )

        self.assertFalse(result.ok)
        self.assertIn("同一层级路径", result.error)

    def test_existing_markdown_issues_do_not_block_unrelated_body_change(self) -> None:
        repeated = "- 这是一条足够长且已经重复的历史偏好规则"
        original = f"# 偏好\n\n### 跳级章节\n{repeated}\n{repeated}\n\n# 历史附录\n"

        result = validate_draft_data(
            {
                "target_file": "preference.md",
                "edit_mode": "append",
                "append_text": "- 这是一条足够长且全新的普通偏好规则",
            },
            original,
        )

        self.assertTrue(result.ok, result.error)

    def test_new_h1_and_heading_jump_are_rejected(self) -> None:
        original = "# 偏好\n\n## 食品\n- 食品规则\n"
        for append_text, expected in (
            ("# 另一个一级标题\n- 新规则", "多个一级标题"),
            ("#### 跳级标题\n- 新规则", "标题层级发生跳跃"),
        ):
            with self.subTest(append_text=append_text):
                result = validate_draft_data(
                    {
                        "target_file": "preference.md",
                        "edit_mode": "append",
                        "append_text": append_text,
                    },
                    original,
                )
                self.assertFalse(result.ok)
                self.assertIn(expected, result.error)

    def test_first_heading_cannot_start_at_a_deep_level(self) -> None:
        result = validate_draft_data(
            {
                "target_file": "preference.md",
                "edit_mode": "append",
                "append_text": "### 跳级标题\n- 新规则",
            },
            "- 尚无标题的旧规则\n",
        )

        self.assertFalse(result.ok)
        self.assertIn("标题层级发生跳跃", result.error)

    def test_message_draft_outcome_supports_noop(self) -> None:
        with patch(
            "smzdm_notice.preferences.builder._draft_with_llm",
            return_value={"target_file": "preference.md", "edit_mode": "noop", "summary": "已有规则已覆盖"},
        ):
            outcome = build_message_draft_outcome("重复要求", self.store)

        self.assertEqual(outcome.status, "noop")
        self.assertIsNone(outcome.draft)

    def test_all_specialized_draft_builders_preserve_noop_outcome(self) -> None:
        original = ConfigDraft("original", "preference.md", "t", "s", "- 待定规则", "test")
        noop = {"target_file": "preference.md", "edit_mode": "noop", "summary": "当前规则已覆盖"}

        with patch("smzdm_notice.preferences.builder._draft_with_llm", return_value=noop):
            deal = build_deal_action_draft_outcome("deal_follow", {"item_title": "抽纸"}, self.store)
            arbitration = build_arbitration_candidate_draft_outcome(
                {"rule": "抽纸可以关注", "reason": "长期偏好"},
                self.store,
            )
            memory = build_memory_rule_draft_outcome(
                {"rule": "抽纸可以关注", "reason": "反馈稳定"},
                "分析摘要",
                self.store,
            )
        with patch("smzdm_notice.preferences.builder._revision_with_llm", return_value=noop):
            revision = build_revision_draft_outcome("保持原样", original, self.store)

        for outcome in (deal, arbitration, memory, revision):
            self.assertEqual(outcome.status, "noop")
            self.assertIsNone(outcome.draft)
            self.assertIn("覆盖", outcome.message)
        self.assertEqual(arbitration.message, "当前 preference.md 已覆盖候选规则，无需修改。")

    def test_arbitration_and_memory_retry_when_model_targets_inventory(self) -> None:
        wrong = {
            "target_file": "inventory.md",
            "edit_mode": "append",
            "append_text": "- 错误目标",
        }
        correct = {
            "target_file": "preference.md",
            "edit_mode": "append",
            "append_text": "- 正确偏好规则",
        }
        for builder in (
            lambda: build_arbitration_candidate_draft_outcome(
                {"rule": "正确偏好规则", "reason": "长期偏好"},
                self.store,
            ),
            lambda: build_memory_rule_draft_outcome(
                {"rule": "正确偏好规则", "reason": "反馈稳定"},
                "分析摘要",
                self.store,
            ),
        ):
            with self.subTest(builder=builder):
                with patch(
                    "smzdm_notice.preferences.builder._draft_with_llm",
                    side_effect=[wrong, correct],
                ) as llm:
                    outcome = builder()

                self.assertEqual(outcome.status, "draft")
                self.assertEqual(outcome.draft.target_file, "preference.md")
                self.assertEqual(llm.call_count, 2)
                self.store.cancel(outcome.draft.draft_id, reason="test_cleanup")

    def test_arbitration_rejects_inventory_target_after_retry(self) -> None:
        wrong = {
            "target_file": "inventory.md",
            "edit_mode": "append",
            "append_text": "- 错误目标",
        }
        with patch("smzdm_notice.preferences.builder._draft_with_llm", return_value=wrong) as llm:
            outcome = build_arbitration_candidate_draft_outcome(
                {"rule": "偏好规则", "reason": "长期偏好"},
                self.store,
            )

        self.assertEqual(outcome.status, "rejected")
        self.assertIn("只能修改 preference.md", outcome.message)
        self.assertEqual(llm.call_count, 2)

    def test_rebase_builds_new_pending_draft_against_latest_file(self) -> None:
        (self.root / "preference.md").write_text("# preference\n\n- 当前规则\n", encoding="utf-8")
        original = ConfigDraft(
            "old",
            "preference.md",
            "t",
            "s",
            "- 旧预览",
            "test",
            edit_mode="replace",
            search_text="- 已消失规则",
            replace_text="- 旧预览",
            revision_history=[{"role": "user", "content": "请调整规则"}],
            metadata={"intent": "保留原意图"},
        )
        self.store.create(original)
        refreshed = {
            "target_file": "preference.md",
            "edit_mode": "replace",
            "search_text": "- 当前规则",
            "replace_text": "- 新预览",
            "summary": "重新定位",
        }

        with patch("smzdm_notice.preferences.builder._revision_with_llm", return_value=refreshed) as llm:
            outcome = build_rebase_draft_outcome(original, self.store)

        self.assertEqual(outcome.status, "draft")
        self.assertEqual(outcome.draft.status, "pending")
        self.assertEqual(outcome.draft.metadata["supersedes_draft_id"], "old")
        self.assertEqual(outcome.draft.metadata["rebase_reason"], "apply_conflict")
        prompt_history = llm.call_args.args[0]
        self.assertIn("原始 metadata", prompt_history[-1]["content"])

    def test_rebase_noop_does_not_cancel_original(self) -> None:
        original = ConfigDraft("old", "preference.md", "t", "s", "- 旧预览", "test")
        self.store.create(original)
        noop = {"target_file": "preference.md", "edit_mode": "noop", "summary": "当前规则已经覆盖"}

        with patch("smzdm_notice.preferences.builder._revision_with_llm", return_value=noop):
            outcome = build_rebase_draft_outcome(original, self.store)

        self.assertEqual(outcome.status, "noop")
        self.assertEqual(self.store.get(original.draft_id).status, "pending")

    def test_message_draft_uses_llm_data(self) -> None:
        with patch(
            "smzdm_notice.preferences.builder._draft_with_llm",
            return_value={
                "target_file": "inventory.md",
                "title": "更新库存",
                "summary": "记录抽纸库存",
                "append_text": "- 抽纸还剩 3 包",
            },
        ):
            draft = build_message_draft("抽纸还剩 3 包", self.store)

        self.assertIsNotNone(draft)
        self.assertEqual(draft.target_file, "inventory.md")
        self.assertIn("抽纸还剩 3 包", draft.append_text)

    def test_message_draft_returns_none_without_llm_fallback(self) -> None:
        with patch("smzdm_notice.preferences.builder.resolve", return_value=_draft_llm_config(api_key="")):
            draft = build_message_draft("拉黑坚果", self.store)

        self.assertIsNone(draft)

    def test_message_draft_llm_uses_draft_model(self) -> None:
        content = (
            '{"target_file":"preference.md","title":"拉黑坚果","summary":"新增排除规则","append_text":"- 不再推荐坚果"}'
        )
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
        create_kwargs = {}

        def create_completion(**kwargs):
            create_kwargs.update(kwargs)
            return response

        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_completion)))

        with (
            patch(
                "smzdm_notice.preferences.builder.resolve",
                return_value=replace(_draft_llm_config(), temperature=0.2, extra_body={"do_sample": False}),
            ),
            patch("smzdm_notice.preferences.builder.get_client_for_config", return_value=fake_client) as get_client,
        ):
            data = _draft_with_llm("拉黑坚果")

        self.assertEqual(data["target_file"], "preference.md")
        get_client.assert_called_once()
        self.assertEqual(create_kwargs["model"], "draft-model")
        self.assertEqual(create_kwargs["temperature"], 0.2)
        self.assertEqual(create_kwargs["response_format"], {"type": "json_object"})
        self.assertEqual(create_kwargs["extra_body"], {"do_sample": False})
        self.assertNotIn("stream", create_kwargs)

    def test_streaming_draft_prefers_reasoning_and_keeps_complete_json(self) -> None:
        content = '{"target_file":"preference.md","append_text":"- 关注抽纸"}'
        chunks = [
            _stream_chunk(with_choices=False),
            _stream_chunk(reasoning_content="先检查"),
            _stream_chunk(reasoning_content="现有偏好"),
            _stream_chunk(content=content[:20]),
            _stream_chunk(content=content[20:]),
        ]
        captured = {}

        def create_completion(**kwargs):
            captured.update(kwargs)
            return iter(chunks)

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_completion))
        )
        events = []
        with (
            patch("smzdm_notice.preferences.builder.resolve", return_value=_draft_llm_config()),
            patch("smzdm_notice.preferences.builder.get_client_for_config", return_value=fake_client),
        ):
            data = _call_llm_for_draft(
                [{"role": "user", "content": "关注抽纸"}],
                stream_callback=events.append,
            )

        self.assertTrue(captured["stream"])
        self.assertEqual(data["append_text"], "- 关注抽纸")
        self.assertEqual("".join(event.text for event in events if event.kind == "reasoning_delta"), "先检查现有偏好")
        self.assertEqual("".join(event.text for event in events if event.kind == "content_delta"), content)

    def test_streaming_draft_uses_content_when_reasoning_is_missing(self) -> None:
        content = '{"target_file":"inventory.md","append_text":"- 抽纸还有 3 包"}'
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_kwargs: iter(
                        [_stream_chunk(content=content[:18]), _stream_chunk(content=content[18:])]
                    )
                )
            )
        )
        events = []
        with (
            patch("smzdm_notice.preferences.builder.resolve", return_value=_draft_llm_config()),
            patch("smzdm_notice.preferences.builder.get_client_for_config", return_value=fake_client),
        ):
            data = _call_llm_for_draft(
                [{"role": "user", "content": "更新库存"}],
                stream_callback=events.append,
            )

        self.assertEqual(data["target_file"], "inventory.md")
        self.assertFalse(any(event.kind == "reasoning_delta" for event in events))
        self.assertEqual("".join(event.text for event in events if event.kind == "content_delta"), content)

    def test_streaming_draft_normalizes_cumulative_reasoning_details(self) -> None:
        content = '{"target_file":"preference.md","append_text":"- 关注蓝莓"}'
        chunks = [
            _stream_chunk(reasoning_details=[{"text": "检查"}]),
            _stream_chunk(reasoning_details=[{"text": "检查偏好"}]),
            _stream_chunk(content=content),
        ]
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_kwargs: iter(chunks)))
        )
        events = []
        with (
            patch("smzdm_notice.preferences.builder.resolve", return_value=_draft_llm_config()),
            patch("smzdm_notice.preferences.builder.get_client_for_config", return_value=fake_client),
        ):
            data = _call_llm_for_draft(
                [{"role": "user", "content": "关注蓝莓"}],
                stream_callback=events.append,
            )

        self.assertEqual(data["append_text"], "- 关注蓝莓")
        self.assertEqual("".join(event.text for event in events if event.kind == "reasoning_delta"), "检查偏好")

    def test_streaming_draft_extracts_split_embedded_think_tags(self) -> None:
        content = '{"target_file":"preference.md","append_text":"- 关注牛奶"}'
        chunks = [
            _stream_chunk(content=" \n<thi"),
            _stream_chunk(content="nk>检查现有"),
            _stream_chunk(content="规则</thi"),
            _stream_chunk(content="nk>\n" + content),
        ]
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_kwargs: iter(chunks)))
        )
        events = []
        with (
            patch("smzdm_notice.preferences.builder.resolve", return_value=_draft_llm_config()),
            patch("smzdm_notice.preferences.builder.get_client_for_config", return_value=fake_client),
        ):
            data = _call_llm_for_draft(
                [{"role": "user", "content": "关注牛奶"}],
                stream_callback=events.append,
            )

        self.assertEqual(data["append_text"], "- 关注牛奶")
        self.assertEqual("".join(event.text for event in events if event.kind == "reasoning_delta"), "检查现有规则")
        self.assertEqual("".join(event.text for event in events if event.kind == "content_delta"), "\n" + content)

    def test_streaming_retry_emits_second_attempt_and_returns_corrected_draft(self) -> None:
        responses = iter(
            [
                [_stream_chunk(content='{"target_file":"preference.md","append_text":"# duplicate root"}')],
                [_stream_chunk(content='{"target_file":"preference.md","append_text":"- 关注纸品"}')],
            ]
        )
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_kwargs: iter(next(responses)))
            )
        )
        events = []
        with (
            patch("smzdm_notice.preferences.builder.resolve", return_value=_draft_llm_config()),
            patch("smzdm_notice.preferences.builder.get_client_for_config", return_value=fake_client),
        ):
            outcome = build_message_draft_outcome(
                "关注纸品",
                self.store,
                stream_callback=events.append,
            )

        self.assertEqual(outcome.status, "draft")
        self.assertEqual(outcome.draft.append_text, "- 关注纸品")
        attempts = [event for event in events if event.kind == "attempt_start"]
        self.assertEqual([event.attempt for event in attempts], [1, 2])
        self.assertIn("正在修正生成结果", attempts[-1].text)

    def test_streaming_callback_failure_does_not_abort_draft_generation(self) -> None:
        content = '{"target_file":"preference.md","append_text":"- 关注湿巾"}'
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_kwargs: iter([_stream_chunk(content=content)]))
            )
        )

        def failing_callback(_event) -> None:
            raise RuntimeError("card update failed")

        with (
            patch("smzdm_notice.preferences.builder.resolve", return_value=_draft_llm_config()),
            patch("smzdm_notice.preferences.builder.get_client_for_config", return_value=fake_client),
        ):
            data = _call_llm_for_draft(
                [{"role": "user", "content": "关注湿巾"}],
                stream_callback=failing_callback,
            )

        self.assertEqual(data["append_text"], "- 关注湿巾")

    def test_streaming_creation_failure_falls_back_to_non_streaming(self) -> None:
        content = '{"target_file":"preference.md","append_text":"- 关注湿巾"}'
        calls = []

        def create_completion(**kwargs):
            calls.append(kwargs)
            if kwargs.get("stream"):
                raise BadRequestError("stream unsupported", response=_openai_response(400), body=None)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_completion))
        )
        events = []
        with (
            patch("smzdm_notice.preferences.builder.resolve", return_value=_draft_llm_config()),
            patch("smzdm_notice.preferences.builder.get_client_for_config", return_value=fake_client),
        ):
            data = _call_llm_for_draft(
                [{"role": "user", "content": "关注湿巾"}],
                stream_callback=events.append,
            )

        self.assertEqual(data["append_text"], "- 关注湿巾")
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0]["stream"])
        self.assertNotIn("stream", calls[1])
        self.assertEqual([event.kind for event in events], ["attempt_start"])

    def test_streaming_iteration_failure_does_not_start_non_streaming_request(self) -> None:
        calls = []

        def failing_stream():
            raise RuntimeError("stream interrupted")
            yield

        def create_completion(**kwargs):
            calls.append(kwargs)
            return failing_stream()

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_completion))
        )
        with (
            patch("smzdm_notice.preferences.builder.resolve", return_value=_draft_llm_config()),
            patch("smzdm_notice.preferences.builder.get_client_for_config", return_value=fake_client),
        ):
            data = _call_llm_for_draft(
                [{"role": "user", "content": "关注湿巾"}],
                stream_callback=lambda _event: None,
            )

        self.assertIsNone(data)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["stream"])

    def test_unrelated_stream_creation_error_does_not_fall_back(self) -> None:
        calls = []

        def create_completion(**kwargs):
            calls.append(kwargs)
            raise BadRequestError("invalid model input", response=_openai_response(400), body=None)

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_completion))
        )
        with (
            patch("smzdm_notice.preferences.builder.resolve", return_value=_draft_llm_config()),
            patch("smzdm_notice.preferences.builder.get_client_for_config", return_value=fake_client),
        ):
            data = _call_llm_for_draft(
                [{"role": "user", "content": "关注湿巾"}],
                stream_callback=lambda _event: None,
            )

        self.assertIsNone(data)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["stream"])

    def test_call_llm_for_draft_returns_none_on_sdk_error(self) -> None:
        class FailingCompletions:
            # 为匹配 OpenAI SDK create 签名，这里保留关键字参数；测试替身只需抛出固定异常。
            def create(self, **_kwargs):
                raise BadRequestError("bad request", response=_openai_response(400), body=None)

        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FailingCompletions()))
        with (
            patch("smzdm_notice.preferences.builder.resolve", return_value=_draft_llm_config()),
            patch("smzdm_notice.preferences.builder.get_client_for_config", return_value=fake_client),
        ):
            data = _call_llm_for_draft([{"role": "user", "content": "改配置"}])

        self.assertIsNone(data)

    def test_message_draft_rejects_invalid_llm_data(self) -> None:
        with patch(
            "smzdm_notice.preferences.builder._draft_with_llm",
            return_value={
                "target_file": "main.py",
                "title": "bad",
                "summary": "bad",
                "append_text": "- bad",
            },
        ):
            self.assertIsNone(build_message_draft("改配置", self.store))
        with patch(
            "smzdm_notice.preferences.builder._draft_with_llm",
            return_value={
                "target_file": "preference.md",
                "title": "bad",
                "summary": "bad",
                "append_text": "",
            },
        ):
            self.assertIsNone(build_message_draft("改配置", self.store))

    def test_parse_llm_draft_json_variants(self) -> None:
        plain = '{"target_file":"preference.md","title":"t","summary":"s","append_text":"- 关注婴儿车"}'
        fenced = '```json\n{"target_file":"inventory.md","title":"t","summary":"s","append_text":"- 抽纸 3 包"}\n```'
        wrapped = (
            '好的，配置如下：{"target_file":"preference.md","title":"t","summary":"s","append_text":"- 不再推荐坚果"}'
        )

        self.assertEqual(_parse_llm_draft_content(plain)["target_file"], "preference.md")
        self.assertEqual(_parse_llm_draft_content(fenced)["target_file"], "inventory.md")
        self.assertIn("坚果", _parse_llm_draft_content(wrapped)["append_text"])

    def test_parse_llm_draft_rejects_empty_or_invalid_json(self) -> None:
        with self.assertRaises(ValueError):
            _parse_llm_draft_content("")
        with self.assertRaises(ValueError):
            _parse_llm_draft_content("没有 JSON")

    def test_apply_draft_writes_backup_and_audit(self) -> None:
        draft = ConfigDraft(
            draft_id="d1",
            target_file="preference.md",
            title="新增关注",
            summary="关注婴儿车",
            append_text="- 关注婴儿车",
            source="test",
        )
        self.store.create(draft)

        outcome = self.store.apply("d1", operator="u1")

        self.assertTrue(outcome.ok, outcome.message)
        self.assertEqual(outcome.status, "applied")
        content = (self.root / "preference.md").read_text(encoding="utf-8")
        self.assertIn("关注婴儿车", content)
        self.assertNotIn("机器人确认修改", content)
        self.assertNotIn("来源：", content)
        self.assertEqual(len(list((self.root / "backups").glob("preference.md.*.bak"))), 1)
        audit = (self.root / "audit.jsonl").read_text(encoding="utf-8")
        self.assertIn('"action": "applied"', audit)
        self.assertIn('"source": "test"', audit)
        self.assertIn('"operator": "u1"', audit)
        self.assertIn('"draft_id": "d1"', audit)
        self.assertIn('"backup":', audit)

    def test_duplicate_append_is_idempotent_against_current_content(self) -> None:
        first = ConfigDraft("d1", "preference.md", "t", "s", "- 不再推荐卷纸", "test")
        second = ConfigDraft("d2", "preference.md", "t", "s", "- 不再推荐卷纸", "test")
        self.store.create(first)
        self.store.create(second)

        self.assertTrue(self.store.apply("d1").ok)
        outcome = self.store.apply("d2")

        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.status, "already_applied")
        self.assertEqual(self.store.get("d2").status, "applied")
        self.assertEqual(self.store.get("d2").metadata["apply_result"], "content_already_present")
        content = (self.root / "preference.md").read_text(encoding="utf-8")
        self.assertEqual(content.count("- 不再推荐卷纸"), 1)
        self.assertEqual(len(list((self.root / "backups").glob("preference.md.*.bak"))), 1)

    def test_append_can_be_reapplied_after_same_block_is_removed(self) -> None:
        first = ConfigDraft("d1", "preference.md", "t", "s", "- 关注蓝莓", "test")
        self.store.create(first)
        self.assertEqual(self.store.apply("d1").status, "applied")

        target = self.root / "preference.md"
        target.write_text(target.read_text(encoding="utf-8").replace("- 关注蓝莓\n", ""), encoding="utf-8")
        second = ConfigDraft("d2", "preference.md", "t", "s", "- 关注蓝莓", "test")
        self.store.create(second)

        outcome = self.store.apply("d2")

        self.assertEqual(outcome.status, "applied")
        self.assertEqual(target.read_text(encoding="utf-8").count("- 关注蓝莓"), 1)
        self.assertEqual(len(list((self.root / "backups").glob("preference.md.*.bak"))), 2)

    def test_append_idempotency_requires_complete_line_match(self) -> None:
        target = self.root / "preference.md"
        target.write_text("# preference\n\n- 不再推荐卷纸增强版\n", encoding="utf-8")
        draft = ConfigDraft("d1", "preference.md", "t", "s", "- 不再推荐卷纸", "test")
        self.store.create(draft)

        outcome = self.store.apply("d1")

        self.assertEqual(outcome.status, "applied")
        self.assertIn("- 不再推荐卷纸增强版\n", target.read_text(encoding="utf-8"))
        self.assertIn("- 不再推荐卷纸\n", target.read_text(encoding="utf-8"))

    def test_append_applies_after_unrelated_file_change(self) -> None:
        original = (self.root / "preference.md").read_text(encoding="utf-8")
        draft = ConfigDraft(
            "stale",
            "preference.md",
            "t",
            "s",
            "- 新规则",
            "test",
        )
        self.store.create(draft)
        (self.root / "preference.md").write_text(original + "- 别的修改\n", encoding="utf-8")

        outcome = self.store.apply("stale")

        self.assertEqual(outcome.status, "applied")
        content = (self.root / "preference.md").read_text(encoding="utf-8")
        self.assertIn("- 别的修改", content)
        self.assertIn("- 新规则", content)

    def test_replace_applies_on_latest_content_and_preserves_unrelated_bytes(self) -> None:
        current = "# preference\n\n## A\n- 旧规则\n\n\n\n## B\n- 用户稍后修改  \n"
        (self.root / "preference.md").write_text(current, encoding="utf-8")
        draft = ConfigDraft(
            "replace-current",
            "preference.md",
            "t",
            "s",
            "- 新规则",
            "test",
            edit_mode="replace",
            search_text="- 旧规则",
            replace_text="- 新规则",
        )
        self.store.create(draft)

        outcome = self.store.apply(draft.draft_id)

        self.assertEqual(outcome.status, "applied")
        self.assertEqual(
            (self.root / "preference.md").read_text(encoding="utf-8"),
            current.replace("- 旧规则", "- 新规则"),
        )

    def test_independent_replace_drafts_can_be_applied_in_sequence(self) -> None:
        (self.root / "preference.md").write_text("# preference\n\n## A\n- A0\n\n## B\n- B0\n", encoding="utf-8")
        for draft_id, old, new in (("a", "- A0", "- A1"), ("b", "- B0", "- B1")):
            self.store.create(
                ConfigDraft(
                    draft_id,
                    "preference.md",
                    "t",
                    "s",
                    new,
                    "test",
                    edit_mode="replace",
                    search_text=old,
                    replace_text=new,
                )
            )

        self.assertEqual(self.store.apply("a").status, "applied")
        self.assertEqual(self.store.apply("b").status, "applied")
        content = (self.root / "preference.md").read_text(encoding="utf-8")
        self.assertIn("- A1", content)
        self.assertIn("- B1", content)

    def test_replace_conflicts_keep_original_draft_pending_for_refresh(self) -> None:
        original = "# preference\n\n## A\n- 旧规则\n"
        (self.root / "preference.md").write_text(original, encoding="utf-8")
        for draft_id, current in (
            ("missing-search", "# preference\n\n## A\n- 已被别人改写\n"),
            ("duplicate-search", "# preference\n\n## A\n- 旧规则\n- 旧规则\n"),
        ):
            with self.subTest(draft_id=draft_id):
                (self.root / "preference.md").write_text(current, encoding="utf-8")
                draft = ConfigDraft(
                    draft_id,
                    "preference.md",
                    "t",
                    "s",
                    "- 新规则",
                    "test",
                    edit_mode="replace",
                    search_text="- 旧规则",
                    replace_text="- 新规则",
                )
                self.store.create(draft)

                outcome = self.store.apply(draft_id)

                self.assertEqual(outcome.status, "needs_refresh")
                self.assertEqual(self.store.get(draft_id).status, "pending")
                self.assertEqual((self.root / "preference.md").read_text(encoding="utf-8"), current)

    def test_structural_conflict_needs_refresh_without_writing(self) -> None:
        current = "# preference\n\n## A\n- 旧规则\n"
        (self.root / "preference.md").write_text(current, encoding="utf-8")
        draft = ConfigDraft(
            "bad-structure",
            "preference.md",
            "t",
            "s",
            "# 第二个一级标题",
            "test",
            edit_mode="replace",
            search_text="- 旧规则",
            replace_text="# 第二个一级标题",
        )
        self.store.create(draft)

        outcome = self.store.apply(draft.draft_id)

        self.assertEqual(outcome.status, "needs_refresh")
        self.assertIn("多个一级标题", outcome.message)
        self.assertEqual((self.root / "preference.md").read_text(encoding="utf-8"), current)

    def test_legacy_hash_is_ignored_on_load_and_removed_on_save(self) -> None:
        raw = {
            "legacy": {
                "draft_id": "legacy",
                "target_file": "preference.md",
                "title": "t",
                "summary": "s",
                "append_text": "- 新规则",
                "source": "test",
                "base_content_hash": "old-hash",
            }
        }
        (self.root / "drafts.json").write_text(json.dumps(raw), encoding="utf-8")

        store = DraftStore(
            draft_file=self.root / "drafts.json",
            backup_dir=self.root / "backups",
            audit_file=self.root / "audit.jsonl",
            root=self.root,
        )
        legacy = store.get("legacy")
        self.assertIsNotNone(legacy)
        store.update(legacy)

        saved = json.loads((self.root / "drafts.json").read_text(encoding="utf-8"))
        self.assertNotIn("base_content_hash", saved["legacy"])

    def test_legacy_arbitration_draft_is_terminally_rejected(self) -> None:
        draft = ConfigDraft(
            "legacy-arbitration",
            "preference.md",
            "t",
            "s",
            "- 新规则",
            "test",
            metadata={
                "card_kind": "arbitration",
                "arbitration_card": {"analysis": "旧分析"},
            },
        )
        self.store.create(draft)

        outcome = self.store.apply(draft.draft_id)

        self.assertEqual(outcome.status, "rejected")
        self.assertEqual(self.store.get(draft.draft_id).status, "cancelled")
        self.assertEqual(
            self.store.get(draft.draft_id).metadata["cancel_reason"],
            "legacy_arbitration_protocol",
        )

    def test_expire_pending_cancels_only_expired_pending_drafts(self) -> None:
        expired = ConfigDraft(
            "expired",
            "preference.md",
            "t",
            "s",
            "- 旧草案",
            "test",
            created_at=100.0,
            preview_message_id="om_expired",
        )
        active = ConfigDraft(
            "active",
            "preference.md",
            "t",
            "s",
            "- 新草案",
            "test",
            created_at=100.0 + 24 * 60 * 60,
        )
        applied = ConfigDraft(
            "applied",
            "preference.md",
            "t",
            "s",
            "- 已应用",
            "test",
            created_at=100.0,
            status="applied",
        )
        self.store.create(expired)
        self.store.create(active)
        self.store.create(applied)

        with patch("smzdm_notice.preferences.models.time.time", return_value=100.0 + 24 * 60 * 60 + 1):
            expired_drafts = self.store.expire_pending()

        self.assertEqual([d.draft_id for d in expired_drafts], ["expired"])
        self.assertEqual(self.store.get("expired").status, "cancelled")
        self.assertEqual(self.store.get("expired").metadata["cancel_reason"], "expired")
        self.assertEqual(self.store.get("active").status, "pending")
        self.assertEqual(self.store.get("applied").status, "applied")
        audit = [json.loads(line) for line in (self.root / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(audit[-1]["action"], "cancelled")
        self.assertEqual(audit[-1]["metadata"]["cancel_reason"], "expired")

    def test_apply_expired_draft_persists_reason_and_audit(self) -> None:
        self.store.create(
            ConfigDraft(
                "expired-on-apply",
                "preference.md",
                "t",
                "s",
                "- 旧草案",
                "test",
                created_at=100.0,
            )
        )

        with patch("smzdm_notice.preferences.models.time.time", return_value=100.0 + 24 * 60 * 60 + 1):
            outcome = self.store.apply("expired-on-apply", operator="u1")

        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.status, "expired")
        self.assertIn("自动失效", outcome.message)
        self.assertEqual(self.store.get("expired-on-apply").metadata["cancel_reason"], "expired")
        audit = json.loads((self.root / "audit.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        self.assertEqual(audit["operator"], "u1")
        self.assertEqual(audit["metadata"]["cancel_reason"], "expired")

    def test_cancel_requires_explicit_reason(self) -> None:
        self.store.create(ConfigDraft("cancel-me", "preference.md", "t", "s", "- 规则", "test"))

        with self.assertRaises(TypeError):
            self.store.cancel("cancel-me")  # type: ignore[call-arg]

    def test_only_user_cancelled_suppresses_recent_suggestion(self) -> None:
        expected = {
            "user_cancelled": True,
            "expired": False,
            "preview_send_failed": False,
            "superseded_by_revision": False,
        }
        for index, (reason, suppressed) in enumerate(expected.items()):
            suggestion_hash = f"suggestion-{index}"
            draft = ConfigDraft(
                f"cancel-{index}",
                "preference.md",
                "t",
                "s",
                f"- 规则 {index}",
                "test",
                metadata={"suggestion_hash": suggestion_hash},
            )
            self.store.create(draft)
            self.store.cancel(draft.draft_id, reason=reason)
            self.assertEqual(
                self.store.has_recent_suggestion(suggestion_hash),
                suppressed,
                reason,
            )

    def test_compact_removes_old_terminal_drafts_only(self) -> None:
        old_cancelled = ConfigDraft(
            "old-cancelled",
            "preference.md",
            "t",
            "s",
            "- old",
            "test",
            created_at=100.0,
            status="cancelled",
        )
        old_applied = ConfigDraft(
            "old-applied",
            "preference.md",
            "t",
            "s",
            "- old applied",
            "test",
            created_at=100.0,
            status="applied",
        )
        old_pending = ConfigDraft(
            "old-pending",
            "preference.md",
            "t",
            "s",
            "- old pending",
            "test",
            created_at=100.0,
        )
        recent_cancelled = ConfigDraft(
            "recent-cancelled",
            "preference.md",
            "t",
            "s",
            "- recent",
            "test",
            created_at=100.0 + 24 * 60 * 60,
            status="cancelled",
        )
        for draft in (old_cancelled, old_applied, old_pending, recent_cancelled):
            self.store.create(draft)

        with patch("smzdm_notice.preferences.store.time.time", return_value=100.0 + 24 * 60 * 60 + 1):
            removed = self.store.compact()

        self.assertEqual({d.draft_id for d in removed}, {"old-cancelled", "old-applied"})
        self.assertIsNone(self.store.get("old-cancelled"))
        self.assertIsNone(self.store.get("old-applied"))
        self.assertIsNotNone(self.store.get("old-pending"))
        self.assertIsNotNone(self.store.get("recent-cancelled"))

    def test_backup_names_do_not_collide_within_same_second(self) -> None:
        target = self.root / "preference.md"
        with patch("smzdm_notice.preferences.store.datetime") as mock_datetime:
            mock_datetime.now.side_effect = [
                datetime(2026, 5, 18, 13, 30, 1, 1),
                datetime(2026, 5, 18, 13, 30, 1, 2),
            ]
            first = self.store._backup(target)
            second = self.store._backup(target)

        self.assertNotEqual(first.name, second.name)
        self.assertTrue(first.exists())
        self.assertTrue(second.exists())

    def test_rejects_unknown_target_file(self) -> None:
        draft = ConfigDraft("d1", "main.py", "t", "s", "bad", "test")
        with self.assertRaises(ValueError):
            self.store.create(draft)

    def test_deal_action_draft(self) -> None:
        (self.root / "preference.md").write_text(
            "# preference\n- **生活家电**：例如电风扇、台灯、加湿器等\n",
            encoding="utf-8",
        )
        llm_data = {
            "target_file": "preference.md",
            "edit_mode": "replace",
            "title": "关注婴儿推车",
            "summary": "把婴儿推车加入关注品类",
            "search_text": "- **生活家电**：例如电风扇、台灯、加湿器等",
            "replace_text": "- **生活家电**：例如电风扇、台灯、加湿器等\n- **婴儿推车**：有好价可以推荐",
        }
        value = {
            "item_title": "婴儿推车 好价",
            "item_brand": "测试品牌",
            "article_id": "1",
            "item_link": "https://example.com/deal/1",
        }
        with patch("smzdm_notice.preferences.builder._draft_with_llm", return_value=llm_data) as draft_with_llm:
            draft = build_deal_action_draft("deal_follow", value, self.store)

        self.assertIsNotNone(draft)
        self.assertEqual(draft.target_file, "preference.md")
        self.assertEqual(draft.edit_mode, "replace")
        self.assertEqual(draft.source, "商品卡片快捷操作")
        self.assertEqual(draft.metadata["article_id"], "1")
        self.assertIn("婴儿推车", draft.replace_text)
        message = draft_with_llm.call_args.args[0]
        self.assertIn("关注与该商品相关或同类商品", message)
        self.assertIn("婴儿推车 好价", message)
        self.assertIn("测试品牌", message)
        self.assertEqual(draft_with_llm.call_args.kwargs["root"], self.store.root)

    def test_deal_action_messages_describe_each_intent(self) -> None:
        value = {"item_title": "抽纸", "item_brand": "蓝月亮", "article_id": "1001"}

        ignore = _deal_action_message("deal_ignore_category", value)
        stock = _deal_action_message("deal_stock_enough", value)
        follow = _deal_action_message("deal_follow", value)

        self.assertIn("不要推荐与该商品同类或高度相似", ignore)
        self.assertIn("库存充足", stock)
        self.assertIn("关注与该商品相关或同类商品", follow)
        self.assertIn("抽纸", ignore)
        self.assertIn("蓝月亮", stock)
        self.assertIn("1001", follow)

    def test_deal_action_draft_returns_none_when_llm_fails(self) -> None:
        with patch("smzdm_notice.preferences.builder._draft_with_llm", return_value=None):
            draft = build_deal_action_draft(
                "deal_follow",
                {"item_title": "婴儿推车 好价", "article_id": "1"},
                self.store,
            )

        self.assertIsNone(draft)

    def test_revision_of_append_can_remove_part_from_pending_draft(self) -> None:
        original = ConfigDraft(
            draft_id="original",
            target_file="preference.md",
            title="开放推荐规则",
            summary="测试",
            append_text="- 开放推荐硬性门槛\n- 推荐理由真实性",
            source="仲裁建议一键采纳",
        )
        with patch(
            "smzdm_notice.preferences.builder._revision_with_llm",
            return_value={
                "target_file": "preference.md",
                "edit_mode": "append",
                "title": "开放推荐规则",
                "summary": "去掉硬性门槛要求，只保留理由真实性",
                "append_text": "- 推荐理由真实性",
            },
        ):
            draft = build_revision_draft("不要硬性门槛要求", original, self.store)

        self.assertIsNotNone(draft)
        self.assertEqual(draft.edit_mode, "append")
        self.assertEqual(draft.append_text, "- 推荐理由真实性")

    def test_revision_retries_when_delete_targets_unapplied_draft_text(self) -> None:
        original = ConfigDraft(
            draft_id="original",
            target_file="preference.md",
            title="开放推荐规则",
            summary="测试",
            append_text="- 开放推荐硬性门槛\n- 推荐理由真实性",
            source="仲裁建议一键采纳",
        )
        bad_delete = {
            "target_file": "preference.md",
            "edit_mode": "delete",
            "title": "移除硬性门槛",
            "summary": "错误地删除未执行草案文本",
            "search_text": "- 开放推荐硬性门槛",
        }
        fixed_append = {
            "target_file": "preference.md",
            "edit_mode": "append",
            "title": "开放推荐规则",
            "summary": "去掉硬性门槛要求，只保留理由真实性",
            "append_text": "- 推荐理由真实性",
        }
        with patch(
            "smzdm_notice.preferences.builder._revision_with_llm",
            side_effect=[bad_delete, fixed_append],
        ) as revise:
            draft = build_revision_draft("不要硬性门槛要求", original, self.store)

        self.assertEqual(revise.call_count, 2)
        self.assertIsNotNone(draft)
        self.assertEqual(draft.edit_mode, "append")
        self.assertEqual(draft.append_text, "- 推荐理由真实性")

    def test_revision_returns_none_when_retry_still_targets_unapplied_draft_text(self) -> None:
        original = ConfigDraft(
            draft_id="original",
            target_file="preference.md",
            title="开放推荐规则",
            summary="测试",
            append_text="- 开放推荐硬性门槛\n- 推荐理由真实性",
            source="仲裁建议一键采纳",
        )
        bad_delete = {
            "target_file": "preference.md",
            "edit_mode": "delete",
            "title": "移除硬性门槛",
            "summary": "错误地删除未执行草案文本",
            "search_text": "- 开放推荐硬性门槛",
        }
        with patch("smzdm_notice.preferences.builder._revision_with_llm", side_effect=[bad_delete, bad_delete]):
            draft = build_revision_draft("不要硬性门槛要求", original, self.store)

        self.assertIsNone(draft)

    def test_revision_allows_delete_when_search_text_exists_in_real_file(self) -> None:
        (self.root / "preference.md").write_text("# preference\n- 旧规则\n", encoding="utf-8")
        original = ConfigDraft(
            draft_id="original",
            target_file="preference.md",
            title="开放推荐规则",
            summary="测试",
            append_text="- 新规则",
            source="用户对话：新增规则",
        )
        with patch(
            "smzdm_notice.preferences.builder._revision_with_llm",
            return_value={
                "target_file": "preference.md",
                "edit_mode": "delete",
                "title": "删除旧规则",
                "summary": "用户明确要求删除真实文件里的旧规则",
                "search_text": "- 旧规则",
            },
        ):
            draft = build_revision_draft("不要追加了，删除文件里的旧规则", original, self.store)

        self.assertIsNotNone(draft)
        self.assertEqual(draft.edit_mode, "delete")
        self.assertEqual(draft.search_text, "- 旧规则")

    def test_revision_allows_replace_when_search_text_exists_in_real_file(self) -> None:
        (self.root / "preference.md").write_text("# preference\n- 旧规则\n", encoding="utf-8")
        original = ConfigDraft(
            draft_id="original",
            target_file="preference.md",
            title="开放推荐规则",
            summary="测试",
            append_text="- 新规则",
            source="用户对话：新增规则",
        )
        with patch(
            "smzdm_notice.preferences.builder._revision_with_llm",
            return_value={
                "target_file": "preference.md",
                "edit_mode": "replace",
                "title": "替换旧规则",
                "summary": "用户明确要求替换真实文件里的旧规则",
                "search_text": "- 旧规则",
                "replace_text": "- 新规则",
            },
        ):
            draft = build_revision_draft("不要追加了，改文件里的旧规则", original, self.store)

        self.assertIsNotNone(draft)
        self.assertEqual(draft.edit_mode, "replace")
        self.assertEqual(draft.search_text, "- 旧规则")
        self.assertEqual(draft.replace_text, "- 新规则")


if __name__ == "__main__":
    unittest.main()
