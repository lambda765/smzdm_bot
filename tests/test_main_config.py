from __future__ import annotations

import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import smzdm_notice.runtime as main
from smzdm_notice.llm.models import ArbiterInfo, FilterDiagnostics, FilterItemsResult, FilterResult
from smzdm_notice.preferences.models import ConfigDraft
from smzdm_notice.preferences.store import DraftStore
from smzdm_notice.smzdm.ranking import RankingItem


def _item() -> RankingItem:
    return RankingItem(
        rank=1,
        title="测试商品",
        article_id="1001",
        price="9.9元",
        worthy=10,
        unworthy=0,
        comments=1,
        favorites=2,
        mall="测试商城",
        brand="测试品牌",
        link="https://example.com/deal/1001",
    )


class MainConfigValidationTests(unittest.TestCase):
    def test_accepts_complete_feishu_app_config(self) -> None:
        with ExitStack() as stack:
            stack.enter_context(patch("smzdm_notice.runtime.config.FEISHU_APP_ID", "cli_real_app_id"))
            stack.enter_context(patch("smzdm_notice.runtime.config.FEISHU_APP_SECRET", "secret"))
            stack.enter_context(patch("smzdm_notice.runtime.config.LLM_API_KEY", "key"))
            self.assertTrue(main._validate_config())

    def test_rejects_incomplete_feishu_app_config(self) -> None:
        with ExitStack() as stack:
            stack.enter_context(patch("smzdm_notice.runtime.config.FEISHU_APP_ID", "cli_real_app_id"))
            stack.enter_context(patch("smzdm_notice.runtime.config.FEISHU_APP_SECRET", ""))
            stack.enter_context(patch("smzdm_notice.runtime.config.LLM_API_KEY", "key"))
            stack.enter_context(patch("smzdm_notice.runtime.logger.error"))
            self.assertFalse(main._validate_config())

    def test_rejects_placeholder_app_id(self) -> None:
        with ExitStack() as stack:
            stack.enter_context(patch("smzdm_notice.runtime.config.FEISHU_APP_ID", "cli_xxx"))
            stack.enter_context(patch("smzdm_notice.runtime.config.FEISHU_APP_SECRET", "secret"))
            stack.enter_context(patch("smzdm_notice.runtime.config.LLM_API_KEY", "key"))
            stack.enter_context(patch("smzdm_notice.runtime.logger.error"))
            self.assertFalse(main._validate_config())

    def test_digest_failure_does_not_clear_near_miss_cache(self) -> None:
        near_miss_mgr = MagicMock()
        near_miss_mgr.get_last_digest_date.return_value = ""
        near_miss_mgr.get_all_sorted.return_value = [{"title": f"item {index}"} for index in range(21)]

        with (
            patch("smzdm_notice.runtime.config.DIGEST_HOUR", 0),
            patch("smzdm_notice.runtime.send_digest", return_value=False) as send_digest,
        ):
            main._check_digest(near_miss_mgr)

        send_digest.assert_called_once()
        near_miss_mgr.clear_and_set_digest_date.assert_not_called()


class MainConfigParsingTests(unittest.TestCase):
    def test_get_bool_parses_common_values_and_falls_back(self) -> None:
        with patch.dict(
            os.environ,
            {
                "BOOL_ONE": "1",
                "BOOL_YES": "yes",
                "BOOL_ON": "on",
                "BOOL_FALSE": "false",
                "BOOL_OFF": "off",
                "BOOL_BAD": "maybe",
            },
        ):
            self.assertTrue(main.config._get_bool("BOOL_ONE"))
            self.assertTrue(main.config._get_bool("BOOL_YES"))
            self.assertTrue(main.config._get_bool("BOOL_ON"))
            self.assertFalse(main.config._get_bool("BOOL_FALSE", True))
            self.assertFalse(main.config._get_bool("BOOL_OFF", True))
            self.assertTrue(main.config._get_bool("BOOL_BAD", True))

    def test_get_float_parses_values_and_falls_back(self) -> None:
        with patch.dict(os.environ, {"FLOAT_VALUE": "0.75", "FLOAT_BAD": "bad"}):
            self.assertEqual(main.config._get_float("FLOAT_VALUE"), 0.75)
            self.assertEqual(main.config._get_float("FLOAT_BAD", 0.25), 0.25)

    def test_get_fallback_uses_fallback_for_missing_or_empty_values(self) -> None:
        with patch.dict(os.environ, {"VALUE": "configured", "EMPTY": ""}, clear=False):
            self.assertEqual(main.config._get_fallback("VALUE", "fallback"), "configured")
            self.assertEqual(main.config._get_fallback("EMPTY", "fallback"), "fallback")
            self.assertEqual(main.config._get_fallback("MISSING", "fallback"), "fallback")

    def test_get_float_fallback_parses_values_and_falls_back(self) -> None:
        with patch.dict(os.environ, {"TIMEOUT": "123.5", "EMPTY_TIMEOUT": "", "BAD_TIMEOUT": "bad"}, clear=False):
            self.assertEqual(main.config._get_float_fallback("TIMEOUT", 300.0), 123.5)
            self.assertEqual(main.config._get_float_fallback("EMPTY_TIMEOUT", 300.0), 300.0)
            self.assertEqual(main.config._get_float_fallback("BAD_TIMEOUT", 300.0), 300.0)
            self.assertEqual(main.config._get_float_fallback("MISSING_TIMEOUT", 300.0), 300.0)

    def test_clamp_rate_limits_to_zero_one_range(self) -> None:
        self.assertEqual(main.config._clamp_rate(-0.1), 0.0)
        self.assertEqual(main.config._clamp_rate(0.6), 0.6)
        self.assertEqual(main.config._clamp_rate(1.5), 1.0)


class MainConfigSummaryTests(unittest.TestCase):
    def _summary_with_prefilter(self, **values) -> str:
        defaults = {
            "PREFILTER_ENABLED": False,
            "PREFILTER_BYPASS_ENABLED": False,
            "PREFILTER_MIN_WORTHY": 0,
            "PREFILTER_MIN_WORTHY_RATE": 0.0,
            "PREFILTER_MIN_COMMENTS": 0,
            "PREFILTER_MIN_FAVORITES": 0,
            "PREFILTER_BYPASS_MIN_COMMENTS": 0,
            "PREFILTER_BYPASS_MIN_WORTHY": 0,
        }
        defaults.update(values)
        with ExitStack() as stack:
            stack.enter_context(patch.object(main, "_ranking_configs", [SimpleNamespace(name="综合榜")]))
            stack.enter_context(patch.object(main, "_search_keywords", []))
            stack.enter_context(patch.object(main, "_current_user_prompt", "偏好正文"))
            for key, value in defaults.items():
                stack.enter_context(patch.object(main.config, key, value))
            return main._config_summary()

    def test_config_summary_shows_prefilter_disabled(self) -> None:
        summary = self._summary_with_prefilter(PREFILTER_ENABLED=False)

        self.assertIn("预筛选: 未启用", summary)

    def test_config_summary_shows_regular_prefilter_thresholds(self) -> None:
        summary = self._summary_with_prefilter(
            PREFILTER_ENABLED=True,
            PREFILTER_MIN_WORTHY=20,
            PREFILTER_MIN_WORTHY_RATE=0.5,
            PREFILTER_MIN_COMMENTS=10,
            PREFILTER_MIN_FAVORITES=5,
        )

        self.assertIn("预筛选: 已启用", summary)
        self.assertIn("值票≥20", summary)
        self.assertIn("值率≥0.50", summary)
        self.assertIn("评论≥10", summary)
        self.assertIn("收藏≥5", summary)
        self.assertIn("强信号直通: 未启用", summary)

    def test_config_summary_shows_bypass_thresholds(self) -> None:
        summary = self._summary_with_prefilter(
            PREFILTER_ENABLED=True,
            PREFILTER_BYPASS_ENABLED=True,
            PREFILTER_BYPASS_MIN_COMMENTS=100,
            PREFILTER_BYPASS_MIN_WORTHY=200,
        )

        self.assertIn("强信号直通: 评论≥100, 值票≥200", summary)


class MainRestartTests(unittest.TestCase):
    def tearDown(self) -> None:
        main._restart_event.clear()
        main._stop_event.clear()

    def test_request_restart_sets_event_once(self) -> None:
        self.assertTrue(main._request_restart())
        self.assertTrue(main._restart_event.is_set())
        self.assertTrue(main._stop_event.is_set())
        self.assertFalse(main._request_restart())

    def test_exec_restart_marks_dotenv_for_reload(self) -> None:
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {"LLM_API_KEY": "old-key"}, clear=True))
            stack.enter_context(patch.object(main.sys, "executable", "/usr/bin/python"))
            stack.enter_context(patch.object(main.sys, "argv", ["main.py"]))
            execve = stack.enter_context(patch("smzdm_notice.runtime.os.execve"))

            main._exec_restart()

        executable, argv, env = execve.call_args.args
        self.assertEqual(executable, "/usr/bin/python")
        self.assertEqual(argv, ["/usr/bin/python", "main.py"])
        self.assertEqual(env["LLM_API_KEY"], "old-key")
        self.assertEqual(env["SMZDM_RESTART_RELOAD_DOTENV"], "1")


class MainLifecycleTests(unittest.TestCase):
    def test_main_closes_smzdm_client_before_shutdown_notification(self) -> None:
        dedup = MagicMock()
        near_miss_mgr = MagicMock()
        binding_store = MagicMock()
        calls = []

        with ExitStack() as stack:
            stack.enter_context(patch("smzdm_notice.runtime._ensure_startup_ready"))
            stack.enter_context(
                patch("smzdm_notice.runtime._initialize_runtime", return_value=(dedup, near_miss_mgr, binding_store))
            )
            stack.enter_context(patch("smzdm_notice.runtime._notify_startup_if_bound"))
            stack.enter_context(
                patch("smzdm_notice.runtime._run_poll_loop", side_effect=lambda *_args: calls.append("poll"))
            )
            stack.enter_context(patch("smzdm_notice.runtime.close_client", side_effect=lambda: calls.append("close")))
            notify = stack.enter_context(
                patch("smzdm_notice.runtime._notify_shutdown_or_restart", side_effect=lambda: calls.append("notify"))
            )

            main.main()

        self.assertEqual(calls, ["poll", "close", "notify"])
        notify.assert_called_once_with()

    def test_main_closes_smzdm_client_when_poll_loop_raises(self) -> None:
        dedup = MagicMock()
        near_miss_mgr = MagicMock()
        binding_store = MagicMock()
        error = RuntimeError("poll failed")

        with ExitStack() as stack:
            stack.enter_context(patch("smzdm_notice.runtime._ensure_startup_ready"))
            stack.enter_context(
                patch("smzdm_notice.runtime._initialize_runtime", return_value=(dedup, near_miss_mgr, binding_store))
            )
            stack.enter_context(patch("smzdm_notice.runtime._notify_startup_if_bound"))
            stack.enter_context(patch("smzdm_notice.runtime._run_poll_loop", side_effect=error))
            close = stack.enter_context(patch("smzdm_notice.runtime.close_client"))
            notify = stack.enter_context(patch("smzdm_notice.runtime._notify_shutdown_or_restart"))

            with self.assertRaises(RuntimeError):
                main.main()

        close.assert_called_once_with()
        notify.assert_not_called()


class MainStopSignalTests(unittest.TestCase):
    def tearDown(self) -> None:
        main._stop_event.clear()

    def _item(self) -> RankingItem:
        return _item()

    def test_stop_during_fetch_skips_llm_filter(self) -> None:
        def fetch_and_stop(*args, **kwargs):
            main._stop_event.set()
            return [self._item()]

        with ExitStack() as stack:
            stack.enter_context(patch("smzdm_notice.runtime.fetch_all_sources", side_effect=fetch_and_stop))
            filter_mock = stack.enter_context(patch("smzdm_notice.runtime.filter_items"))
            dedup = MagicMock()
            near_miss_mgr = MagicMock()

            main._poll_once_unlocked(dedup, near_miss_mgr)

            filter_mock.assert_not_called()
            dedup.is_new.assert_not_called()

    def test_stop_after_dedup_skips_llm_filter(self) -> None:
        def mark_stop(_link: str) -> bool:
            main._stop_event.set()
            return True

        with ExitStack() as stack:
            stack.enter_context(patch("smzdm_notice.runtime.fetch_all_sources", return_value=[self._item()]))
            filter_mock = stack.enter_context(patch("smzdm_notice.runtime.filter_items"))
            dedup = MagicMock()
            dedup.is_new.side_effect = mark_stop
            near_miss_mgr = MagicMock()

            main._poll_once_unlocked(dedup, near_miss_mgr)

            filter_mock.assert_not_called()


class MainArbitrationDraftTests(unittest.TestCase):
    def tearDown(self) -> None:
        main._draft_store = None

    def test_poll_stores_arbitration_draft_before_sending_card(self) -> None:
        item = _item()
        info = ArbiterInfo(
            chosen="B",
            reason="B 更准确",
            analysis="A 过度扩展黑名单。",
            suggestion="黑名单只精确匹配。",
            result_a=FilterResult(),
            result_b=FilterResult(),
            items={},
            config_change_draft={
                "target_file": "preference.md",
                "title": "限制黑名单扩展",
                "summary": "避免误判",
                "append_text": "- 黑名单只按字面精确匹配。",
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            main._draft_store = DraftStore(
                draft_file=root / "drafts.json",
                backup_dir=root / "backups",
                audit_file=root / "audit.jsonl",
                root=root,
            )
            dedup = MagicMock()
            dedup.is_new.return_value = True
            near_miss_mgr = MagicMock()

            with ExitStack() as stack:
                stack.enter_context(patch("smzdm_notice.runtime.fetch_all_sources", return_value=[item]))
                stack.enter_context(
                    patch("smzdm_notice.runtime.filter_items", return_value=FilterItemsResult(arbiter_info=info))
                )
                stack.enter_context(patch("smzdm_notice.runtime._check_digest"))
                stack.enter_context(patch("smzdm_notice.runtime._check_heartbeat"))

                def send_arbitration_side_effect(_info, draft):
                    draft.preview_message_id = "om_arbiter"
                    return True

                send_arbitration = stack.enter_context(
                    patch("smzdm_notice.runtime.send_arbitration", side_effect=send_arbitration_side_effect)
                )

                main._poll_once_unlocked(dedup, near_miss_mgr)

        sent_info, sent_draft = send_arbitration.call_args.args
        self.assertIs(sent_info, info)
        self.assertIsNotNone(sent_draft)
        self.assertEqual(sent_draft.target_file, "preference.md")
        self.assertIn("字面精确匹配", sent_draft.append_text)
        self.assertEqual(main._draft_store.get(sent_draft.draft_id).preview_message_id, "om_arbiter")

    def test_maintain_config_drafts_expires_and_compacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            main._draft_store = DraftStore(
                draft_file=root / "drafts.json",
                backup_dir=root / "backups",
                audit_file=root / "audit.jsonl",
                root=root,
            )
            expired = main._draft_store.create(
                ConfigDraft(
                    draft_id="expired",
                    target_file="preference.md",
                    title="过期草案",
                    summary="测试",
                    append_text="- 旧规则",
                    source="test",
                    created_at=100.0,
                    preview_message_id="om_expired",
                )
            )
            main._draft_store.create(
                ConfigDraft(
                    draft_id="old-cancelled",
                    target_file="preference.md",
                    title="旧取消草案",
                    summary="测试",
                    append_text="- 旧规则",
                    source="test",
                    created_at=100.0,
                    status="cancelled",
                )
            )

            with (
                patch("smzdm_notice.preferences.models.time.time", return_value=100.0 + 24 * 60 * 60 + 1),
                patch("smzdm_notice.preferences.store.time.time", return_value=100.0 + 24 * 60 * 60 + 1),
                patch("smzdm_notice.runtime.disable_draft_card") as disable_card,
            ):
                main._maintain_config_drafts("测试清理")

            self.assertIsNone(main._draft_store.get("expired"))
            self.assertIsNone(main._draft_store.get("old-cancelled"))
            disable_card.assert_called_once_with(
                "om_expired",
                "测试清理：草案已超过 24 小时自动失效",
                expired,
            )


class MainSearchPriceBypassTests(unittest.TestCase):
    def tearDown(self) -> None:
        main._stop_event.clear()

    def _search_item(
        self,
        article_id: str,
        numeric_price: float | None,
        max_price: float,
    ) -> RankingItem:
        item = _item()
        item.article_id = article_id
        item.link = f"https://example.com/deal/{article_id}"
        item.price = "" if numeric_price is None else str(numeric_price)
        item.numeric_price = numeric_price
        item.source_type = "search"
        item.search_keyword = "AirPods Pro 2"
        item.search_max_price = max_price
        return item

    def test_price_bypass_is_item_level_and_excluded_from_llm(self) -> None:
        bypass = self._search_item("bypass", 10.0, 10.0)
        llm_item = self._search_item("llm", 10.1, 10.0)
        dedup = MagicMock()
        dedup.is_new.return_value = True
        near_miss_mgr = MagicMock()

        with ExitStack() as stack:
            stack.enter_context(patch("smzdm_notice.runtime._load_search_keywords", return_value=[]))
            stack.enter_context(patch("smzdm_notice.runtime.fetch_all_sources", return_value=[bypass, llm_item]))
            filter_items = stack.enter_context(
                patch(
                    "smzdm_notice.runtime.filter_items",
                    return_value=FilterItemsResult(matched=[(llm_item, "LLM 推荐")]),
                )
            )
            send_deals = stack.enter_context(patch("smzdm_notice.runtime.send_deals", return_value=True))
            stack.enter_context(patch("smzdm_notice.runtime._check_digest"))
            stack.enter_context(patch("smzdm_notice.runtime._check_heartbeat"))

            outcome = main._poll_once_unlocked(dedup, near_miss_mgr)

        self.assertEqual(outcome.status, "success")
        self.assertEqual(filter_items.call_args.kwargs["items"], [llm_item])
        sent = send_deals.call_args.args[0]
        self.assertEqual([item.article_id for item, _ in sent], ["bypass", "llm"])
        self.assertEqual(send_deals.call_args.kwargs["price_bypass_article_ids"], {"bypass"})
        self.assertIn("小于等于阈值", sent[0][1])
        dedup.mark_batch.assert_called_once_with([bypass.link, llm_item.link])

    def test_dedup_runs_before_price_bypass(self) -> None:
        bypass = self._search_item("bypass", 9.9, 10.0)
        dedup = MagicMock()
        dedup.is_new.return_value = False

        with ExitStack() as stack:
            stack.enter_context(patch("smzdm_notice.runtime._load_search_keywords", return_value=[]))
            stack.enter_context(patch("smzdm_notice.runtime.fetch_all_sources", return_value=[bypass]))
            filter_items = stack.enter_context(patch("smzdm_notice.runtime.filter_items"))
            send_deals = stack.enter_context(patch("smzdm_notice.runtime.send_deals"))
            stack.enter_context(patch("smzdm_notice.runtime._check_heartbeat"))

            main._poll_once_unlocked(dedup, MagicMock())

        filter_items.assert_not_called()
        send_deals.assert_not_called()

    def test_missing_numeric_price_does_not_bypass(self) -> None:
        llm_item = self._search_item("missing-price", None, 10.0)
        dedup = MagicMock()
        dedup.is_new.return_value = True

        with ExitStack() as stack:
            stack.enter_context(patch("smzdm_notice.runtime._load_search_keywords", return_value=[]))
            stack.enter_context(patch("smzdm_notice.runtime.fetch_all_sources", return_value=[llm_item]))
            filter_items = stack.enter_context(
                patch("smzdm_notice.runtime.filter_items", return_value=FilterItemsResult())
            )
            send_deals = stack.enter_context(patch("smzdm_notice.runtime.send_deals"))
            stack.enter_context(patch("smzdm_notice.runtime._check_digest"))
            stack.enter_context(patch("smzdm_notice.runtime._check_heartbeat"))

            main._poll_once_unlocked(dedup, MagicMock())

        self.assertEqual(filter_items.call_args.kwargs["items"], [llm_item])
        send_deals.assert_not_called()


class MainPollFailureTests(unittest.TestCase):
    def tearDown(self) -> None:
        main._stop_event.clear()
        main._poll_failure_tracker = main.PollFailureTracker()

    def test_fetch_failure_returns_failure_outcome(self) -> None:
        with patch("smzdm_notice.runtime.fetch_all_sources", side_effect=RuntimeError("network down")):
            outcome = main._poll_once_unlocked(MagicMock(), MagicMock())

        self.assertEqual(outcome.status, "failure")
        self.assertEqual(outcome.reason, "ranking_fetch_failed")
        self.assertIn("network down", outcome.detail)

    def test_llm_failure_returns_failure_outcome(self) -> None:
        dedup = MagicMock()
        dedup.is_new.return_value = True
        with ExitStack() as stack:
            stack.enter_context(patch("smzdm_notice.runtime.fetch_all_sources", return_value=[_item()]))
            stack.enter_context(
                patch(
                    "smzdm_notice.runtime.filter_items",
                    return_value=FilterItemsResult(
                        diagnostics=FilterDiagnostics(llm_failed=True, error_summary="usage limit")
                    ),
                )
            )
            stack.enter_context(patch("smzdm_notice.runtime._check_digest"))
            stack.enter_context(patch("smzdm_notice.runtime._check_heartbeat"))

            outcome = main._poll_once_unlocked(dedup, MagicMock())

        self.assertEqual(outcome.status, "failure")
        self.assertEqual(outcome.reason, "llm_failed")
        self.assertEqual(outcome.detail, "usage limit")

    def test_poll_failure_tracker_warns_once_per_failure_streak(self) -> None:
        tracker = main.PollFailureTracker()
        with patch("smzdm_notice.runtime.send_poll_failure_warning", return_value=True) as warn:
            tracker.record(main.PollOutcome.failure("ranking_fetch_failed", "network"))
            tracker.record(main.PollOutcome.skipped("stopped"))
            tracker.record(main.PollOutcome.failure("llm_failed", "429"))
            warn.assert_not_called()

            tracker.record(main.PollOutcome.failure("llm_failed", "429 again"))
            warn.assert_called_once_with(3, "llm_failed", "429 again")

            tracker.record(main.PollOutcome.failure("ranking_fetch_failed", "network again"))
            warn.assert_called_once()

            tracker.record(main.PollOutcome.success())
            tracker.record(main.PollOutcome.failure("llm_failed", "a"))
            tracker.record(main.PollOutcome.failure("llm_failed", "b"))
            tracker.record(main.PollOutcome.failure("llm_failed", "c"))

        self.assertEqual(warn.call_count, 2)
        self.assertEqual(warn.call_args.args, (3, "llm_failed", "c"))

    def test_poll_once_records_outcome(self) -> None:
        tracker = MagicMock()
        main._poll_failure_tracker = tracker
        with patch(
            "smzdm_notice.runtime._poll_once_unlocked",
            return_value=main.PollOutcome.failure("llm_failed", "429"),
        ):
            main._poll_once(MagicMock(), MagicMock())

        tracker.record.assert_called_once()
        self.assertEqual(tracker.record.call_args.args[0].reason, "llm_failed")


if __name__ == "__main__":
    unittest.main()
