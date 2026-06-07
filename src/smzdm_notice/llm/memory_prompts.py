"""Deal Memory 相关 Prompt 模板。"""

from __future__ import annotations

from smzdm_notice.core import config
from smzdm_notice.core.calibration import load_calibration_profile_data
from smzdm_notice.llm.categories import (
    UNCATEGORIZED_CATEGORY,
    candidate_calibration_categories,
    candidate_search_keywords,
)
from smzdm_notice.smzdm.ranking import RankingItem


def build_calibration_section(items: list[RankingItem] | None = None) -> str:
    """构建校准参考段落，注入到 filter prompt 的 user_message 中。

    Returns 空字符串如果 memory 未启用或校准数据不足。
    """
    if not config.DEAL_MEMORY_ENABLED:
        return ""

    items = items or []
    if not items:
        return ""

    profile = load_calibration_profile_data(config.CALIBRATION_FILE)
    calibration_by_category = profile.get("calibration_by_category", {})
    if not isinstance(calibration_by_category, dict) or not calibration_by_category:
        return ""

    source_categories = candidate_calibration_categories(items)
    search_keywords = candidate_search_keywords(items)
    sections: list[str] = []
    for category, data in sorted(calibration_by_category.items()):
        if category == UNCATEGORIZED_CATEGORY or not isinstance(data, dict):
            continue
        if category in source_categories:
            sections.append(str(data.get("text") or ""))
            continue
        historical_keywords = {
            str(keyword).strip() for keyword in data.get("search_keywords", []) if str(keyword).strip()
        }
        if historical_keywords and historical_keywords.intersection(search_keywords):
            sections.append(str(data.get("text") or ""))

    sections = [section for section in sections if section.strip()]
    if not sections:
        return ""

    return "## 历史决策校准参考（参考信息，不覆盖上述规则）\n\n" + "\n\n".join(sections) + "\n\n"


MEMORY_ANALYSIS_SYSTEM_PROMPT = """\
<role>
你是一个购物偏好分析助手。你的任务是分析用户对推荐商品的「好价/不值」反馈历史，发现用户的长期偏好模式。
</role>

<input>
你会收到一个 JSON 数组，每个元素是一条推荐记录，包含：
- 商品信息（标题、价格、品牌、商城、值票、不值票、评论、收藏、标签、品类提示）
- recommendation: 推荐时的筛选理由、是否经过仲裁、推荐时间
- context: 推荐时的轻量上下文（筛选理由和快照时间）
- feedback: 用户的评价（deal_good = 好价，deal_not_worth = 不值）和评价时间
</input>

<task>
1. 分析好价和不值记录的差异，找出规律
2. 结合商品信号、品类提示、推荐理由和用户反馈判断模式
3. 如果发现足够强的模式，生成可写入用户偏好文件 preference.md 的具体规则
4. 如果数据不足以形成可靠结论，不要强行生成规则
</task>

<analysis_dimensions>
你可以从以下维度分析，但不限于这些：
- 品类偏好：哪些品类用户倾向好价，哪些倾向不值
- 价格区间：用户对不同品类接受的价格范围
- 信号阈值：好价案例和不值案例在值票、评论、值率上的差异
- 上下文关联：同样的品类/价格在不同偏好/库存状态下用户评价是否不同
- 标签效应：历史低价、好价等标签对用户评价的影响
</analysis_dimensions>

<output_format>
严格输出 JSON：

{
  "summary": "一段话总结用户的长期偏好特征（50-100字）",
  "patterns": [
    {
      "dimension": "品类偏好/价格区间/信号阈值/上下文关联",
      "description": "发现的具体模式",
      "evidence_count": 5,
      "confidence": "high/medium/low"
    }
  ],
  "suggested_rules": [
    {
      "rule": "可以直接写入 preference.md 的具体筛选规则",
      "reason": "为什么建议添加这条规则",
      "evidence": "支撑这条规则的数据摘要，必须写明 good/not_worth 样本数量",
      "good_count": 5,
      "not_worth_count": 0,
      "evidence_scope": "可选：该规则基于哪个品类、标签或场景"
    }
  ]
}

suggested_rules 的 rule 字段必须是面向 preference.md 的具体规则文本，不含分析包装。
每条 rule 应该是一个独立、明确的筛选条件或偏好表达。
suggested_rules 不使用 evidence_count 作为证据字段，必须使用 good_count 和 not_worth_count。
patterns 可以使用 evidence_count 记录探索性发现，但 patterns 不会自动进入 preference.md 草案。
如果 patterns 的 confidence 都是 low 或数据不足以形成规则，suggested_rules 输出空数组。
不要重复用户偏好中已有的规则。
</output_format>

<constraints>
- 不得生成硬阈值规则，例如"值票 >= X"、"评论 >= X"、"值率 >= X"、"必须 X 票以上"。
- 每条 suggested_rules 必须输出 good_count 和 not_worth_count；两者都必须是非负整数，可以为 0。
- 每条 suggested_rules 还必须在 reason 或 evidence 中用自然语言注明 good/not_worth 样本数量；合计少于 5 条时不要生成规则。
- 同一品类 good/not_worth 比例在 1:2 到 2:1 之间时，说明正反样本不稳定，不要生成该品类规则。
- 不要重复用户偏好中已有的规则；不确定是否已有时只写入 patterns，不写 suggested_rules。
</constraints>
"""
