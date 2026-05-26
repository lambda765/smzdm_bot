"""LLM JSON 响应解析工具。"""

from __future__ import annotations

import json
import re


def extract_json_object(content: str | None) -> dict | None:
    """从模型响应中提取 JSON 对象，兼容 think 标签和 Markdown 代码块。"""
    if not content:
        return None

    # 不同模型和网关对 response_format 的遵守程度不同：有的会带 <think>，
    # 有的会包 Markdown 代码块，也有的在 JSON 前后补解释文本。这里集中兼容，
    # 让业务层只处理“有没有拿到对象”，不散落解析兜底。
    clean = re.sub(r"<think[^>]*>.*?</think\s*>", "", content, flags=re.DOTALL).strip()
    if not clean:
        clean = content

    json_block_match = re.search(r"```json\s*(.*?)\s*```", clean, re.DOTALL)
    if json_block_match:
        data = _loads_object(json_block_match.group(1))
        if data is not None:
            return data

    data = _loads_object(clean)
    if data is not None:
        return data

    json_match = re.search(r"\{.*\}", clean, re.DOTALL)
    if json_match:
        data = _loads_object(json_match.group())
        if data is not None:
            return data

    return _scan_first_json_object(clean)


def parse_json_object(content: str) -> dict:
    """解析包含 JSON 对象的文本；失败时抛 ValueError。"""
    text = (content or "").strip()
    if not text:
        raise ValueError("LLM 返回内容为空")
    data = extract_json_object(text)
    if not isinstance(data, dict):
        raise ValueError("LLM 返回内容不是 JSON 对象")
    return data


def _loads_object(text: str) -> dict | None:
    try:
        data = json.loads(text.strip())
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError):
        return None
    return None


def _scan_first_json_object(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            data, _ = decoder.raw_decode(text[start:])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    return None
