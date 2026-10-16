"""通用 LLM 调用辅助工具

统一从节点的 config['metadata']['llm_cfg'] 读取大模型配置（config/sp/up），
使用 Jinja2 渲染用户提示词并调用 LLMClient，返回响应的纯文本。
"""
import json
import os
from typing import Any, Optional, Mapping

from coze_coding_dev_sdk import LLMClient
from coze_coding_utils.runtime_ctx.context import Context
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from jinja2 import Template
from langgraph.runtime import Runtime


def resolve_llm_cfg(config: Mapping[str, Any]) -> dict:
    """从节点 config metadata 解析大模型配置文件路径并加载"""
    metadata = dict(config.get("metadata") or {})
    cfg_rel = metadata.get("llm_cfg", "")
    cfg_path = os.path.join(os.getenv("COZE_WORKSPACE_PATH", ""), cfg_rel)
    with open(cfg_path, "r", encoding="utf-8") as fd:
        return json.load(fd)


def invoke_llm(
    runtime: Runtime,
    config: Mapping[str, Any],
    render_params: Optional[dict] = None,
) -> str:
    """读取 LLM 配置并调用模型，返回响应纯文本。

    Args:
        runtime: 节点 runtime，用于上下文追踪
        config: 节点 RunnableConfig（含 metadata.llm_cfg）
        render_params: 用于渲染 up 模板的参数字典
    """
    ctx: Context = runtime.context
    _cfg = resolve_llm_cfg(config)

    llm_config: dict = _cfg.get("config", {}) or {}
    sp: str = _cfg.get("sp", "")
    up: str = _cfg.get("up", "")

    up_tpl = Template(up)
    user_prompt = up_tpl.render(**(render_params or {}))

    messages = [
        SystemMessage(content=sp),
        HumanMessage(content=user_prompt),
    ]

    client = LLMClient(ctx=ctx)
    response = client.invoke(
        messages=messages,
        model=llm_config.get("model", "doubao-seed-2-0-pro-260215"),
        temperature=llm_config.get("temperature", 0.1),
        top_p=llm_config.get("top_p", 0.7),
        max_completion_tokens=llm_config.get("max_completion_tokens", 8000),
        thinking=llm_config.get("thinking", "disabled"),
        max_tokens=llm_config.get("max_tokens"),
    )
    content = getattr(response, "content", "")
    return content if isinstance(content, str) else str(content)


def _balanced_first(cleaned: str) -> dict:
    """抓取第一个平衡的 JSON 对象"""
    start = cleaned.find("{")
    if start == -1:
        return {}
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start : i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    return {}
    return {}


def extract_json(text: str) -> dict:
    """从 LLM 响应文本中稳健地提取 JSON 对象（含截断修复）。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned
    cleaned = cleaned.strip()

    # 1) 整体直接解析
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 2) 截断修复：响应可能被 max_tokens 截断导致括号/引号未闭合，
    #    尝试补齐对象 } 与数组 ] 的闭合括号
    for k in range(0, 12):
        for m in range(0, 12):
            try:
                return json.loads(cleaned + "]" * m + "}" * k)
            except Exception:
                continue

    # 3) 退化为抓取第一个内置的完整对象/数组（至少保底取出部分数据）
    return _balanced_first(cleaned)
