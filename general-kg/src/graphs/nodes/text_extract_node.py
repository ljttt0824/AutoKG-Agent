"""文本提取与预处理节点（Worker 层 W1/W2）

从上传的源文档（pdf/docx/pptx/txt/图片）中提取文本并做基础清洗，
为下游实体/关系抽取提供干净文本。
"""
from typing import List
import logging
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from coze_coding_utils.runtime_ctx.context import Context
from utils.file.file import File, FileOps
from graphs.state import TextExtractInput, TextExtractOutput


def _clean_text(raw: str) -> str:
    """基础清洗：折叠空白、合并多余空行"""
    lines = [ln.strip() for ln in raw.splitlines()]
    cleaned_lines: List[str] = []
    for ln in lines:
        if not ln:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue
        cleaned_lines.append(ln)
    return "\n".join(cleaned_lines).strip()


def text_extract_node(
    state: TextExtractInput,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> TextExtractOutput:
    """
    title: 文本提取与预处理
    desc: 从源文档中提取文本并做基础清洗，作为实体/关系抽取的统一输入。
    """
    ctx = runtime.context

    parts: List[str] = []
    for file_obj in state.files or []:
        try:
            text = FileOps.extract_text(file_obj)
        except Exception as exc:  # noqa: BLE001
            text = ""
            logging.getLogger("graph").warning("文件文本提取失败: %s", exc)
        if text:
            parts.append(text)

    # 合并联网检索聚合文本（无文档输入时作为主源）
    research_text = state.research_text or ""
    if research_text.strip():
        parts.append("[联网检索资料]\n" + research_text)

    raw = "\n\n".join(parts)
    cleaned = _clean_text(raw)
    return TextExtractOutput(text_content=cleaned, total_chars=len(cleaned))
