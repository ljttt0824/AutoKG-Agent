"""通用领域知识图谱自动构建 Agent —— 状态与数据结构定义

三层多 Agent 架构（Planner-Worker-Critic）的数据契约：
- GlobalState: 工作流全局过程状态
- GraphInput / GraphOutput: 主图出入参
- 各节点独立输入/输出: 遵循最小化范围原则
"""
from typing import Optional, List
from pydantic import BaseModel, Field
from utils.file.file import File


# ============================================================
# 全局状态（工作流内部合并容器的载体）
# ============================================================
class GlobalState(BaseModel):
    """知识图谱构建工作流全局状态"""
    # ---- 输入 ----
    files: List[File] = Field(default=[], description="上传的源文档（pdf/docx/pptx/txt/图片）")
    user_query: str = Field(default="", description="用户自然语言构建需求")
    existing_kg_id: Optional[str] = Field(default=None, description="已存在图谱ID（扩展/修改场景）")
    domain_context: Optional[str] = Field(default=None, description="用户指定领域上下文")

    # ---- Planner 层产物 ----
    task_contract: dict = Field(default={}, description="任务解析 Agent 输出的任务合约")
    execution_plan: dict = Field(default={}, description="流程编排 Agent 输出的执行计划")
    design_proposal: dict = Field(default={}, description="设计确认层产出的领域设计方案（节点类型/关系类型/节点属性）与确认状态")

    # ---- 数据来源（联网检索）----
    research_text: str = Field(default="", description="从联网检索聚合出的研究文本，作为抽取的补充源文本（无文档输入时作为主源）")

    # ---- Worker 层产物 ----
    text_content: str = Field(default="", description="从源文档提取并预处理的纯文本（含联网检索文本合并）")
    total_chars: int = Field(default=0, description="提取文本总字符数")
    entities: List[dict] = Field(default=[], description="W3 抽取的原始实体列表")
    relations: List[dict] = Field(default=[], description="W4 抽取的原始关系列表")
    isolated_entities: List[dict] = Field(default=[], description="W4 识别的孤立实体及建议关系")
    matched_entities: List[dict] = Field(default=[], description="W5 匹配去重后的实体列表")
    matched_relations: List[dict] = Field(default=[], description="W6 去重定向后的关系列表")
    entity_definitions: List[dict] = Field(default=[], description="W7 生成的实体定义")
    relation_definitions: List[dict] = Field(default=[], description="W8 生成的关系定义")
    type_standardization: dict = Field(default={}, description="W9 类型标准化/探索结果")

    # ---- 最终产物 ----
    knowledge_graph: dict = Field(default={}, description="W10 构建的最终知识图谱 JSON")
    quality_report: dict = Field(default={}, description="Critic 质量评估报告")
    critic_decision: str = Field(default="", description="Critic 决策：pass/conditional_pass/rollback/critical_fail")
    retry_count: int = Field(default=0, description="回滚重试次数")
    graph_file_url: str = Field(default="", description="图谱结果文件对象存储 URL")
    decision: str = Field(default="", description="最终决策结论（pass/conditional_pass/rollback/critical_fail）")
    summary: str = Field(default="", description="构建结果摘要")


# ============================================================
# 主图出入参
# ============================================================
class GraphInput(BaseModel):
    """主工作流输入"""
    files: List[File] = Field(default=[], description="上传的源文档（pdf/docx/pptx/txt/图片），可为空则使用 user_query 中的文本")
    user_query: str = Field(..., description="用户自然语言构建需求（如：构建 计算机网络 领域知识图谱）")
    existing_kg_id: Optional[str] = Field(default=None, description="已存在图谱ID（扩展/修改场景）")
    domain_context: Optional[str] = Field(default=None, description="用户指定领域上下文")


class GraphOutput(BaseModel):
    """主工作流输出"""
    knowledge_graph: dict = Field(default={}, description="构建完成的知识图谱 JSON")
    quality_report: dict = Field(default={}, description="Critic 质量评估报告")
    design_proposal: dict = Field(default={}, description="领域设计方案（含节点类型/关系类型/节点属性及确认状态），供用户确认")
    gate_pending: bool = Field(default=False, description="是否因信息不足而暂停构图，等待用户确认/补充")
    confirmation_questions: list = Field(default=[], description="信息不足时返回给用户的追问澄清项")
    decision: str = Field(default="", description="Critic 决策结论")
    summary: str = Field(default="", description="构建结果摘要")


# ============================================================
# Planner 层节点
# ============================================================
class TaskParserInput(BaseModel):
    """任务解析节点输入"""
    user_query: str = Field(..., description="用户自然语言需求")
    files: List[File] = Field(default=[], description="上传的源文档")
    existing_kg_id: Optional[str] = Field(default=None, description="已存在图谱ID")
    domain_context: Optional[str] = Field(default=None, description="领域上下文")


class TaskParserOutput(BaseModel):
    """任务解析节点输出"""
    task_contract: dict = Field(..., description="结构化任务合约 JSON")


class DesignGateInput(BaseModel):
    """设计确认门节点输入：判断是否需暂停向用户追问"""
    design_proposal: dict = Field(default={}, description="设计确认层产出的设计方案与确认状态")
    task_contract: dict = Field(default={}, description="任务合约（含 clarity 信息充分度）")
    files: List[File] = Field(default=[], description="是否已有源文档（影响是否可立即构图）")
    gate_decision: Optional[str] = Field(default=None, description="设计确认门自身决策（供条件分支读取）")


class DesignGateOutput(BaseModel):
    """设计确认门节点输出：分支决策结果"""
    gate_decision: str = Field(default="proceed", description="决策：need_confirmation / proceed")
    gate_reason: str = Field(default="", description="决策依据说明")
    gate_pending: bool = Field(default=False, description="是否暂停构图等待用户确认")
    confirmation_questions: list = Field(default=[], description="返回给用户的追问澄清项")


class FlowOrchestratorInput(BaseModel):
    """流程编排节点输入"""
    task_contract: dict = Field(..., description="任务合约")
    files: List[File] = Field(default=[], description="是否已有源文档（影响数据来源模式判断）")


class FlowOrchestratorOutput(BaseModel):
    """流程编排节点输出"""
    execution_plan: dict = Field(..., description="执行计划 JSON")


class DesignReviewInput(BaseModel):
    """设计确认层节点输入"""
    task_contract: dict = Field(default={}, description="任务合约")
    execution_plan: dict = Field(default={}, description="执行计划")


class DesignReviewOutput(BaseModel):
    """设计确认层节点输出"""
    design_proposal: dict = Field(..., description="领域设计方案：包含 domain / node_types / relation_types / node_attribute_schema / status")


# ============================================================
# Worker 层节点（W1/W2 文本提取）
# ============================================================
class TextExtractInput(BaseModel):
    """文本提取与预处理节点输入"""
    files: List[File] = Field(..., description="源文档")
    research_text: str = Field(default="", description="联网检索聚合文本，与文档文本合并作为抽取源")


class TextExtractOutput(BaseModel):
    """文本提取与预处理节点输出"""
    text_content: str = Field(..., description="清洗后的文档全文（含联网检索文本合并）")
    total_chars: int = Field(..., description="文本总字符数")


# ============================================================
# W3 实体抽取
# ============================================================
class EntityExtractInput(BaseModel):
    """实体抽取节点输入"""
    text_content: str = Field(..., description="预处理后文本")
    task_contract: dict = Field(..., description="任务合约（含期望实体类型）")
    execution_plan: dict = Field(default={}, description="执行计划（含领域自适应本体与领域抽取 prompt）")


class EntityExtractOutput(BaseModel):
    """实体抽取节点输出"""
    entities: List[dict] = Field(..., description="抽取出的实体列表")


# ============================================================
# W4 关系抽取
# ============================================================
class RelationExtractInput(BaseModel):
    """关系抽取节点输入"""
    text_content: str = Field(..., description="预处理后文本")
    entities: List[dict] = Field(..., description="已抽取实体")
    task_contract: dict = Field(..., description="任务合约（含期望关系类型）")
    execution_plan: dict = Field(default={}, description="执行计划（含领域关系类型与领域抽取 prompt）")


class RelationExtractOutput(BaseModel):
    """关系抽取节点输出"""
    relations: List[dict] = Field(..., description="抽取出的关系列表")
    isolated_entities: List[dict] = Field(default=[], description="孤立实体及建议关系")


# ============================================================
# W5 实体匹配
# ============================================================
class EntityMatchInput(BaseModel):
    """实体匹配节点输入"""
    entities: List[dict] = Field(..., description="抽取的实体")


class EntityMatchOutput(BaseModel):
    """实体匹配节点输出"""
    matched_entities: List[dict] = Field(..., description="去重后的实体列表")
    match_stats: dict = Field(default={}, description="匹配统计")


# ============================================================
# W6 关系匹配
# ============================================================
class RelationMatchInput(BaseModel):
    """关系匹配节点输入"""
    relations: List[dict] = Field(..., description="抽取的关系")
    matched_entities: List[dict] = Field(..., description="匹配后实体（用于更新实体ID）")


class RelationMatchOutput(BaseModel):
    """关系匹配节点输出"""
    matched_relations: List[dict] = Field(..., description="去重定向后的关系")


# ============================================================
# W7 实体定义
# ============================================================
class EntityDefineInput(BaseModel):
    """实体定义节点输入"""
    matched_entities: List[dict] = Field(..., description="匹配后实体")
    text_content: str = Field(..., description="原文本上下文")


class EntityDefineOutput(BaseModel):
    """实体定义节点输出"""
    entity_definitions: List[dict] = Field(..., description="带定义的实体列表")


# ============================================================
# W8 关系定义
# ============================================================
class RelationDefineInput(BaseModel):
    """关系定义节点输入"""
    matched_relations: List[dict] = Field(..., description="匹配后关系")


class RelationDefineOutput(BaseModel):
    """关系定义节点输出"""
    relation_definitions: List[dict] = Field(..., description="带定义的关系列表")


# ============================================================
# W9 类型标准化
# ============================================================
class TypeStandardInput(BaseModel):
    """类型标准化节点输入"""
    entity_definitions: List[dict] = Field(..., description="实体定义")
    relation_definitions: List[dict] = Field(..., description="关系定义")
    task_contract: dict = Field(..., description="任务合约")


class TypeStandardOutput(BaseModel):
    """类型标准化节点输出"""
    type_standardization: dict = Field(..., description="类型标准化/探索结果")


# ============================================================
# W10 图谱构建
# ============================================================
class GraphBuildInput(BaseModel):
    """图谱构建节点输入"""
    entity_definitions: List[dict] = Field(..., description="实体定义")
    relation_definitions: List[dict] = Field(..., description="关系定义")
    type_standardization: dict = Field(..., description="类型标准化结果")
    task_contract: dict = Field(..., description="任务合约")
    execution_plan: dict = Field(default={}, description="执行计划（含领域自适应本体 node_types/relation_types/node_attribute_schema）")


class GraphBuildOutput(BaseModel):
    """图谱构建节点输出"""
    knowledge_graph: dict = Field(..., description="最终知识图谱 JSON")


# ============================================================
# Critic 层节点
# ============================================================
class CriticInput(BaseModel):
    """质量评估节点输入"""
    knowledge_graph: dict = Field(..., description="待评估知识图谱")
    task_contract: dict = Field(..., description="任务合约")
    execution_plan: dict = Field(..., description="执行计划（含质检阈值）")


class CriticOutput(BaseModel):
    """质量评估节点输出"""
    quality_report: dict = Field(..., description="质量评估报告")


class CriticDecisionInput(BaseModel):
    """回滚决策节点输入"""
    quality_report: dict = Field(..., description="质量评估报告")
    retry_count: int = Field(default=0, description="当前重试次数")


class CriticDecisionOutput(BaseModel):
    """回滚决策节点输出"""
    critic_decision: str = Field(..., description="Critic 决策：pass/conditional_pass/rollback/critical_fail")
    retry_count: int = Field(..., description="更新后的重试次数")
    final_decision: str = Field(..., description="是否最终放行：pass / retry / pause")


# ============================================================
# 数据来源路由 + 联网检索节点
# ============================================================
class SourceGateInput(BaseModel):
    """数据来源路由节点输入：判断是否需要联网检索补充源文本"""
    files: List[File] = Field(default=[], description="是否已有源文档（无文档则倾向联网检索）")
    execution_plan: dict = Field(default={}, description="执行计划（含 sources 数据来源模式）")


class SourceGateOutput(BaseModel):
    """数据来源路由节点输出"""
    source_decision: str = Field(default="direct", description="数据来源决策：web / direct")
    source_reason: str = Field(default="", description="决策依据说明")


class WebResearchInput(BaseModel):
    """联网检索节点输入"""
    execution_plan: dict = Field(default={}, description="执行计划（从 sources.search_queries 提取检索词）")
    files: List[File] = Field(default=[], description="源文档（用于检测是否已有文档，避免无谓检索）")


class WebResearchOutput(BaseModel):
    """联网检索节点输出"""
    research_text: str = Field(default="", description="从联网检索聚合出的研究文本，作为抽取的补充源文本")
    search_item_count: int = Field(default=0, description="检索到的结果条目总数")


# ============================================================
# 子图调用节点（主图承载）
# ============================================================
class KnowledgeBuildInput(BaseModel):
    """知识图谱构建子图调用节点输入"""
    files: List[File] = Field(default=[], description="源文档（可为空，由联网检索文本补充）")
    task_contract: dict = Field(..., description="任务合约")
    execution_plan: dict = Field(..., description="执行计划")
    research_text: str = Field(default="", description="联网检索聚合文本，与文档文本合并后作为抽取源")


class KnowledgeBuildOutput(BaseModel):
    """知识图谱构建子图调用节点输出"""
    knowledge_graph: dict = Field(..., description="最终知识图谱 JSON")
    quality_report: dict = Field(default={}, description="质量评估报告")
    decision: str = Field(default="", description="最终决策结论")
    graph_file_url: str = Field(default="", description="图谱文件对象存储 URL")
    summary: str = Field(default="", description="构建结果摘要")


# ============================================================
# 子图出入参（loop_graph）
# ============================================================
class SubGraphInput(BaseModel):
    """子图输入"""
    files: List[File] = Field(..., description="源文档")
    task_contract: dict = Field(..., description="任务合约")
    execution_plan: dict = Field(..., description="执行计划")
    research_text: str = Field(default="", description="联网检索聚合文本，与文档文本合并后作为抽取源")


class SubGraphOutput(BaseModel):
    """子图输出"""
    knowledge_graph: dict = Field(..., description="最终知识图谱 JSON")
    quality_report: dict = Field(default={}, description="质量评估报告")
    critic_decision: str = Field(default="", description="Critic 决策")
    graph_file_url: str = Field(default="", description="图谱文件对象存储 URL")


class LoopRouteState(BaseModel):
    """子图回滚路由输入（为条件分支 path 函数使用的独立最小化类型）"""
    critic_decision: str = Field(default="", description="Critic 决策")
    retry_count: int = Field(default=0, description="回滚重试次数")
