import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState, type FormEvent, type ReactNode } from "react";
import {
  Activity,
  ArrowDownToLine,
  ArrowRight,
  BookOpen,
  CheckCircle2,
  ChevronRight,
  CircleHelp,
  Clock3,
  FileText,
  LayoutDashboard,
  LoaderCircle,
  MessageSquareText,
  Network,
  Plus,
  Search,
  ShieldCheck,
  Upload,
  X,
} from "lucide-react";
import { colorForType } from "./components/graphColors";
import LearningAssistant from "./components/LearningAssistant";
import { createRun, loadCases, loadModelStatus, pollRun, previewFiles, subscribeRun, testModelConnection } from "./api";
import type { DemoCase, DemoRun, ExtractionPreview, GraphEdge, GraphNode, ModelStatus, QualityCheck, WorkspaceMode } from "./types";

type View = "graph" | "learning" | "cases" | "runs" | "quality";
type Selection = { kind: "node" | "edge"; id: string } | null;
const stages = ["planner", "agentic_search", "worker", "critic", "completed"];
const GraphCanvas = lazy(() => import("./components/GraphCanvas"));

const navItems: Array<{ id: View; label: string; icon: typeof LayoutDashboard }> = [
  { id: "graph", label: "工作台", icon: LayoutDashboard },
  { id: "learning", label: "学习助手", icon: MessageSquareText },
  { id: "cases", label: "案例库", icon: BookOpen },
  { id: "runs", label: "运行记录", icon: Activity },
  { id: "quality", label: "质量报告", icon: ShieldCheck },
];

function statusLabel(status: DemoRun["status"]) {
  if (status === "completed") return "已完成";
  if (status === "failed") return "失败";
  if (status === "cancelled") return "已取消";
  if (status === "running") return "运行中";
  return "排队中";
}

function formatTime(value?: string | null) {
  if (!value) return "刚刚";
  return new Intl.DateTimeFormat("zh-CN", { hour: "2-digit", minute: "2-digit" }).format(new Date(value));
}

function graphEntity(caseData: DemoCase, selection: Selection): GraphNode | GraphEdge | null {
  if (!selection) return null;
  return selection.kind === "node"
    ? caseData.graph.nodes.find((item) => item.id === selection.id) ?? null
    : caseData.graph.edges.find((item) => item.id === selection.id) ?? null;
}

export default function App() {
  const [cases, setCases] = useState<DemoCase[]>([]);
  const [selectedCaseId, setSelectedCaseId] = useState("");
  const [mode, setMode] = useState<WorkspaceMode>("static");
  const [view, setView] = useState<View>("graph");
  const [selection, setSelection] = useState<Selection>(null);
  const [search, setSearch] = useState("");
  const [enabledTypes, setEnabledTypes] = useState<Set<string>>(new Set());
  const [taskOpen, setTaskOpen] = useState(false);
  const [runOpen, setRunOpen] = useState(false);
  const [draftQuery, setDraftQuery] = useState("从上传材料中构建可追溯的知识图谱");
  const [draftDomain, setDraftDomain] = useState("通用领域");
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [run, setRun] = useState<DemoRun | null>(null);
  const [runHistory, setRunHistory] = useState<DemoRun[]>([]);
  const [isStarting, setIsStarting] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const eventStopRef = useRef<(() => void) | null>(null);
  const pollTimerRef = useRef<number | null>(null);

  const activeCase = cases.find((item) => item.id === selectedCaseId) ?? cases[0];

  useEffect(() => {
    let mounted = true;
    loadCases().then(({ cases: loadedCases, mode: loadedMode }) => {
      if (!mounted) return;
      setCases(loadedCases);
      setSelectedCaseId(loadedCases[0]?.id ?? "");
      setMode(loadedMode);
    }).catch(() => {
      if (mounted) setErrorMessage("案例数据暂时无法读取。");
    });
    loadModelStatus().then((status) => {
      if (mounted) setModelStatus(status);
    });
    return () => {
      mounted = false;
      eventStopRef.current?.();
      if (pollTimerRef.current) window.clearInterval(pollTimerRef.current);
    };
  }, []);

  useEffect(() => {
    setSelection(null);
    setSearch("");
    setEnabledTypes(new Set());
  }, [selectedCaseId]);

  const nodeTypes = useMemo(() => [...new Set(activeCase?.graph.nodes.map((node) => node.type) ?? [])], [activeCase]);
  const visibleNodes = useMemo(() => {
    if (!activeCase) return [];
    const normalizedSearch = search.trim().toLocaleLowerCase();
    return activeCase.graph.nodes.filter((node) => {
      const typeMatches = enabledTypes.size === 0 || enabledTypes.has(node.type);
      const queryMatches = !normalizedSearch || `${node.label} ${node.type} ${node.description}`.toLocaleLowerCase().includes(normalizedSearch);
      return typeMatches && queryMatches;
    });
  }, [activeCase, enabledTypes, search]);
  const visibleNodeIds = useMemo(() => new Set(visibleNodes.map((node) => node.id)), [visibleNodes]);
  const visibleEdges = useMemo(
    () => activeCase?.graph.edges.filter((edge) => visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target)) ?? [],
    [activeCase, visibleNodeIds],
  );

  const entity = activeCase ? graphEntity(activeCase, selection) : null;
  const entityEvidence = activeCase && entity
    ? activeCase.evidence.filter((evidence) => entity.evidence_ids.includes(evidence.id))
    : [];

  const updateRun = useCallback((nextRun: DemoRun) => {
    setRun(nextRun);
    setRunHistory((current) => {
      const existing = current.some((item) => item.run_id === nextRun.run_id);
      return existing
        ? current.map((item) => item.run_id === nextRun.run_id ? nextRun : item)
        : [nextRun, ...current];
    });
    if (nextRun.result) {
      setCases((current) => [nextRun.result!, ...current.filter((item) => item.id !== nextRun.result!.id)]);
      setSelectedCaseId(nextRun.result.id);
      setView("graph");
    }
    if (["completed", "failed", "cancelled"].includes(nextRun.status)) {
      eventStopRef.current?.();
      eventStopRef.current = null;
      if (pollTimerRef.current) window.clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  }, []);

  const handleStartRun = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!activeCase || draftQuery.trim().length < 3 || !selectedFiles.length) return;
    setIsStarting(true);
    setErrorMessage("");
    try {
      const apiRun = await createRun(draftDomain.trim() || "通用领域", draftQuery.trim(), selectedFiles);
      setTaskOpen(false);
      setRunOpen(true);
      setMode("api");
      updateRun(apiRun);
      eventStopRef.current?.();
      eventStopRef.current = subscribeRun(apiRun.run_id, updateRun, () => {
        eventStopRef.current?.();
        eventStopRef.current = null;
        if (pollTimerRef.current) return;
        pollTimerRef.current = window.setInterval(async () => {
          const latest = await pollRun(apiRun.run_id);
          if (latest) updateRun(latest);
        }, 850);
      });
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "构建任务创建失败");
    } finally {
      setIsStarting(false);
    }
  };

  const openNewRun = () => {
    setDraftDomain(activeCase.domain === "电子商务" || activeCase.domain === "实验教学" || activeCase.domain === "项目管理" ? activeCase.domain : "通用领域");
    setDraftQuery("从上传材料中构建可追溯的知识图谱");
    setSelectedFiles([]);
    setErrorMessage("");
    setTaskOpen(true);
  };

  const toggleType = (type: string) => {
    setEnabledTypes((current) => {
      const next = new Set(current);
      if (next.has(type)) next.delete(type);
      else next.add(type);
      return next;
    });
  };

  const exportCase = () => {
    if (!activeCase) return;
    const blob = new Blob([JSON.stringify(activeCase.graph, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${activeCase.id}-graph.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  if (!activeCase) {
    return <div className="loading-screen"><div className="loader-mark"><Network size={23} /></div><span>正在载入工作区</span></div>;
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-row">
          <div className="brand-mark"><Network size={20} strokeWidth={2.2} /></div>
          <div className="brand-text"><strong>AutoKG Agent</strong><span>知识图谱工作台</span></div>
        </div>
        <div className="workspace-label">工作空间</div>
        <div className="workspace-switch"><div className="workspace-avatar">{activeCase.id === "computer-network-course" ? "网" : "通"}</div><div><strong>{activeCase.id === "computer-network-course" ? "计算机网络知识库" : "通用领域知识库"}</strong><span>{mode === "static" ? "只读案例空间" : "个人工作空间"}</span></div><ChevronRight size={15} /></div>
        <nav className="side-nav" aria-label="主导航">
          <span className="nav-caption">项目</span>
          {navItems.map((item) => {
            const Icon = item.icon;
            return <button key={item.id} className={`nav-item ${view === item.id ? "active" : ""}`} onClick={() => setView(item.id)} type="button"><Icon size={17} /><span>{item.label}</span>{item.id === "runs" && runHistory.length > 0 && <b>{runHistory.length}</b>}</button>;
          })}
        </nav>
        <div className="sidebar-bottom">
          <div className="local-mode"><span className={`status-dot ${mode === "static" ? "" : modelStatus?.configured ? "" : "offline"}`} /><div><strong>{mode === "static" ? "静态演示模式" : modelStatus?.configured ? "DeepSeek 已配置" : "等待模型配置"}</strong><span>{mode === "static" ? "交互式案例 · 不调用 API" : `本地 API · ${modelStatus?.model ?? "deepseek-chat"}`}</span></div></div>
          <div className="profile-row"><div className="profile-avatar">KG</div><div><strong>项目演示</strong><span>作品集空间</span></div><span title="关于此演示" aria-label="关于此演示"><CircleHelp size={17} className="muted-icon" /></span></div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumbs"><span>工作空间</span><ChevronRight size={14} /><strong>{navItems.find((item) => item.id === view)?.label ?? "工作台"}</strong></div>
          <div className="topbar-right"><span className={`demo-badge ${mode === "static" || modelStatus?.configured ? "configured" : "needs-config"}`}><span className="status-dot" />{mode === "static" ? "静态演示 · 不连接模型" : modelStatus?.configured ? `${modelStatus.provider} · ${modelStatus.model}` : "模型未配置"}</span><button className="icon-button top-help" title="查看演示案例" aria-label="查看演示案例" type="button" onClick={() => setView("cases")}><CircleHelp size={17} /></button></div>
        </header>

        <div className="page-content">
          <div className="page-heading">
            <div>
              <div className="eyebrow">{view === "learning" ? "KG-RAG / COMPUTER NETWORKS" : activeCase.id === "computer-network-course" ? "SUBJECT KNOWLEDGE GRAPH / COMPUTER NETWORKS" : "AUTOKG / GENERAL KNOWLEDGE GRAPH"}</div>
              <h1>{view === "graph" ? activeCase.id === "computer-network-course" ? "计算机网络课程知识图谱" : "知识图谱工作台" : view === "learning" ? "知识图谱学习助手" : view === "cases" ? "案例库" : view === "runs" ? "运行记录" : "质量报告"}</h1>
              <p>{view === "graph" ? `${activeCase.name} · ${activeCase.domain}` : view === "learning" ? "课程问答溯源与图谱学习路径推荐" : view === "cases" ? "项目案例与本地生成结果" : view === "runs" ? "当前会话的模型构建任务" : `${activeCase.name} · 质量检查与来源覆盖`}</p>
            </div>
            <div className="heading-actions">
              {view === "graph" && <button className="secondary-button" type="button" onClick={exportCase}><ArrowDownToLine size={16} />导出 JSON</button>}
              {mode === "api" && <button className="primary-button" type="button" onClick={openNewRun}><Plus size={17} />新建构建任务</button>}
            </div>
          </div>

          {view === "graph" && <div className="summary-strip">
            <Metric icon={<Network size={17} />} label="展示节点" value={activeCase.graph.nodes.length.toString()} tone="teal" />
            <Metric icon={<Activity size={17} />} label="展示关系" value={activeCase.graph.edges.length.toString()} tone="blue" />
            <Metric icon={<FileText size={17} />} label="来源证据" value={activeCase.evidence.length.toString()} tone="amber" />
            <Metric icon={<ShieldCheck size={17} />} label="Critic 自评" value={activeCase.history.critic_score?.toFixed(2) ?? "—"} tone="green" footnote="非人工金标准" />
          </div>}

          {errorMessage && <div className="error-banner" role="alert">{errorMessage}<button type="button" className="icon-button" onClick={() => setErrorMessage("")} aria-label="关闭"><X size={15} /></button></div>}

          {view === "graph" && <GraphView
            caseData={activeCase}
            nodeTypes={nodeTypes}
            visibleNodes={visibleNodes}
            visibleEdges={visibleEdges}
            enabledTypes={enabledTypes}
            search={search}
            setSearch={setSearch}
            toggleType={toggleType}
            selection={selection}
            setSelection={setSelection}
            entity={entity}
            entityEvidence={entityEvidence}
          />}
          {view === "learning" && <LearningAssistant />}
          {view === "cases" && <CaseLibrary cases={cases} selectedCaseId={activeCase.id} onSelect={(id) => { setSelectedCaseId(id); setView("graph"); }} />}
          {view === "quality" && <QualityView caseData={activeCase} />}
          {view === "runs" && <RunHistory runs={runHistory} onOpen={(item) => { setRun(item); setSelectedCaseId(item.case_id); setRunOpen(true); }} />}

          <footer className="page-footer"><span>AutoKG Agent · Planner / Worker / Critic</span><span>{mode === "static" ? "静态案例展示 · 不上传文件、不调用模型" : "案例为脱敏数据；新任务使用真实模型生成"}</span></footer>
        </div>
      </main>

      {taskOpen && <TaskDrawer
        caseData={activeCase}
        query={draftQuery}
        setQuery={setDraftQuery}
        domain={draftDomain}
        setDomain={setDraftDomain}
        files={selectedFiles}
        setFiles={setSelectedFiles}
        modelStatus={modelStatus}
        isStarting={isStarting}
        onClose={() => setTaskOpen(false)}
        onSubmit={handleStartRun}
      />}
      {runOpen && run && <RunDrawer run={run} caseData={cases.find((item) => item.id === run.case_id) ?? activeCase} mode={mode} onClose={() => setRunOpen(false)} />}
    </div>
  );
}

function Metric({ icon, label, value, tone, footnote }: { icon: ReactNode; label: string; value: string; tone: string; footnote?: string }) {
  return <div className="metric"><div className={`metric-icon ${tone}`}>{icon}</div><div className="metric-copy"><span>{label}</span><strong>{value}</strong>{footnote && <small>{footnote}</small>}</div></div>;
}

function GraphView({
  caseData, nodeTypes, visibleNodes, visibleEdges, enabledTypes, search, setSearch, toggleType, selection, setSelection, entity, entityEvidence,
}: {
  caseData: DemoCase;
  nodeTypes: string[];
  visibleNodes: GraphNode[];
  visibleEdges: GraphEdge[];
  enabledTypes: Set<string>;
  search: string;
  setSearch: (value: string) => void;
  toggleType: (type: string) => void;
  selection: Selection;
  setSelection: (selection: Selection) => void;
  entity: GraphNode | GraphEdge | null;
  entityEvidence: DemoCase["evidence"];
}) {
  const isNode = entity && "type" in entity;
  const selectedNode = isNode ? entity as GraphNode : null;
  const selectedEdge = entity && "source" in entity ? entity as GraphEdge : null;
  const sourceNode = selectedEdge ? caseData.graph.nodes.find((node) => node.id === selectedEdge.source) : null;
  const targetNode = selectedEdge ? caseData.graph.nodes.find((node) => node.id === selectedEdge.target) : null;

  return <section className="graph-workspace">
    <div className="graph-main-panel">
      <div className="graph-panel-header">
        <div><strong>关系网络</strong><span>交互式图谱视图</span></div>
        <div className="graph-header-controls">
          <label className="search-field"><Search size={15} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="搜索实体或类型" aria-label="搜索实体" /></label>
          <span className="graph-count">{visibleNodes.length} 节点 · {visibleEdges.length} 关系</span>
        </div>
      </div>
      <div className="type-filter-row" aria-label="实体类型过滤">
        <span className="filter-label">类型</span>
        {nodeTypes.map((type) => <button key={type} className={`type-filter ${enabledTypes.size === 0 || enabledTypes.has(type) ? "selected" : "dimmed"}`} type="button" aria-pressed={enabledTypes.size === 0 || enabledTypes.has(type)} onClick={() => toggleType(type)}><i style={{ backgroundColor: colorForType(type) }} />{type}</button>)}
        {enabledTypes.size > 0 && <button className="clear-filter" type="button" onClick={() => nodeTypes.forEach((type) => enabledTypes.has(type) && toggleType(type))}>清除</button>}
      </div>
      <Suspense fallback={<div className="graph-loading"><LoaderCircle size={18} className="spin" />正在载入图谱</div>}>
        <GraphCanvas nodes={visibleNodes} edges={visibleEdges} selection={selection} onSelect={setSelection} />
      </Suspense>
      <div className="graph-status-row"><span><i className="status-dot" />来源可追溯</span><span>拖拽节点调整布局 · 滚轮缩放</span><span>{caseData.provenance}</span></div>
    </div>
    <aside className="inspector-panel">
      <div className="inspector-heading"><div><span className="panel-kicker">检查器</span><strong>{selectedNode ? "实体详情" : selectedEdge ? "关系详情" : "图谱概览"}</strong></div>{selection && <button className="icon-button" type="button" title="清除选择" aria-label="清除选择" onClick={() => setSelection(null)}><X size={16} /></button>}</div>
      {selectedNode ? <>
        <div className="entity-title"><span className="entity-type-dot" style={{ backgroundColor: colorForType(selectedNode.type) }} /><div><strong>{selectedNode.label}</strong><span>{selectedNode.type}</span></div></div>
        <p className="entity-description">{selectedNode.description}</p>
        <div className="inspector-section"><div className="section-title">属性 <span>{Object.keys(selectedNode.properties).length}</span></div><div className="property-list">{Object.entries(selectedNode.properties).map(([key, value]) => <div className="property-row" key={key}><span>{key}</span><strong>{String(value)}</strong></div>)}</div></div>
        <div className="inspector-section"><div className="section-title">来源证据 <span>{entityEvidence.length}</span></div><EvidenceList items={entityEvidence} /></div>
        <div className="confidence-row"><span>抽取置信度</span><strong>{selectedNode.confidence == null ? "未记录" : `${Math.round(selectedNode.confidence * 100)}%`}</strong></div>
      </> : selectedEdge ? <>
        <div className="relation-title"><span className="relation-endpoint">{sourceNode?.label ?? selectedEdge.source}</span><ArrowRight size={15} /><span className="relation-endpoint">{targetNode?.label ?? selectedEdge.target}</span></div>
        <div className="relation-label">{selectedEdge.label}</div>
        <div className="confidence-row"><span>关系置信度</span><strong>{selectedEdge.confidence == null ? "未记录" : `${Math.round(selectedEdge.confidence * 100)}%`}</strong></div>
        <div className="inspector-section"><div className="section-title">来源证据 <span>{entityEvidence.length}</span></div><EvidenceList items={entityEvidence} /></div>
      </> : <>
        <div className="overview-graphic"><div className="overview-ring"><Network size={31} /></div><div className="overview-caption"><strong>{caseData.name}</strong><span>{caseData.domain}</span></div></div>
        <p className="overview-description">{caseData.description}</p>
        <div className="inspector-section"><div className="section-title">实体类型 <span>{nodeTypes.length}</span></div><div className="legend-list">{nodeTypes.map((type) => <div className="legend-item" key={type}><i style={{ backgroundColor: colorForType(type) }} /><span>{type}</span><b>{caseData.graph.nodes.filter((node) => node.type === type).length}</b></div>)}</div></div>
        <div className="provenance-note"><FileText size={15} /><span>{caseData.provenance}</span></div>
      </>}
    </aside>
  </section>;
}

function EvidenceList({ items }: { items: DemoCase["evidence"] }) {
  if (!items.length) return <div className="empty-inline">暂无关联证据</div>;
  return <div className="evidence-list">{items.map((item) => <article className="evidence-item" key={item.id}><div className="evidence-meta"><b>{item.id}</b><span>{item.locator}</span></div><strong>{item.title}</strong><p>“{item.excerpt}”</p><small>{item.source} · {item.method}</small></article>)}</div>;
}

function QualityView({ caseData }: { caseData: DemoCase }) {
  return <section className="quality-layout">
    <div className="quality-main-panel">
      <div className="section-heading"><div><span className="panel-kicker">QUALITY GATE</span><h2>质量检查</h2></div><span className="review-status"><ShieldCheck size={15} />含人工复核项</span></div>
      <div className="quality-table"><div className="quality-table-head"><span>检查项</span><span>当前结果</span><span>状态</span></div>{caseData.quality.checks.map((check) => <QualityRow key={check.name} check={check} />)}</div>
      <div className="quality-notes"><strong>评估边界</strong>{caseData.quality.notes.map((note) => <p key={note}>{note}</p>)}</div>
    </div>
    <aside className="quality-side-panel">
      <div className="history-score-label">{caseData.history.label}</div>
      <div className="history-score"><strong>{caseData.history.critic_score?.toFixed(2) ?? "—"}</strong><span>Critic</span></div>
      <p>{caseData.id.startsWith("generated_") ? "本次 Critic 模型自评，不代表人工金标准准确率。" : "历史单次运行记录，不代表人工金标准准确率。"}</p>
      <div className="history-stats"><div><span>历史节点</span><strong>{caseData.history.nodes}</strong></div><div><span>历史关系</span><strong>{caseData.history.edges}</strong></div></div>
      <div className="result-decision"><span>决策</span><strong>{caseData.history.decision === "conditional_pass" ? "条件通过" : caseData.history.decision}</strong></div>
    </aside>
  </section>;
}

function QualityRow({ check }: { check: QualityCheck }) {
  const icon = check.state === "pass" ? <CheckCircle2 size={16} /> : check.state === "review" ? <Clock3 size={16} /> : <CircleHelp size={16} />;
  const label = check.state === "pass" ? "通过" : check.state === "review" ? "待复核" : "说明";
  return <div className="quality-row"><div><strong>{check.name}</strong><small>{check.detail}</small></div><b>{check.value}</b><span className={`check-badge ${check.state}`}>{icon}{label}</span></div>;
}

function CaseLibrary({ cases, selectedCaseId, onSelect }: { cases: DemoCase[]; selectedCaseId: string; onSelect: (id: string) => void }) {
  return <section className="case-library"><div className="section-heading"><div><span className="panel-kicker">IMPLEMENTED EXAMPLES</span><h2>已实现案例</h2></div><span className="case-count">{cases.length} 个案例</span></div><div className="case-grid">{cases.map((item, index) => <button type="button" className={`case-card ${item.id === selectedCaseId ? "active" : ""}`} key={item.id} onClick={() => onSelect(item.id)}><div className={`case-symbol case-symbol-${index}`}><Network size={19} /></div><div className="case-domain">{item.domain}</div><h3>{item.name}</h3><p>{item.description}</p><div className="case-card-footer"><span>{item.graph.nodes.length} 节点</span><span>{item.graph.edges.length} 关系</span><span>Critic {item.history.critic_score?.toFixed(2) ?? "—"}</span><ChevronRight size={16} /></div></button>)}</div><div className="case-footnote">展示内容为精选子图；模型评分与人工复核状态按案例分别标注。</div></section>;
}

function RunHistory({ runs, onOpen }: { runs: DemoRun[]; onOpen: (run: DemoRun) => void }) {
  if (!runs.length) return <section className="empty-state"><div className="empty-icon"><Activity size={24} /></div><h2>暂无运行记录</h2><p>创建一次演示任务后，运行轨迹会显示在这里。</p></section>;
  return <section className="run-history"><div className="section-heading"><div><span className="panel-kicker">SESSION RUNS</span><h2>当前会话</h2></div><span className="case-count">{runs.length} 条记录</span></div><div className="run-table"><div className="run-table-head"><span>任务</span><span>案例</span><span>状态</span><span>创建时间</span><span /></div>{runs.map((item) => <button type="button" className="run-row" key={item.run_id} onClick={() => onOpen(item)}><div><strong>{item.user_query}</strong><small>{item.run_id}</small></div><span>{item.case_id}</span><span className={`run-status ${item.status}`}><i />{statusLabel(item.status)}</span><span>{formatTime(item.created_at)}</span><ChevronRight size={16} /></button>)}</div></section>;
}

function TaskDrawer({ caseData, query, setQuery, domain, setDomain, files, setFiles, modelStatus, isStarting, onClose, onSubmit }: {
  caseData: DemoCase;
  query: string;
  setQuery: (value: string) => void;
  domain: string;
  setDomain: (value: string) => void;
  files: File[];
  setFiles: (files: File[]) => void;
  modelStatus: ModelStatus | null;
  isStarting: boolean;
  onClose: () => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
}) {
  const [connectionState, setConnectionState] = useState<"idle" | "checking" | "ok" | "error">("idle");
  const [connectionMessage, setConnectionMessage] = useState("");
  const [preview, setPreview] = useState<ExtractionPreview | null>(null);
  const [previewState, setPreviewState] = useState<"idle" | "extracting" | "error">("idle");
  const [previewError, setPreviewError] = useState("");
  const [fileWarning, setFileWarning] = useState("");
  const totalFileBytes = files.reduce((total, file) => total + file.size, 0);
  const runFilePreview = async () => {
    if (!files.length) return;
    setPreviewState("extracting");
    setPreviewError("");
    try {
      setPreview(await previewFiles(files));
      setPreviewState("idle");
    } catch (error) {
      setPreview(null);
      setPreviewState("error");
      setPreviewError(error instanceof Error ? error.message : "文件抽取失败");
    }
  };
  const runModelCheck = async () => {
    setConnectionState("checking");
    setConnectionMessage("");
    try {
      await testModelConnection();
      setConnectionState("ok");
      setConnectionMessage("模型连接成功");
    } catch (error) {
      setConnectionState("error");
      setConnectionMessage(error instanceof Error ? error.message : "模型连接测试失败");
    }
  };

  return <div className="drawer-scrim" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
    <section className="drawer" role="dialog" aria-modal="true" aria-labelledby="task-drawer-title">
      <header className="drawer-header"><div><span className="panel-kicker">DOCUMENT TO GRAPH</span><h2 id="task-drawer-title">新建构建任务</h2></div><button className="icon-button" type="button" onClick={onClose} aria-label="关闭"><X size={18} /></button></header>
      <form className="drawer-form" onSubmit={onSubmit}>
        <label className="field-label" htmlFor="run-domain">目标领域<span>Planner 将据此生成 schema</span></label>
        <input id="run-domain" className="text-field" value={domain} onChange={(event) => setDomain(event.target.value)} maxLength={120} required placeholder={caseData.domain || "通用领域"} />
        <label className="field-label" htmlFor="run-query">构建目标</label>
        <textarea id="run-query" value={query} onChange={(event) => setQuery(event.target.value)} rows={4} maxLength={2_000} placeholder="描述希望从材料中构建的实体和关系" required minLength={3} />
        <div className="field-counter">{query.length} / 2,000</div>
        <label className="field-label" htmlFor="source-files">来源文件<span>最多 10 个，合计 25 MB</span></label>
        <label className="file-picker" htmlFor="source-files"><Upload size={17} /><span>选择 PDF、DOCX、PPTX、XLSX、CSV、TXT、MD 或 JSON 文件</span><input id="source-files" type="file" multiple accept=".pdf,.docx,.pptx,.csv,.txt,.md,.json,.xlsx" onChange={(event) => { const nextFiles = [...files, ...Array.from(event.target.files ?? [])]; setFileWarning(nextFiles.length > 10 ? "单次最多选择 10 个文件。" : ""); setFiles(nextFiles.slice(0, 10)); setPreview(null); setPreviewError(""); event.target.value = ""; }} /></label>
        {files.length > 0 && <ul className="selected-files">{files.map((file, index) => <li key={`${file.name}-${file.lastModified}-${index}`}><div><strong>{file.name}</strong><span>{(file.size / 1024).toFixed(0)} KB</span></div><button className="icon-button" type="button" aria-label={`移除 ${file.name}`} onClick={() => { setFiles(files.filter((_, fileIndex) => fileIndex !== index)); setFileWarning(""); setPreview(null); }}><X size={15} /></button></li>)}</ul>}
        {fileWarning && <div className="connection-message error" role="alert">{fileWarning}</div>}
        {totalFileBytes > 25 * 1024 * 1024 && <div className="connection-message error" role="alert">文件合计超过 25 MB，请移除部分文件。</div>}
        <div className="preview-actions"><span>本机文本解析，不调用模型</span><button type="button" className="secondary-button" onClick={runFilePreview} disabled={!files.length || previewState === "extracting" || Boolean(fileWarning) || totalFileBytes > 25 * 1024 * 1024}>{previewState === "extracting" ? <LoaderCircle size={14} className="spin" /> : <FileText size={14} />}{previewState === "extracting" ? "抽取中" : "预览抽取文本"}</button></div>
        {previewError && <div className="connection-message error" role="alert">{previewError}</div>}
        {preview && <section className="extract-preview" aria-label="文件抽取预览"><div className="extract-preview-summary">{preview.file_count} 个文件 · {preview.chunk_count} 个文本块 · {preview.character_count.toLocaleString()} 字符</div>{preview.chunks.map((chunk) => <article key={chunk.id}><div><strong>{chunk.source}</strong><span>{chunk.locator} · {chunk.character_count.toLocaleString()} 字符</span></div><pre>{chunk.preview}{chunk.character_count > chunk.preview.length ? "…" : ""}</pre></article>)}</section>}
        <div className="model-config-row"><div><strong>{modelStatus?.configured ? `${modelStatus.provider} · ${modelStatus.model}` : "DeepSeek 尚未配置"}</strong><span>密钥只由本地后端读取</span></div><button type="button" className="secondary-button" onClick={runModelCheck} disabled={connectionState === "checking"}>{connectionState === "checking" ? <LoaderCircle size={14} className="spin" /> : <Activity size={14} />}{connectionState === "checking" ? "测试中" : "测试连接"}</button></div>
        {!modelStatus?.configured && <div className="connection-message setup">复制 general-kg/.env.example 为 .env，填写 AUTOKG_LLM_API_KEY 后重启后端。</div>}
        {connectionMessage && <div className={`connection-message ${connectionState}`} role="status">{connectionMessage}</div>}
        <div className="privacy-disclosure"><ShieldCheck size={16} /><div><strong>文件处理与隐私</strong><p>文件在本机解析，文本片段会发送至 DeepSeek API。不要上传未经授权的个人信息或机密材料；单次最多处理 100,000 字符。</p></div></div>
        <div className="drawer-bottom"><button type="button" className="secondary-button full-width" onClick={onClose}>取消</button><button type="submit" className="primary-button full-width" disabled={isStarting || query.trim().length < 3 || files.length === 0 || !modelStatus?.configured || Boolean(fileWarning) || totalFileBytes > 25 * 1024 * 1024}>{isStarting ? <LoaderCircle className="spin" size={16} /> : <ArrowRight size={16} />}{isStarting ? "创建任务" : "开始构建"}</button></div>
      </form>
    </section>
  </div>;
}

function RunDrawer({ run, caseData, mode, onClose }: { run: DemoRun; caseData: DemoCase; mode: WorkspaceMode; onClose: () => void }) {
  const currentIndex = stages.indexOf(run.current_stage);
  return <div className="drawer-scrim" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
    <section className="drawer" role="dialog" aria-modal="true" aria-labelledby="run-drawer-title">
      <header className="drawer-header"><div><span className="panel-kicker">{run.run_id}</span><h2 id="run-drawer-title">构建过程</h2></div><button className="icon-button" type="button" onClick={onClose} aria-label="关闭"><X size={18} /></button></header>
      <div className="run-drawer-content">
        <div className="run-summary"><div><span>领域</span><strong>{run.domain ?? caseData.domain}</strong></div><div><span>目标</span><strong>{run.user_query}</strong></div><div><span>模型</span><strong>{mode === "api" ? "DeepSeek · 本地后端" : "未连接"}</strong></div>{run.source_files?.length ? <div><span>来源文件</span><strong>{run.source_files.join("、")}</strong></div> : null}</div>
        <div className="progress-header"><span>{run.stage_label}</span><strong>{Math.round(run.progress * 100)}%</strong></div><div className="progress-track"><i style={{ width: `${Math.round(run.progress * 100)}%` }} /></div>
        <ol className="run-timeline">{[
          ["planner", "Planner", "生成领域 schema"],
          ["agentic_search", "文件解析", "抽取本地文本证据"],
          ["worker", "Worker", "分块抽取并逐字核验引用"],
          ["critic", "Critic", "检查图结构与证据覆盖"],
          ["completed", "完成", "展示本次模型生成图谱"],
        ].map(([stage, title, detail]) => {
          const stageIndex = stages.indexOf(stage);
          const done = currentIndex > stageIndex || run.status === "completed";
          const active = currentIndex === stageIndex && run.status === "running";
          return <li className={`${done ? "done" : ""} ${active ? "current" : ""}`} key={stage}><span className="timeline-marker">{done ? <CheckCircle2 size={17} /> : active ? <LoaderCircle size={17} className="spin" /> : <span />}</span><div><strong>{title}</strong><small>{detail}</small></div><em>{done ? "完成" : active ? "处理中" : "等待"}</em></li>;
        })}</ol>
        {run.status === "failed" && <div className="run-error" role="alert">{run.error_message ?? "构建失败，请稍后重试。"}</div>}
        {run.status === "completed" && run.result && <div className="run-result-summary"><strong>本次生成结果</strong><span>{run.result.graph.nodes.length} 个实体 · {run.result.graph.edges.length} 条关系 · {run.result.evidence.length} 条证据</span><span>Critic 自评 {run.result.history.critic_score?.toFixed(2) ?? "待复核"} · 需人工核验</span></div>}
        <div className="privacy-disclosure"><ShieldCheck size={16} /><div><strong>质量与数据边界</strong><p>引用已与上传原文逐字核验；模型评分不能代替专家复核。文本片段已发送到 DeepSeek API。</p></div></div>
      </div>
      <footer className="drawer-footer"><button type="button" className="secondary-button full-width" onClick={onClose}>{run.status === "completed" ? "返回工作台" : "收起面板"}</button>{run.status === "completed" && run.result && <button type="button" className="primary-button full-width" onClick={onClose}>查看生成图谱<ArrowRight size={16} /></button>}</footer>
    </section>
  </div>;
}
