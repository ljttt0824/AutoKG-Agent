import { useEffect, useState } from "react";
import { BookOpen, CheckCircle2, CircleAlert, Route, ShieldCheck } from "lucide-react";
import type { LearningExamples } from "../types";

type LearningView = "answers" | "paths";

export default function LearningAssistant() {
  const [examples, setExamples] = useState<LearningExamples | null>(null);
  const [loadFailed, setLoadFailed] = useState(false);
  const [view, setView] = useState<LearningView>("answers");
  const [selectedAnswerId, setSelectedAnswerId] = useState("");
  const [selectedPathId, setSelectedPathId] = useState("");

  useEffect(() => {
    let active = true;
    fetch(`${import.meta.env.BASE_URL}demo-data/learning-assistant.json`)
      .then((response) => {
        if (!response.ok) throw new Error("学习案例暂时无法读取。");
        return response.json() as Promise<LearningExamples>;
      })
      .then((data) => {
        if (!active) return;
        setExamples(data);
        setSelectedAnswerId(data.answers[0]?.id ?? "");
        setSelectedPathId(data.paths[0]?.id ?? "");
      })
      .catch(() => {
        if (active) setLoadFailed(true);
      });
    return () => { active = false; };
  }, []);

  if (!examples) return <section className="assistant-loading" role={loadFailed ? "alert" : undefined}>{loadFailed ? "学习案例暂时无法读取，请稍后重试。" : "正在载入学习案例"}</section>;

  const activeAnswer = examples.answers.find((item) => item.id === selectedAnswerId) ?? examples.answers[0];
  const activePath = examples.paths.find((item) => item.id === selectedPathId) ?? examples.paths[0];

  return <section className="learning-assistant">
    <div className="assistant-toolbar">
      <div className="assistant-intro"><span className="assistant-live-mark"><ShieldCheck size={16} /></span><div><strong>计算机网络 · 知识图谱学习助手</strong><span>真实本地运行案例 · 脱敏静态展示</span></div></div>
      <div className="assistant-tabs" role="tablist" aria-label="学习案例类型">
        <button type="button" role="tab" aria-selected={view === "answers"} className={view === "answers" ? "selected" : ""} onClick={() => setView("answers")}>问答溯源 <span>{examples.answers.length}</span></button>
        <button type="button" role="tab" aria-selected={view === "paths"} className={view === "paths" ? "selected" : ""} onClick={() => setView("paths")}>学习路径 <span>{examples.paths.length}</span></button>
      </div>
    </div>

    {view === "answers" && activeAnswer && <div className="assistant-layout">
      <nav className="assistant-case-list" aria-label="问答案例">
        {examples.answers.map((item, index) => <button key={item.id} type="button" className={`assistant-case-option ${item.id === activeAnswer.id ? "active" : ""}`} onClick={() => setSelectedAnswerId(item.id)}>
          <span className="assistant-case-index">0{index + 1}</span><span className="assistant-case-question">{item.question}</span><span className={`answer-state ${item.status}`}>{item.status === "pass" ? "通过" : "部分支持"}</span>
        </button>)}
      </nav>
      <article className="answer-detail">
        <div className="assistant-detail-heading"><div><span className="panel-kicker">QUESTION & GROUNDED ANSWER</span><h2>{activeAnswer.question}</h2></div><span className={`answer-status ${activeAnswer.status}`}>{activeAnswer.status === "pass" ? <CheckCircle2 size={15} /> : <CircleAlert size={15} />}{activeAnswer.statusLabel}</span></div>
        <div className="answer-body"><strong>回答摘要</strong><p>{activeAnswer.answer}</p></div>
        <div className="answer-citations"><div className="citation-heading"><BookOpen size={15} /><strong>溯源页码</strong><span>{activeAnswer.pages.length} 处定位</span></div><div className="citation-pages">{activeAnswer.pages.map((page) => <span key={page}>第 {page} 页</span>)}</div></div>
        <div className={`answer-note ${activeAnswer.status}`}><span>{activeAnswer.note}</span></div>
      </article>
    </div>}

    {view === "paths" && activePath && <div className="path-layout">
      <nav className="path-case-list" aria-label="推荐路径">
        {examples.paths.map((item, index) => <button key={item.id} type="button" className={`path-case-option ${item.id === activePath.id ? "active" : ""}`} onClick={() => setSelectedPathId(item.id)}><span className="assistant-case-index">0{index + 1}</span><span><strong>{item.title}</strong><small>{item.nodes.length} 个图谱节点</small></span></button>)}
      </nav>
      <article className="path-detail">
        <div className="assistant-detail-heading"><div><span className="panel-kicker">GRAPH-BASED STUDY PATH</span><h2>{activePath.title}</h2><p>{activePath.summary}</p></div><Route size={20} className="path-heading-icon" /></div>
        <ol className="learning-path-steps">{activePath.nodes.map((node, index) => <li key={`${node.label}-${index}`}><span className="path-step-index">{String(index + 1).padStart(2, "0")}</span><div className="path-step-content"><strong>{node.label}</strong>{node.detail && <span>{node.detail}</span>}<small>依据：第 {node.pages.join("、")} 页</small></div></li>)}</ol>
        <div className="path-note"><CircleAlert size={16} /><p>{activePath.note}</p></div>
      </article>
    </div>}

    <div className="assistant-boundary"><ShieldCheck size={15} /><p>{examples.provenance} {examples.boundary}</p></div>
  </section>;
}
