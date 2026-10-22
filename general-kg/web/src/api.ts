import type { DemoCase, DemoRun, ExtractionPreview, ModelStatus, WorkspaceMode } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/api/v1";
const STATIC_DEMO = import.meta.env.VITE_STATIC_DEMO === "true";

async function loadFixtures(): Promise<DemoCase[]> {
  const response = await fetch(`${import.meta.env.BASE_URL}demo-data/cases.json`);
  if (!response.ok) throw new Error("Demo fixtures unavailable");
  const networkResponse = await fetch(`${import.meta.env.BASE_URL}demo-data/computer-network-course.json`);
  if (!networkResponse.ok) throw new Error("Computer network demo unavailable");
  const cases = (await response.json()) as DemoCase[];
  const networkCase = (await networkResponse.json()) as DemoCase;
  return [networkCase, ...cases.filter((item) => item.id !== networkCase.id)];
}

export async function loadCases(): Promise<{ cases: DemoCase[]; mode: WorkspaceMode }> {
  if (STATIC_DEMO) return { cases: await loadFixtures(), mode: "static" };
  try {
    const response = await fetch(`${API_BASE}/cases`);
    if (!response.ok) throw new Error("Demo API unavailable");
    const summaries = (await response.json()) as Array<{ id: string }>;
    const cases = await Promise.all(summaries.map((item) => loadCase(item.id, false)));
    const networkCase = (await loadFixtures())[0];
    return { cases: [networkCase, ...cases.filter((item) => item.id !== networkCase.id)], mode: "api" };
  } catch {
    return { cases: await loadFixtures(), mode: "static" };
  }
}

export async function loadCase(caseId: string, allowFallback = true): Promise<DemoCase> {
  try {
    const response = await fetch(`${API_BASE}/cases/${encodeURIComponent(caseId)}`);
    if (!response.ok) throw new Error("Case API unavailable");
    return (await response.json()) as DemoCase;
  } catch (error) {
    if (!allowFallback) throw error;
    const fixture = (await loadFixtures()).find((item) => item.id === caseId);
    if (!fixture) throw new Error("案例不存在");
    return fixture;
  }
}

export async function loadModelStatus(): Promise<ModelStatus> {
  if (STATIC_DEMO) return { provider: "Static demo", model: "read-only fixtures", configured: false };
  try {
    const response = await fetch(`${API_BASE}/config`);
    if (!response.ok) throw new Error("API unavailable");
    return (await response.json()) as ModelStatus;
  } catch {
    return { provider: "DeepSeek", model: "deepseek-chat", configured: false };
  }
}

export async function testModelConnection(): Promise<void> {
  const response = await fetch(`${API_BASE}/model/test`, { method: "POST" });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : "模型连接测试失败");
  }
}

export async function previewFiles(files: File[]): Promise<ExtractionPreview> {
  const form = new FormData();
  files.forEach((file) => form.append("files", file));
  const response = await fetch(`${API_BASE}/extract-preview`, { method: "POST", body: form });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : "文件抽取失败");
  }
  return (await response.json()) as ExtractionPreview;
}

export async function createRun(domain: string, userQuery: string, files: File[]): Promise<DemoRun> {
  const form = new FormData();
  form.set("domain", domain);
  form.set("user_query", userQuery);
  files.forEach((file) => form.append("files", file));

  const response = await fetch(`${API_BASE}/runs`, { method: "POST", body: form });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : "构建任务创建失败");
  }
  return (await response.json()) as DemoRun;
}

export function subscribeRun(runId: string, onRun: (run: DemoRun) => void, onDisconnect: () => void) {
  const stream = new EventSource(`${API_BASE}/runs/${encodeURIComponent(runId)}/events`);
  stream.addEventListener("run", (event) => {
    try {
      onRun(JSON.parse((event as MessageEvent<string>).data) as DemoRun);
    } catch {
      onDisconnect();
    }
  });
  stream.onerror = () => onDisconnect();
  return () => stream.close();
}

export async function pollRun(runId: string): Promise<DemoRun | null> {
  try {
    const response = await fetch(`${API_BASE}/runs/${encodeURIComponent(runId)}`);
    if (!response.ok) return null;
    return (await response.json()) as DemoRun;
  } catch {
    return null;
  }
}
