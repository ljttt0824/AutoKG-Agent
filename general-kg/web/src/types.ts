export type GraphNode = {
  id: string;
  label: string;
  type: string;
  description: string;
  confidence: number | null;
  properties: Record<string, string | number | boolean>;
  evidence_ids: string[];
};

export type GraphEdge = {
  id: string;
  source: string;
  target: string;
  label: string;
  confidence: number | null;
  evidence_ids: string[];
};

export type Evidence = {
  id: string;
  title: string;
  source: string;
  locator: string;
  excerpt: string;
  method: string;
};

export type QualityCheck = {
  name: string;
  value: string;
  state: "pass" | "review" | "info";
  detail: string;
};

export type DemoCase = {
  id: string;
  name: string;
  domain: string;
  description: string;
  provenance: string;
  graph: { nodes: GraphNode[]; edges: GraphEdge[] };
  evidence: Evidence[];
  quality: { checks: QualityCheck[]; notes: string[] };
  history: {
    nodes: number;
    edges: number;
    critic_score: number | null;
    decision: string;
    label: string;
  };
};

export type RunEvent = { stage: string; label: string; status: string; at: string };

export type DemoRun = {
  run_id: string;
  case_id: string;
  user_query: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  current_stage: string;
  stage_label: string;
  progress: number;
  created_at: string;
  completed_at?: string | null;
  timeline: RunEvent[];
  domain?: string;
  source_files?: string[];
  result?: DemoCase | null;
  error_message?: string | null;
};

export type WorkspaceMode = "api" | "static";

export type ModelStatus = {
  provider: string;
  model: string;
  configured: boolean;
};

export type ExtractionPreview = {
  file_count: number;
  chunk_count: number;
  character_count: number;
  chunks: Array<{
    id: string;
    source: string;
    locator: string;
    character_count: number;
    preview: string;
  }>;
};
