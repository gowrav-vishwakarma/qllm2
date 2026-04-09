const BASE = "/api";

async function request<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

export const api = {
  saveProject: (data: unknown) =>
    request<{ path: string }>("/projects/save", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  loadProject: (path: string) =>
    request<unknown>(`/projects/load?path=${encodeURIComponent(path)}`),

  listProjects: () => request<string[]>("/projects/list"),

  inferPorts: (code: string) =>
    request<{
      inputs: { name: string; type: string; optional?: boolean }[];
      outputs: { name: string; type: string }[];
      constructorParams: { name: string; type: string; default?: unknown }[];
    }>("/modules/infer-ports", {
      method: "POST",
      body: JSON.stringify({ code }),
    }),

  generateCode: (data: unknown) =>
    request<{ outputDir: string; files: string[] }>("/codegen/generate", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  startRun: (outputDir: string) =>
    request<{ runId: string }>("/runs/start", {
      method: "POST",
      body: JSON.stringify({ output_dir: outputDir }),
    }),

  stopRun: (runId: string) =>
    request<{ status: string }>(`/runs/${runId}/stop`, { method: "POST" }),

  getRunStatus: (runId: string) =>
    request<{ status: string; epoch?: number; loss?: number }>(`/runs/${runId}/status`),

  listRuns: () =>
    request<{ id: string; status: string; startedAt: string }[]>("/runs/list"),

  getModules: () =>
    request<unknown[]>("/modules/list"),
};
