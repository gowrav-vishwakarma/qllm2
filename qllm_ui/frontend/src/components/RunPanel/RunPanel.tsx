import { useState, useCallback, useRef, useEffect } from "react";
import { useStore } from "@/store/useStore";
import { api } from "@/utils/api";
import { serializeProject } from "@/utils/serialization";

export default function RunPanel() {
  const { nodes, edges, moduleRegistry, project, training, runs, addRun, updateRun } = useStore();
  const [expanded, setExpanded] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [generating, setGenerating] = useState(false);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const handleGenerate = useCallback(async () => {
    setGenerating(true);
    setLogs((prev) => [...prev, "[INFO] Generating code..."]);
    try {
      const data = serializeProject(nodes, edges, moduleRegistry, project, training);
      const result = await api.generateCode(data);
      setLogs((prev) => [
        ...prev,
        `[OK] Generated ${result.files.length} files in ${result.outputDir}`,
        ...result.files.map((f: string) => `  - ${f}`),
      ]);
    } catch (e: any) {
      setLogs((prev) => [...prev, `[ERROR] ${e.message}`]);
    } finally {
      setGenerating(false);
    }
  }, [nodes, edges, moduleRegistry, project, training]);

  const handleRun = useCallback(async () => {
    setLogs((prev) => [...prev, "[INFO] Starting training run..."]);
    try {
      const data = serializeProject(nodes, edges, moduleRegistry, project, training);
      await api.generateCode(data);
      const { runId } = await api.startRun(project.outputDir);
      addRun({
        id: runId,
        status: "running",
        startedAt: new Date().toISOString(),
        outputDir: project.outputDir,
      });

      const wsUrl = `ws://${window.location.host}/ws/runs/${runId}/logs`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;
      ws.onmessage = (event) => {
        const msg = event.data;
        setLogs((prev) => [...prev, msg]);
        const epochMatch = msg.match(/epoch\s+(\d+)/i);
        const lossMatch = msg.match(/loss[:\s]+([0-9.]+)/i);
        if (epochMatch || lossMatch) {
          updateRun(runId, {
            ...(epochMatch ? { epoch: parseInt(epochMatch[1]) } : {}),
            ...(lossMatch ? { loss: parseFloat(lossMatch[1]) } : {}),
          });
        }
      };
      ws.onclose = () => {
        setLogs((prev) => [...prev, "[INFO] Run completed"]);
        updateRun(runId, { status: "completed" });
      };
      ws.onerror = () => {
        setLogs((prev) => [...prev, "[ERROR] WebSocket connection failed"]);
        updateRun(runId, { status: "failed" });
      };
    } catch (e: any) {
      setLogs((prev) => [...prev, `[ERROR] ${e.message}`]);
    }
  }, [nodes, edges, moduleRegistry, project, training, addRun, updateRun]);

  const handleStop = useCallback(async () => {
    const activeRun = runs.find((r) => r.status === "running");
    if (activeRun) {
      try {
        await api.stopRun(activeRun.id);
        updateRun(activeRun.id, { status: "stopped" });
        wsRef.current?.close();
        setLogs((prev) => [...prev, "[INFO] Run stopped"]);
      } catch (e: any) {
        setLogs((prev) => [...prev, `[ERROR] ${e.message}`]);
      }
    }
  }, [runs, updateRun]);

  const isRunning = runs.some((r) => r.status === "running");

  return (
    <div
      className={`bg-[#1a1a2e] border-t border-white/10 transition-all ${
        expanded ? "h-72" : "h-10"
      }`}
    >
      {/* Header bar */}
      <div className="h-10 flex items-center gap-2 px-4 border-b border-white/10">
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-xs text-gray-400 hover:text-white"
        >
          {expanded ? "▼" : "▲"} Run
        </button>

        <div className="flex-1" />

        {isRunning && (
          <span className="text-xs text-green-400 animate-pulse mr-2">
            ● Running
          </span>
        )}

        <button
          onClick={handleGenerate}
          disabled={generating || nodes.length === 0}
          className="px-3 py-1 text-xs font-medium bg-blue-600 hover:bg-blue-700 disabled:opacity-40 text-white rounded transition-colors"
        >
          Generate Code
        </button>
        <button
          onClick={handleRun}
          disabled={isRunning || nodes.length === 0}
          className="px-3 py-1 text-xs font-medium bg-green-600 hover:bg-green-700 disabled:opacity-40 text-white rounded transition-colors"
        >
          Run Training
        </button>
        {isRunning && (
          <button
            onClick={handleStop}
            className="px-3 py-1 text-xs font-medium bg-red-600 hover:bg-red-700 text-white rounded transition-colors"
          >
            Stop
          </button>
        )}
      </div>

      {/* Logs */}
      {expanded && (
        <div className="h-[calc(100%-2.5rem)] overflow-y-auto p-3 font-mono text-xs text-gray-300">
          {logs.length === 0 && (
            <div className="text-gray-600">
              No output yet. Generate code or start a run.
            </div>
          )}
          {logs.map((line, i) => (
            <div
              key={i}
              className={
                line.startsWith("[ERROR]")
                  ? "text-red-400"
                  : line.startsWith("[OK]")
                  ? "text-green-400"
                  : line.startsWith("[INFO]")
                  ? "text-blue-400"
                  : ""
              }
            >
              {line}
            </div>
          ))}
          <div ref={logsEndRef} />
        </div>
      )}
    </div>
  );
}
