import { useCallback, useRef } from "react";
import { useStore } from "@/store/useStore";
import { api } from "@/utils/api";
import { serializeProject, deserializeProject } from "@/utils/serialization";
import { builtinModules } from "@/utils/builtinModules";
import type { SerializedProject } from "@/types";

export default function TopBar() {
  const store = useStore();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSave = useCallback(async () => {
    const data = serializeProject(
      store.nodes,
      store.edges,
      store.moduleRegistry,
      store.project,
      store.training
    );
    try {
      const result = await api.saveProject(data);
      alert(`Saved to ${result.path}`);
    } catch {
      const blob = new Blob([JSON.stringify(data, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${store.project.name}.json`;
      a.click();
      URL.revokeObjectURL(url);
    }
  }, [store]);

  const handleLoad = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const onFileSelected = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;
      const text = await file.text();
      try {
        const data = JSON.parse(text) as SerializedProject;
        const builtins = store.moduleRegistry.filter((m) => !m.isCustom);
        const result = deserializeProject(data, builtins);
        for (const mod of result.customModules) {
          store.addCustomModule(mod);
        }
        store.setProject(result.project);
        store.setTraining(result.training);

        useStore.setState({
          nodes: result.nodes,
          edges: result.edges,
        });
        store.autoLayout();
      } catch (e: any) {
        alert(`Failed to load: ${e.message}`);
      }
      event.target.value = "";
    },
    [store]
  );

  const handleNew = useCallback(() => {
    if (!confirm("Create new project? Unsaved changes will be lost.")) return;
    useStore.setState({
      nodes: [],
      edges: [],
      selectedNodeId: null,
      nodeCounter: 0,
    });
    store.setProject({
      name: "new_experiment",
      outputDir: "./experiments/new_experiment",
      checkpointDir: "./experiments/new_experiment/checkpoints",
      logDir: "./experiments/new_experiment/logs",
    });
  }, [store]);

  const nodeCount = store.nodes.length;
  const edgeCount = store.edges.length;

  return (
    <div className="h-11 bg-[#1a1a2e] border-b border-white/10 flex items-center px-4 gap-4 shrink-0">
      <div className="text-sm font-bold text-indigo-400 tracking-wide mr-2">
        QLLM Builder
      </div>

      <button
        onClick={handleNew}
        className="px-2.5 py-1 text-xs text-gray-300 hover:text-white hover:bg-white/5 rounded transition-colors"
      >
        New
      </button>
      <button
        onClick={handleSave}
        className="px-2.5 py-1 text-xs text-gray-300 hover:text-white hover:bg-white/5 rounded transition-colors"
      >
        Save
      </button>
      <button
        onClick={handleLoad}
        className="px-2.5 py-1 text-xs text-gray-300 hover:text-white hover:bg-white/5 rounded transition-colors"
      >
        Load
      </button>
      <input
        ref={fileInputRef}
        type="file"
        accept=".json"
        className="hidden"
        onChange={onFileSelected}
      />

      <div className="w-px h-5 bg-white/10" />

      <button
        onClick={() => store.autoLayout()}
        className="px-2.5 py-1 text-xs text-gray-300 hover:text-white hover:bg-white/5 rounded transition-colors"
        title="Auto-layout: arrange nodes left-to-right by data flow"
      >
        Layout
      </button>

      <div className="flex-1" />

      <div className="text-xs text-gray-500">
        {nodeCount} nodes · {edgeCount} connections
      </div>
      <div className="text-xs text-gray-500 font-mono">
        {store.project.name}
      </div>
    </div>
  );
}
