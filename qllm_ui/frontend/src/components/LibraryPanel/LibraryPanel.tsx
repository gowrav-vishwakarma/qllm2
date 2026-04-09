import { useState, useMemo, useCallback } from "react";
import { useStore } from "@/store/useStore";
import type { ModuleDef } from "@/types";
import { templates } from "@/templates";
import { deserializeProject } from "@/utils/serialization";

function ModuleItem({ mod }: { mod: ModuleDef }) {
  const onDragStart = useCallback(
    (event: React.DragEvent) => {
      event.dataTransfer.setData("application/qllm-module", mod.id);
      event.dataTransfer.effectAllowed = "move";
    },
    [mod.id]
  );

  return (
    <div
      draggable
      onDragStart={onDragStart}
      className="px-3 py-1.5 text-sm text-gray-300 hover:bg-white/5 cursor-grab active:cursor-grabbing rounded transition-colors select-none"
      title={mod.code.slice(0, 200)}
    >
      {mod.name}
    </div>
  );
}

const CATEGORY_ORDER = [
  "QLLM Custom",
  "Standard",
  "SSM/Mamba",
  "Templates",
  "My Modules",
  "Training",
];

export default function LibraryPanel() {
  const moduleRegistry = useStore((s) => s.moduleRegistry);
  const openEditor = useStore((s) => s.openEditor);
  const [search, setSearch] = useState("");
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  const filtered = useMemo(() => {
    const q = search.toLowerCase();
    return moduleRegistry.filter(
      (m) =>
        m.name.toLowerCase().includes(q) ||
        m.category.toLowerCase().includes(q)
    );
  }, [moduleRegistry, search]);

  const grouped = useMemo(() => {
    const map = new Map<string, ModuleDef[]>();
    for (const cat of CATEGORY_ORDER) map.set(cat, []);
    for (const m of filtered) {
      const arr = map.get(m.category) || [];
      arr.push(m);
      map.set(m.category, arr);
    }
    for (const [k, v] of map) {
      if (v.length === 0) map.delete(k);
    }
    return map;
  }, [filtered]);

  const toggleCategory = useCallback((cat: string) => {
    setCollapsed((prev) => ({ ...prev, [cat]: !prev[cat] }));
  }, []);

  return (
    <div className="w-56 bg-[#1a1a2e] border-r border-white/10 flex flex-col h-full">
      <div className="p-3 border-b border-white/10">
        <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
          Modules
        </h2>
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search..."
          className="w-full px-2 py-1.5 text-sm bg-[#0f0f1a] border border-white/10 rounded text-gray-200 placeholder-gray-500 focus:outline-none focus:border-indigo-500"
        />
      </div>

      <div className="flex-1 overflow-y-auto py-1">
        {Array.from(grouped.entries()).map(([category, modules]) => (
          <div key={category}>
            <button
              onClick={() => toggleCategory(category)}
              className="w-full flex items-center gap-1 px-3 py-1.5 text-xs font-semibold text-gray-400 uppercase tracking-wider hover:bg-white/5"
            >
              <span className="text-[10px]">
                {collapsed[category] ? "▶" : "▼"}
              </span>
              {category}
              <span className="ml-auto text-gray-600 normal-case">
                {modules.length}
              </span>
            </button>
            {!collapsed[category] &&
              modules.map((m) => <ModuleItem key={m.id} mod={m} />)}
          </div>
        ))}
      </div>

      {/* Templates */}
      <div className="border-t border-white/10 py-1">
        <button
          onClick={() => toggleCategory("__templates__")}
          className="w-full flex items-center gap-1 px-3 py-1.5 text-xs font-semibold text-gray-400 uppercase tracking-wider hover:bg-white/5"
        >
          <span className="text-[10px]">
            {collapsed["__templates__"] ? "▶" : "▼"}
          </span>
          Templates
          <span className="ml-auto text-gray-600 normal-case">{templates.length}</span>
        </button>
        {!collapsed["__templates__"] &&
          templates.map((t) => (
            <button
              key={t.id}
              onClick={() => {
                if (!confirm(`Load "${t.name}" template? This replaces the current graph.`)) return;
                const builtins = moduleRegistry.filter((m: ModuleDef) => !m.isCustom);
                const result = deserializeProject(t.data, builtins);
                for (const mod of result.customModules) {
                  useStore.getState().addCustomModule(mod);
                }
                useStore.setState({
                  nodes: result.nodes,
                  edges: result.edges,
                  nodeCounter: result.nodes.length,
                });
                useStore.getState().setProject(result.project);
                useStore.getState().setTraining(result.training);
                useStore.getState().autoLayout();
              }}
              className="w-full px-3 py-1.5 text-sm text-gray-300 hover:bg-white/5 rounded transition-colors text-left"
              title={t.description}
            >
              {t.name}
            </button>
          ))}
      </div>

      <div className="p-3 border-t border-white/10">
        <button
          onClick={() => openEditor(null)}
          className="w-full px-3 py-2 text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 rounded transition-colors"
        >
          + New Module
        </button>
      </div>
    </div>
  );
}
