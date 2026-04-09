import { memo, useCallback } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import type { NodeData, ModuleDef } from "@/types";
import { useStore } from "@/store/useStore";

const CATEGORY_COLORS: Record<string, string> = {
  "QLLM Custom": "#6366f1",
  Standard: "#0ea5e9",
  "SSM/Mamba": "#f59e0b",
  "My Modules": "#10b981",
  Training: "#ef4444",
};

function ModuleNodeInner({ id, data, selected }: NodeProps & { data: NodeData }) {
  const openEditor = useStore((s) => s.openEditor);
  const setSelectedNodeId = useStore((s) => s.setSelectedNodeId);
  const drillInto = useStore((s) => s.drillInto);
  const moduleRegistry = useStore((s) => s.moduleRegistry);

  const accent = CATEGORY_COLORS[data.category || ""] || "#6b7280";
  const mod = moduleRegistry.find((m: ModuleDef) => m.id === data.moduleId);
  const hasSubGraph = !!mod?.subGraph;

  const onDoubleClick = useCallback(() => {
    if (hasSubGraph) {
      drillInto(id);
    } else {
      openEditor(data.moduleId);
    }
  }, [hasSubGraph, drillInto, id, openEditor, data.moduleId]);

  const onClick = useCallback(() => {
    setSelectedNodeId(id);
  }, [setSelectedNodeId, id]);

  return (
    <div
      onClick={onClick}
      onDoubleClick={onDoubleClick}
      className="min-w-[160px] rounded-lg shadow-lg border transition-shadow"
      style={{
        borderColor: selected ? accent : "rgba(255,255,255,0.1)",
        boxShadow: selected ? `0 0 0 2px ${accent}40` : undefined,
        background: "#1e1e2e",
      }}
    >
      {/* Header */}
      <div
        className="px-3 py-1.5 rounded-t-lg text-xs font-semibold text-white truncate flex items-center gap-1"
        style={{ background: accent }}
      >
        {data.moduleName}
        {hasSubGraph && (
          <span className="ml-auto text-[9px] opacity-60" title="Double-click to drill in">
            &#x25B6;&#x25B6;
          </span>
        )}
      </div>

      {/* Ports */}
      <div className="px-2 py-2 flex justify-between gap-4">
        {/* Input ports */}
        <div className="flex flex-col gap-1">
          {data.inputs.map((port) => (
            <div key={port.name} className="relative flex items-center gap-1.5">
              <Handle
                type="target"
                position={Position.Left}
                id={port.name}
                className="!w-2.5 !h-2.5 !bg-blue-400 !border-blue-600 !border"
                style={{ position: "relative", top: 0, transform: "none" }}
              />
              <span className="text-[10px] text-gray-400">
                {port.name}
                {port.optional ? "?" : ""}
              </span>
            </div>
          ))}
        </div>

        {/* Output ports */}
        <div className="flex flex-col gap-1 items-end">
          {data.outputs.map((port) => (
            <div key={port.name} className="relative flex items-center gap-1.5">
              <span className="text-[10px] text-gray-400">{port.name}</span>
              <Handle
                type="source"
                position={Position.Right}
                id={port.name}
                className="!w-2.5 !h-2.5 !bg-orange-400 !border-orange-600 !border"
                style={{ position: "relative", top: 0, transform: "none" }}
              />
            </div>
          ))}
        </div>
      </div>

      {/* Params preview */}
      {Object.keys(data.params).length > 0 && (
        <div className="px-2 pb-2 border-t border-white/5 pt-1.5">
          {Object.entries(data.params)
            .slice(0, 3)
            .map(([k, v]) => (
              <div key={k} className="text-[9px] text-gray-500 truncate">
                {k}: <span className="text-gray-300">{String(v)}</span>
              </div>
            ))}
          {Object.keys(data.params).length > 3 && (
            <div className="text-[9px] text-gray-600">...</div>
          )}
        </div>
      )}
    </div>
  );
}

export const ModuleNode = memo(ModuleNodeInner);
