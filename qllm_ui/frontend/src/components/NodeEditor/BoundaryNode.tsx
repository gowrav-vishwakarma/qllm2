import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import type { NodeData } from "@/types";

function BoundaryNodeInner({ data }: NodeProps & { data: NodeData }) {
  const isInput = data.moduleId === "__input__";
  const ports = isInput ? data.outputs : data.inputs;
  const accentBg = isInput ? "#3b82f6" : "#f97316";
  const accentBorder = isInput ? "#2563eb" : "#ea580c";

  return (
    <div
      className="min-w-[120px] rounded-lg shadow-lg border-2"
      style={{
        borderColor: accentBorder,
        background: "#0f172a",
      }}
    >
      <div
        className="px-3 py-1.5 rounded-t-md text-xs font-bold text-white tracking-wide text-center"
        style={{ background: accentBg }}
      >
        {isInput ? "\u25B6 " : ""}{data.moduleName}{!isInput ? " \u25B6" : ""}
      </div>

      <div className="px-3 py-2 flex flex-col gap-1">
        {ports.map((port) => (
          <div
            key={port.name}
            className="relative flex items-center gap-1.5"
            style={{ justifyContent: isInput ? "flex-end" : "flex-start" }}
          >
            {!isInput && (
              <Handle
                type="target"
                position={Position.Left}
                id={port.name}
                className="!w-3 !h-3 !bg-blue-400 !border-blue-600 !border-2"
                style={{ position: "relative", top: 0, transform: "none" }}
              />
            )}
            <span className="text-[11px] text-gray-300 font-medium">
              {port.name}
              <span className="text-gray-500 ml-1 text-[9px]">{port.type}</span>
            </span>
            {isInput && (
              <Handle
                type="source"
                position={Position.Right}
                id={port.name}
                className="!w-3 !h-3 !bg-orange-400 !border-orange-600 !border-2"
                style={{ position: "relative", top: 0, transform: "none" }}
              />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export const BoundaryNode = memo(BoundaryNodeInner);
