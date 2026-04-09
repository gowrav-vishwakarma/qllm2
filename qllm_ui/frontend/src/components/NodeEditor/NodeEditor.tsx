import { useCallback, useRef, useMemo, useState } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  type NodeTypes,
  type OnSelectionChangeParams,
  ReactFlowProvider,
  useReactFlow,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useStore } from "@/store/useStore";
import { ModuleNode } from "./ModuleNode";
import { BoundaryNode } from "./BoundaryNode";
import ContextMenu from "./ContextMenu";
import Breadcrumb from "./Breadcrumb";
import type { NodeData, ModuleDef } from "@/types";

const nodeTypes: NodeTypes = {
  moduleNode: ModuleNode as any,
  boundaryNode: BoundaryNode as any,
};

function FlowInner() {
  const {
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
    setSelectedNodeId,
    addNode,
    moduleRegistry,
    nodeCounter,
  } = useStore();

  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const { screenToFlowPosition } = useReactFlow();
  const [contextMenu, setContextMenu] = useState<{
    x: number;
    y: number;
    nodeId: string | null;
  } | null>(null);

  const onSelectionChange = useCallback(
    ({ nodes: selected }: OnSelectionChangeParams) => {
      setSelectedNodeId(selected.length === 1 ? selected[0].id : null);
    },
    [setSelectedNodeId]
  );

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      const moduleId = event.dataTransfer.getData("application/qllm-module");
      if (!moduleId) return;

      const mod = moduleRegistry.find((m: ModuleDef) => m.id === moduleId);
      if (!mod) return;

      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      const defaultParams: Record<string, unknown> = {};
      for (const p of mod.constructorParams) {
        if (p.default !== undefined) defaultParams[p.name] = p.default;
      }

      const newNode = {
        id: `${mod.id.replace(/[^a-zA-Z0-9]/g, "_")}_${nodeCounter}`,
        type: "moduleNode",
        position,
        data: {
          moduleId: mod.id,
          moduleName: mod.name,
          params: defaultParams,
          inputs: mod.inputs,
          outputs: mod.outputs,
          code: mod.code,
          category: mod.category,
        } as NodeData,
      };
      addNode(newNode);
    },
    [moduleRegistry, screenToFlowPosition, addNode, nodeCounter]
  );

  const defaultEdgeOptions = useMemo(
    () => ({
      type: "default",
      animated: true,
      style: { stroke: "#6366f1", strokeWidth: 2 },
    }),
    []
  );

  const onNodeContextMenu = useCallback(
    (event: React.MouseEvent, node: any) => {
      event.preventDefault();
      setContextMenu({ x: event.clientX, y: event.clientY, nodeId: node.id });
    },
    []
  );

  const onPaneContextMenu = useCallback((event: MouseEvent | React.MouseEvent) => {
    event.preventDefault();
    setContextMenu({ x: (event as React.MouseEvent).clientX, y: (event as React.MouseEvent).clientY, nodeId: null });
  }, []);

  const onPaneClick = useCallback(() => {
    setContextMenu(null);
  }, []);

  return (
    <div ref={reactFlowWrapper} className="flex-1 h-full flex flex-col">
      <Breadcrumb />
      <div className="flex-1 min-h-0">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onSelectionChange={onSelectionChange}
        onDragOver={onDragOver}
        onDrop={onDrop}
        onNodeContextMenu={onNodeContextMenu}
        onPaneContextMenu={onPaneContextMenu}
        onPaneClick={onPaneClick}
        nodeTypes={nodeTypes}
        defaultEdgeOptions={defaultEdgeOptions}
        fitView
        deleteKeyCode="Delete"
        multiSelectionKeyCode="Shift"
        className="bg-[#0f0f1a]"
      >
        <Background color="#333" gap={20} />
        <Controls className="!bg-[#1e1e2e] !border-white/10 !text-white [&>button]:!bg-[#1e1e2e] [&>button]:!border-white/10 [&>button]:!text-white [&>button:hover]:!bg-[#2a2a3e]" />
        <MiniMap
          nodeColor={(n) => {
            const cat = (n.data as NodeData)?.category || "";
            const colors: Record<string, string> = {
              "QLLM Custom": "#6366f1",
              Standard: "#0ea5e9",
              "SSM/Mamba": "#f59e0b",
              "My Modules": "#10b981",
            };
            return colors[cat] || "#6b7280";
          }}
          className="!bg-[#0f0f1a] !border-white/10"
        />
      </ReactFlow>
      </div>
      {contextMenu && (
        <ContextMenu
          x={contextMenu.x}
          y={contextMenu.y}
          nodeId={contextMenu.nodeId}
          onClose={() => setContextMenu(null)}
        />
      )}
    </div>
  );
}

export default function NodeEditor() {
  return (
    <ReactFlowProvider>
      <FlowInner />
    </ReactFlowProvider>
  );
}
