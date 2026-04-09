import { useCallback } from "react";
import { useStore } from "@/store/useStore";
import type { NodeData, ModuleDef, PortDef } from "@/types";
import type { Node, Edge } from "@xyflow/react";

interface ContextMenuProps {
  x: number;
  y: number;
  nodeId: string | null;
  onClose: () => void;
}

export default function ContextMenu({ x, y, nodeId, onClose }: ContextMenuProps) {
  const { nodes, edges, removeNode, openEditor, addCustomModule, moduleRegistry, nodeCounter } = useStore();

  const handleDelete = useCallback(() => {
    if (nodeId) removeNode(nodeId);
    onClose();
  }, [nodeId, removeNode, onClose]);

  const handleViewCode = useCallback(() => {
    if (!nodeId) return;
    const node = nodes.find((n) => n.id === nodeId);
    if (node) openEditor(node.data.moduleId);
    onClose();
  }, [nodeId, nodes, openEditor, onClose]);

  const handleDuplicate = useCallback(() => {
    if (!nodeId) return;
    const node = nodes.find((n) => n.id === nodeId);
    if (!node) return;

    const newNode: Node<NodeData> = {
      id: `${node.data.moduleId.replace(/[^a-zA-Z0-9]/g, "_")}_${nodeCounter}`,
      type: "moduleNode",
      position: { x: node.position.x + 50, y: node.position.y + 50 },
      data: { ...node.data, params: { ...node.data.params } },
    };
    useStore.getState().addNode(newNode);
    onClose();
  }, [nodeId, nodes, nodeCounter, onClose]);

  const handleGroupSelected = useCallback(() => {
    const selectedNodes = nodes.filter((n) => n.selected);
    if (selectedNodes.length < 2) {
      onClose();
      return;
    }

    const selectedIds = new Set(selectedNodes.map((n) => n.id));

    // Find external inputs (edges coming from outside the group)
    const externalInputs: PortDef[] = [];
    const externalOutputs: PortDef[] = [];
    const seenInputs = new Set<string>();
    const seenOutputs = new Set<string>();

    for (const edge of edges) {
      if (selectedIds.has(edge.target) && !selectedIds.has(edge.source)) {
        const portKey = `${edge.targetHandle || "x"}`;
        if (!seenInputs.has(portKey)) {
          externalInputs.push({ name: portKey, type: "Tensor" });
          seenInputs.add(portKey);
        }
      }
      if (selectedIds.has(edge.source) && !selectedIds.has(edge.target)) {
        const portKey = `${edge.sourceHandle || "out"}`;
        if (!seenOutputs.has(portKey)) {
          externalOutputs.push({ name: portKey, type: "Tensor" });
          seenOutputs.add(portKey);
        }
      }
    }

    if (externalInputs.length === 0) externalInputs.push({ name: "x", type: "Tensor" });
    if (externalOutputs.length === 0) externalOutputs.push({ name: "out", type: "Tensor" });

    // Build composite code from selected nodes
    const subModuleNames = selectedNodes.map((n) => n.data.moduleName).join(" + ");
    const groupName = `Group_${nodeCounter}`;

    const initLines = selectedNodes.map((n) => {
      const paramStr = Object.entries(n.data.params)
        .map(([k, v]) => `${k}=${typeof v === "string" ? `"${v}"` : v}`)
        .join(", ");
      return `        self.${n.id.replace(/[^a-zA-Z0-9_]/g, "_")} = ${n.data.moduleName}(${paramStr})`;
    });

    const forwardArgs = externalInputs.map((p) => `${p.name}: torch.Tensor`).join(", ");
    const code = `class ${groupName}(nn.Module):
    """Composite: ${subModuleNames}"""
    def __init__(self):
        super().__init__()
${initLines.join("\n")}

    def forward(self, ${forwardArgs}) -> torch.Tensor:
        x = ${externalInputs[0].name}
        # TODO: wire internal forward pass
        return x`;

    const newMod: ModuleDef = {
      id: groupName,
      name: groupName,
      category: "My Modules",
      code,
      inputs: externalInputs,
      outputs: externalOutputs,
      constructorParams: [],
      isCustom: true,
    };
    addCustomModule(newMod);

    // Calculate center position of group
    const avgX = selectedNodes.reduce((s, n) => s + n.position.x, 0) / selectedNodes.length;
    const avgY = selectedNodes.reduce((s, n) => s + n.position.y, 0) / selectedNodes.length;

    // Remove selected nodes
    for (const n of selectedNodes) {
      removeNode(n.id);
    }

    // Add the composite node
    const compositeNode: Node<NodeData> = {
      id: `${groupName}_${nodeCounter}`,
      type: "moduleNode",
      position: { x: avgX, y: avgY },
      data: {
        moduleId: groupName,
        moduleName: groupName,
        params: {},
        inputs: externalInputs,
        outputs: externalOutputs,
        code,
        category: "My Modules",
      },
    };
    useStore.getState().addNode(compositeNode);
    onClose();
  }, [nodes, edges, removeNode, addCustomModule, nodeCounter, onClose]);

  return (
    <div
      className="fixed z-[100] bg-[#1e1e2e] border border-white/10 rounded-lg shadow-xl py-1 min-w-[160px]"
      style={{ left: x, top: y }}
    >
      {nodeId && (
        <>
          <button
            onClick={handleViewCode}
            className="w-full px-3 py-1.5 text-sm text-gray-300 hover:bg-white/5 text-left"
          >
            View Code
          </button>
          <button
            onClick={handleDuplicate}
            className="w-full px-3 py-1.5 text-sm text-gray-300 hover:bg-white/5 text-left"
          >
            Duplicate
          </button>
          <div className="border-t border-white/10 my-1" />
          <button
            onClick={handleDelete}
            className="w-full px-3 py-1.5 text-sm text-red-400 hover:bg-white/5 text-left"
          >
            Delete
          </button>
        </>
      )}
      {nodes.filter((n) => n.selected).length >= 2 && (
        <>
          <div className="border-t border-white/10 my-1" />
          <button
            onClick={handleGroupSelected}
            className="w-full px-3 py-1.5 text-sm text-green-400 hover:bg-white/5 text-left"
          >
            Group Selected → Module
          </button>
        </>
      )}
      <button
        onClick={onClose}
        className="w-full px-3 py-1.5 text-sm text-gray-500 hover:bg-white/5 text-left"
      >
        Cancel
      </button>
    </div>
  );
}
