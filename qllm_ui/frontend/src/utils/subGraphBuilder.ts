import type { Node, Edge } from "@xyflow/react";
import type { ModuleDef, NodeData, SubGraphDef, PortDef } from "@/types";
import { getLayoutedNodes } from "./autoLayout";

/**
 * Convert a SubGraphDef into React Flow nodes and edges for drill-down view.
 *
 * Boundary nodes (__input__ / __output__) become special "boundaryNode" types
 * that show the parent's port names as entry/exit markers.
 */
export function buildSubGraphView(
  subGraph: SubGraphDef,
  parentInputs: PortDef[],
  parentOutputs: PortDef[],
  registry: ModuleDef[]
): { nodes: Node<NodeData>[]; edges: Edge[] } {
  const nodes: Node<NodeData>[] = [];

  for (const sgNode of subGraph.nodes) {
    if (sgNode.type === "__input__") {
      nodes.push({
        id: sgNode.id,
        type: "boundaryNode",
        position: { x: 0, y: 0 },
        data: {
          moduleId: "__input__",
          moduleName: sgNode.label || "Input",
          params: {},
          inputs: [],
          outputs: sgNode.portName
            ? [{ name: sgNode.portName, type: "Tensor" }]
            : parentInputs.map((p) => ({ name: p.name, type: p.type })),
          category: "__boundary__",
          boundaryDirection: "input" as unknown,
        } as NodeData,
      });
    } else if (sgNode.type === "__output__") {
      nodes.push({
        id: sgNode.id,
        type: "boundaryNode",
        position: { x: 0, y: 0 },
        data: {
          moduleId: "__output__",
          moduleName: sgNode.label || "Output",
          params: {},
          inputs: sgNode.portName
            ? [{ name: sgNode.portName, type: "Tensor" }]
            : parentOutputs.map((p) => ({ name: p.name, type: p.type })),
          outputs: [],
          category: "__boundary__",
          boundaryDirection: "output" as unknown,
        } as NodeData,
      });
    } else {
      const mod = registry.find((m) => m.id === sgNode.type);
      nodes.push({
        id: sgNode.id,
        type: "moduleNode",
        position: { x: 0, y: 0 },
        data: {
          moduleId: sgNode.type,
          moduleName: sgNode.label || mod?.name || sgNode.type,
          params: sgNode.params,
          inputs: mod?.inputs || [{ name: "x", type: "Tensor" }],
          outputs: mod?.outputs || [{ name: "out", type: "Tensor" }],
          code: mod?.code,
          category: mod?.category,
        } as NodeData,
      });
    }
  }

  const edges: Edge[] = subGraph.connections.map((c, i) => ({
    id: `sg-e-${i}`,
    source: c.source,
    sourceHandle: c.sourcePort,
    target: c.target,
    targetHandle: c.targetPort,
    type: "default",
  }));

  const layouted = getLayoutedNodes(nodes, edges);
  return { nodes: layouted, edges };
}
