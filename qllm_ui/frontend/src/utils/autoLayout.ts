import dagre from "@dagrejs/dagre";
import type { Node, Edge } from "@xyflow/react";
import type { NodeData } from "@/types";

const NODE_WIDTH = 200;
const NODE_HEIGHT = 120;

/**
 * Compute a left-to-right DAG layout using dagre.
 * Nodes that share data (fan-out / fan-in) get placed apart vertically.
 * Returns a new array of nodes with updated positions.
 */
export function getLayoutedNodes(
  nodes: Node<NodeData>[],
  edges: Edge[],
  direction: "LR" | "TB" = "LR"
): Node<NodeData>[] {
  if (nodes.length === 0) return nodes;

  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({
    rankdir: direction,
    nodesep: 60,
    ranksep: 200,
    edgesep: 30,
    marginx: 40,
    marginy: 40,
  });

  for (const node of nodes) {
    const nPorts = Math.max(
      (node.data as NodeData).inputs?.length ?? 1,
      (node.data as NodeData).outputs?.length ?? 1
    );
    const nParams = Object.keys((node.data as NodeData).params ?? {}).length;
    const h = 40 + nPorts * 20 + Math.min(nParams, 3) * 12 + (nParams > 0 ? 16 : 0);
    g.setNode(node.id, { width: NODE_WIDTH, height: Math.max(NODE_HEIGHT, h) });
  }

  for (const edge of edges) {
    g.setEdge(edge.source, edge.target);
  }

  dagre.layout(g);

  return nodes.map((node) => {
    const pos = g.node(node.id);
    if (!pos) return node;
    return {
      ...node,
      position: {
        x: pos.x - NODE_WIDTH / 2,
        y: pos.y - (pos.height ?? NODE_HEIGHT) / 2,
      },
    };
  });
}
