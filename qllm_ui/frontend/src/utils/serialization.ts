import type { Node, Edge } from "@xyflow/react";
import type { NodeData, SerializedProject, ModuleDef, TrainingConfig, ProjectConfig } from "@/types";

export function serializeProject(
  nodes: Node<NodeData>[],
  edges: Edge[],
  modules: ModuleDef[],
  project: ProjectConfig,
  training: TrainingConfig
): SerializedProject {
  const customModules: SerializedProject["customModules"] = {};
  for (const mod of modules.filter((m) => m.isCustom)) {
    customModules[mod.id] = {
      code: mod.code,
      ports: { inputs: mod.inputs, outputs: mod.outputs },
      constructorParams: Object.fromEntries(
        mod.constructorParams.map((p) => [p.name, { type: p.type, default: p.default }])
      ),
    };
  }

  return {
    version: "1.0",
    project,
    customModules,
    nodes: nodes.map((n) => ({
      id: n.id,
      type: n.data.moduleId,
      position: n.position,
      params: n.data.params,
      inputPorts: n.data.inputs,
      outputPorts: n.data.outputs,
    })),
    connections: edges.map((e) => ({
      source: e.source,
      sourcePort: e.sourceHandle || "out",
      target: e.target,
      targetPort: e.targetHandle || "x",
    })),
    training,
  };
}

export function deserializeProject(
  data: SerializedProject,
  registry: ModuleDef[]
): {
  nodes: Node<NodeData>[];
  edges: Edge[];
  customModules: ModuleDef[];
  project: ProjectConfig;
  training: TrainingConfig;
} {
  const customModules: ModuleDef[] = [];
  for (const [id, def] of Object.entries(data.customModules || {})) {
    customModules.push({
      id,
      name: id,
      category: "My Modules",
      code: def.code,
      inputs: def.ports.inputs,
      outputs: def.ports.outputs,
      constructorParams: Object.entries(def.constructorParams).map(([name, p]) => ({
        name,
        type: p.type,
        default: p.default,
      })),
      isCustom: true,
    });
  }

  const allModules = [...registry, ...customModules];

  const nodes: Node<NodeData>[] = data.nodes.map((n) => {
    const mod = allModules.find((m) => m.id === n.type);
    return {
      id: n.id,
      type: "moduleNode",
      position: n.position,
      data: {
        moduleId: n.type,
        moduleName: mod?.name || n.type,
        params: n.params,
        inputs: mod?.inputs || [{ name: "x", type: "Tensor" }],
        outputs: mod?.outputs || [{ name: "out", type: "Tensor" }],
        code: mod?.code,
        category: mod?.category,
      },
    };
  });

  const edges: Edge[] = data.connections.map((c, i) => ({
    id: `e-${i}`,
    source: c.source,
    sourceHandle: c.sourcePort,
    target: c.target,
    targetHandle: c.targetPort,
    type: "default",
  }));

  return {
    nodes,
    edges,
    customModules,
    project: data.project,
    training: data.training,
  };
}
