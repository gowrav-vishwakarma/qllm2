import { create } from "zustand";
import {
  type Node,
  type Edge,
  type OnNodesChange,
  type OnEdgesChange,
  type OnConnect,
  applyNodeChanges,
  applyEdgeChanges,
  addEdge,
} from "@xyflow/react";
import type {
  ModuleDef,
  NodeData,
  ProjectConfig,
  TrainingConfig,
  RunStatus,
} from "@/types";
import { getLayoutedNodes } from "@/utils/autoLayout";
import { buildSubGraphView } from "@/utils/subGraphBuilder";

interface AppState {
  nodes: Node<NodeData>[];
  edges: Edge[];
  onNodesChange: OnNodesChange;
  onEdgesChange: OnEdgesChange;
  onConnect: OnConnect;
  addNode: (node: Node<NodeData>) => void;
  removeNode: (id: string) => void;
  updateNodeParams: (id: string, params: Record<string, unknown>) => void;

  selectedNodeId: string | null;
  setSelectedNodeId: (id: string | null) => void;

  moduleRegistry: ModuleDef[];
  setModuleRegistry: (modules: ModuleDef[]) => void;
  addCustomModule: (mod: ModuleDef) => void;
  updateCustomModule: (id: string, mod: Partial<ModuleDef>) => void;

  project: ProjectConfig;
  setProject: (p: Partial<ProjectConfig>) => void;

  training: TrainingConfig;
  setTraining: (t: Partial<TrainingConfig>) => void;

  editorOpen: boolean;
  editorModuleId: string | null;
  openEditor: (moduleId: string | null) => void;
  closeEditor: () => void;

  runs: RunStatus[];
  addRun: (r: RunStatus) => void;
  updateRun: (id: string, r: Partial<RunStatus>) => void;

  autoLayout: () => void;

  viewStack: Array<{
    parentNodeId: string;
    parentLabel: string;
    nodes: Node<NodeData>[];
    edges: Edge[];
  }>;
  drillInto: (nodeId: string) => void;
  drillOut: () => void;
  drillOutTo: (depth: number) => void;

  nodeCounter: number;
}

const defaultTraining: TrainingConfig = {
  dataset: "tinystories",
  tokenizer: "gpt2",
  seqLen: 512,
  batchSize: 32,
  optimizer: { type: "AdamW", lr: 3e-4, betas: [0.9, 0.95], weightDecay: 0.1 },
  scheduler: { type: "cosine", warmupSteps: 500 },
  loss: { type: "cross_entropy", auxLosses: [] },
  epochs: 50,
  gradClip: 1.0,
  amp: true,
  compile: false,
  gradAccumulation: 1,
};

const defaultProject: ProjectConfig = {
  name: "new_experiment",
  outputDir: "./experiments/new_experiment",
  checkpointDir: "./experiments/new_experiment/checkpoints",
  logDir: "./experiments/new_experiment/logs",
};

export const useStore = create<AppState>((set, get) => ({
  nodes: [],
  edges: [],
  onNodesChange: (changes) =>
    set({ nodes: applyNodeChanges(changes, get().nodes) as Node<NodeData>[] }),
  onEdgesChange: (changes) =>
    set({ edges: applyEdgeChanges(changes, get().edges) }),
  onConnect: (connection) => set({ edges: addEdge(connection, get().edges) }),
  addNode: (node) => set({ nodes: [...get().nodes, node], nodeCounter: get().nodeCounter + 1 }),
  removeNode: (id) =>
    set({
      nodes: get().nodes.filter((n) => n.id !== id),
      edges: get().edges.filter((e) => e.source !== id && e.target !== id),
    }),
  updateNodeParams: (id, params) =>
    set({
      nodes: get().nodes.map((n) =>
        n.id === id ? { ...n, data: { ...n.data, params: { ...n.data.params, ...params } } } : n
      ),
    }),

  selectedNodeId: null,
  setSelectedNodeId: (id) => set({ selectedNodeId: id }),

  moduleRegistry: [],
  setModuleRegistry: (modules) => set({ moduleRegistry: modules }),
  addCustomModule: (mod) =>
    set({ moduleRegistry: [...get().moduleRegistry, mod] }),
  updateCustomModule: (id, mod) =>
    set({
      moduleRegistry: get().moduleRegistry.map((m) =>
        m.id === id ? { ...m, ...mod } : m
      ),
    }),

  project: defaultProject,
  setProject: (p) => set({ project: { ...get().project, ...p } }),

  training: defaultTraining,
  setTraining: (t) => set({ training: { ...get().training, ...t } }),

  editorOpen: false,
  editorModuleId: null,
  openEditor: (moduleId) => set({ editorOpen: true, editorModuleId: moduleId }),
  closeEditor: () => set({ editorOpen: false, editorModuleId: null }),

  runs: [],
  addRun: (r) => set({ runs: [...get().runs, r] }),
  updateRun: (id, r) =>
    set({ runs: get().runs.map((run) => (run.id === id ? { ...run, ...r } : run)) }),

  autoLayout: () => {
    const { nodes, edges } = get();
    set({ nodes: getLayoutedNodes(nodes, edges) });
  },

  viewStack: [],

  drillInto: (nodeId: string) => {
    const { nodes, edges, moduleRegistry } = get();
    const node = nodes.find((n) => n.id === nodeId);
    if (!node) return;
    const mod = moduleRegistry.find((m) => m.id === (node.data as NodeData).moduleId);
    if (!mod?.subGraph) return;
    const nodeData = node.data as NodeData;
    const { nodes: subNodes, edges: subEdges } = buildSubGraphView(
      mod.subGraph,
      mod.inputs,
      mod.outputs,
      moduleRegistry
    );
    set({
      viewStack: [
        ...get().viewStack,
        { parentNodeId: nodeId, parentLabel: nodeData.moduleName, nodes, edges },
      ],
      nodes: subNodes,
      edges: subEdges,
      selectedNodeId: null,
    });
  },

  drillOut: () => {
    const stack = get().viewStack;
    if (stack.length === 0) return;
    const top = stack[stack.length - 1];
    set({
      viewStack: stack.slice(0, -1),
      nodes: top.nodes,
      edges: top.edges,
      selectedNodeId: null,
    });
  },

  drillOutTo: (depth: number) => {
    const stack = get().viewStack;
    if (depth >= stack.length || depth < 0) return;
    const target = stack[depth];
    set({
      viewStack: stack.slice(0, depth),
      nodes: target.nodes,
      edges: target.edges,
      selectedNodeId: null,
    });
  },

  nodeCounter: 0,
}));
