export interface PortDef {
  name: string;
  type: string; // "Tensor" | "Scalar" | "Optional[Tensor]" etc.
  optional?: boolean;
}

export interface ConstructorParam {
  name: string;
  type: string; // "int" | "float" | "bool" | "str"
  default?: unknown;
  description?: string;
}

export interface SubGraphNode {
  id: string;
  type: string;
  label?: string;
  params: Record<string, unknown>;
  portName?: string;
}

export interface SubGraphConnection {
  source: string;
  sourcePort: string;
  target: string;
  targetPort: string;
}

export interface SubGraphDef {
  nodes: SubGraphNode[];
  connections: SubGraphConnection[];
}

export interface ModuleDef {
  id: string;
  name: string;
  category: string;
  code: string;
  inputs: PortDef[];
  outputs: PortDef[];
  constructorParams: ConstructorParam[];
  isCustom?: boolean;
  isTemplate?: boolean;
  subGraph?: SubGraphDef;
}

export interface NodeData {
  [key: string]: unknown;
  moduleId: string;
  moduleName: string;
  params: Record<string, unknown>;
  inputs: PortDef[];
  outputs: PortDef[];
  code?: string;
  category?: string;
}

export interface ProjectConfig {
  name: string;
  outputDir: string;
  checkpointDir: string;
  logDir: string;
}

export interface TrainingConfig {
  dataset: string;
  tokenizer: string;
  seqLen: number;
  batchSize: number;
  optimizer: {
    type: string;
    lr: number;
    betas: [number, number];
    weightDecay: number;
  };
  scheduler: {
    type: string;
    warmupSteps: number;
  };
  loss: {
    type: string;
    auxLosses: { name: string; weight: number }[];
  };
  epochs: number;
  gradClip: number;
  amp: boolean;
  compile: boolean;
  gradAccumulation: number;
}

export interface SerializedProject {
  version: string;
  project: ProjectConfig;
  customModules: Record<string, {
    code: string;
    ports: { inputs: PortDef[]; outputs: PortDef[] };
    constructorParams: Record<string, { type: string; default?: unknown }>;
  }>;
  nodes: {
    id: string;
    type: string;
    position: { x: number; y: number };
    params: Record<string, unknown>;
    inputPorts?: PortDef[];
    outputPorts?: PortDef[];
  }[];
  connections: {
    source: string;
    sourcePort: string;
    target: string;
    targetPort: string;
  }[];
  training: TrainingConfig;
}

export interface RunStatus {
  id: string;
  status: "running" | "stopped" | "completed" | "failed";
  epoch?: number;
  step?: number;
  loss?: number;
  ppl?: number;
  startedAt?: string;
  outputDir?: string;
}
