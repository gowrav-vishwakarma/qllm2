import type { SerializedProject } from "@/types";

export const v5Template: SerializedProject = {
  version: "1.0",
  project: {
    name: "v5_algebraic_lm",
    outputDir: "./experiments/v5_algebraic_lm",
    checkpointDir: "./experiments/v5_algebraic_lm/checkpoints",
    logDir: "./experiments/v5_algebraic_lm/logs",
  },
  customModules: {},
  nodes: [
    {
      id: "embed",
      type: "ComplexEmbed",
      position: { x: 50, y: 250 },
      params: { vocab_size: 50257, dim: 256 },
    },
    {
      id: "ssm_1",
      type: "ComplexSSM",
      position: { x: 350, y: 150 },
      params: { dim: 256, state_dim: 64, n_layers: 1 },
    },
    {
      id: "ssm_2",
      type: "ComplexSSM",
      position: { x: 350, y: 350 },
      params: { dim: 256, state_dim: 64, n_layers: 1 },
    },
    {
      id: "attn",
      type: "PhaseAttention",
      position: { x: 650, y: 250 },
      params: { dim: 256, n_heads: 4, window_size: 256 },
    },
    {
      id: "ssm_3",
      type: "ComplexSSM",
      position: { x: 950, y: 150 },
      params: { dim: 256, state_dim: 64, n_layers: 1 },
    },
    {
      id: "ssm_4",
      type: "ComplexSSM",
      position: { x: 950, y: 350 },
      params: { dim: 256, state_dim: 64, n_layers: 1 },
    },
    {
      id: "head",
      type: "ComplexLMHead",
      position: { x: 1250, y: 250 },
      params: { dim: 256, vocab_size: 50257 },
    },
  ],
  connections: [
    { source: "embed", sourcePort: "out", target: "ssm_1", targetPort: "x" },
    { source: "ssm_1", sourcePort: "out", target: "ssm_2", targetPort: "x" },
    { source: "ssm_2", sourcePort: "out", target: "attn", targetPort: "x" },
    { source: "attn", sourcePort: "out", target: "ssm_3", targetPort: "x" },
    { source: "ssm_3", sourcePort: "out", target: "ssm_4", targetPort: "x" },
    { source: "ssm_4", sourcePort: "out", target: "head", targetPort: "x" },
  ],
  training: {
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
  },
};
