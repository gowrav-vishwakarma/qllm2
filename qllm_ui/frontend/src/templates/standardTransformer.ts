import type { SerializedProject } from "@/types";

export const standardTransformerTemplate: SerializedProject = {
  version: "1.0",
  project: {
    name: "standard_transformer",
    outputDir: "./experiments/standard_transformer",
    checkpointDir: "./experiments/standard_transformer/checkpoints",
    logDir: "./experiments/standard_transformer/logs",
  },
  customModules: {},
  nodes: [
    {
      id: "embed",
      type: "nn.Embedding",
      position: { x: 50, y: 200 },
      params: { num_embeddings: 50257, embedding_dim: 512 },
    },
    {
      id: "transformer_1",
      type: "TransformerEncoderLayer",
      position: { x: 350, y: 100 },
      params: { d_model: 512, nhead: 8, dim_feedforward: 2048, dropout: 0.1 },
    },
    {
      id: "transformer_2",
      type: "TransformerEncoderLayer",
      position: { x: 350, y: 250 },
      params: { d_model: 512, nhead: 8, dim_feedforward: 2048, dropout: 0.1 },
    },
    {
      id: "transformer_3",
      type: "TransformerEncoderLayer",
      position: { x: 350, y: 400 },
      params: { d_model: 512, nhead: 8, dim_feedforward: 2048, dropout: 0.1 },
    },
    {
      id: "norm",
      type: "nn.LayerNorm",
      position: { x: 650, y: 250 },
      params: { normalized_shape: 512 },
    },
    {
      id: "head",
      type: "LMHead",
      position: { x: 900, y: 250 },
      params: { dim: 512, vocab_size: 50257 },
    },
  ],
  connections: [
    { source: "embed", sourcePort: "out", target: "transformer_1", targetPort: "x" },
    { source: "transformer_1", sourcePort: "out", target: "transformer_2", targetPort: "x" },
    { source: "transformer_2", sourcePort: "out", target: "transformer_3", targetPort: "x" },
    { source: "transformer_3", sourcePort: "out", target: "norm", targetPort: "x" },
    { source: "norm", sourcePort: "out", target: "head", targetPort: "x" },
  ],
  training: {
    dataset: "tinystories",
    tokenizer: "gpt2",
    seqLen: 512,
    batchSize: 32,
    optimizer: { type: "AdamW", lr: 3e-4, betas: [0.9, 0.95], weightDecay: 0.1 },
    scheduler: { type: "cosine", warmupSteps: 500 },
    loss: { type: "cross_entropy", auxLosses: [] },
    epochs: 20,
    gradClip: 1.0,
    amp: true,
    compile: false,
    gradAccumulation: 1,
  },
};
