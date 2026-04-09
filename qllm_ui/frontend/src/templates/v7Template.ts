import type { SerializedProject } from "@/types";

/**
 * V7 LM template — compact form using V7Block composite nodes.
 *
 * Double-click any V7Block to drill into its internal wiring:
 *   norm1 → CGU → Residual(cgu_out + x)
 *   norm2 → PAM → Residual(pam_out + post-cgu)
 *
 * Top-level flow:
 *   ComplexEmbed → embed_norm → V7Block_0 → V7Block_1 → V7Block_2 → output_norm → ComplexLMHead
 *
 * Hierarchical dt_bias: -6.91 (global) → -3.45 (mid) → 0.0 (step)
 */

const DIM = 384;
const N_HEADS = 6;
const HEAD_DIM = 64;
const EXPAND = 3;
const ACTIVATION = "modrelu";

export const v7Template: SerializedProject = {
  version: "1.0",
  project: {
    name: "v7_experiment",
    outputDir: "./experiments/v7_experiment",
    checkpointDir: "./experiments/v7_experiment/checkpoints",
    logDir: "./experiments/v7_experiment/logs",
  },
  customModules: {},
  nodes: [
    {
      id: "embed",
      type: "ComplexEmbed",
      position: { x: 50, y: 200 },
      params: { vocab_size: 50257, dim: DIM },
    },
    {
      id: "embed_norm",
      type: "ComplexNorm",
      position: { x: 250, y: 200 },
      params: { dim: DIM },
    },
    {
      id: "block_0",
      type: "V7Block",
      position: { x: 450, y: 150 },
      params: {
        dim: DIM, expand: EXPAND, activation: ACTIVATION, dropout: 0.1,
        n_heads: N_HEADS, head_dim: HEAD_DIM, use_rope: true, use_gsp: true,
        chunk_size: 256, use_reverse_assoc: true, max_seq_len: 2048,
        dt_bias_init: -6.91, layer_idx: 0,
      },
    },
    {
      id: "block_1",
      type: "V7Block",
      position: { x: 700, y: 150 },
      params: {
        dim: DIM, expand: EXPAND, activation: ACTIVATION, dropout: 0.1,
        n_heads: N_HEADS, head_dim: HEAD_DIM, use_rope: true, use_gsp: true,
        chunk_size: 256, use_reverse_assoc: true, max_seq_len: 2048,
        dt_bias_init: -3.45, layer_idx: 1,
      },
    },
    {
      id: "block_2",
      type: "V7Block",
      position: { x: 950, y: 150 },
      params: {
        dim: DIM, expand: EXPAND, activation: ACTIVATION, dropout: 0.1,
        n_heads: N_HEADS, head_dim: HEAD_DIM, use_rope: true, use_gsp: true,
        chunk_size: 256, use_reverse_assoc: true, max_seq_len: 2048,
        dt_bias_init: 0.0, layer_idx: 2,
      },
    },
    {
      id: "output_norm",
      type: "ComplexNorm",
      position: { x: 1200, y: 200 },
      params: { dim: DIM },
    },
    {
      id: "head",
      type: "ComplexLMHead",
      position: { x: 1400, y: 200 },
      params: { dim: DIM, vocab_size: 50257 },
    },
  ],
  connections: [
    { source: "embed", sourcePort: "out", target: "embed_norm", targetPort: "z" },
    { source: "embed_norm", sourcePort: "out", target: "block_0", targetPort: "x" },
    { source: "block_0", sourcePort: "out", target: "block_1", targetPort: "x" },
    { source: "block_1", sourcePort: "out", target: "block_2", targetPort: "x" },
    { source: "block_2", sourcePort: "out", target: "output_norm", targetPort: "z" },
    { source: "output_norm", sourcePort: "out", target: "head", targetPort: "x" },
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
