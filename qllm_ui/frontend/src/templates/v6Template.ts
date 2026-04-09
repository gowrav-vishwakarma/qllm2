import type { SerializedProject } from "@/types";

/**
 * V6 PhaseFieldLM template — matches the best V6 run: medium-pam-v3
 *
 * Compact form using V6Block composite nodes matching PhaseFieldBackbone
 * (single_bank=True, interleave_pam=True). Double-click any V6Block
 * to drill into its internal norm→CGU→dropout→residual→pam_norm→PAM→residual wiring.
 *
 * Config from v6/config.py "medium-pam-v3":
 *   dim=384, num_layers=16, single_bank=True, bank_expand=3
 *   GSP=True, interleave_pam=True
 *   PAM: n_heads=6, head_dim=64, RoPE=True, fused_qkv=True
 *   NO working memory, NO internal memory
 *   LR=1e-4, warmup=1000, seq_len=2048, wikitext103
 *   Result: Val PPL 29.95 (~100M params)
 *
 * Top-level flow:
 *   ComplexEmbed → embed_norm → V6Block ×3 → pam_output_norm →
 *   lm_head_proj → lm_head_norm → ComplexLMHead
 */

const DIM = 384;
const N_HEADS = 6;
const HEAD_DIM = 64;
const EXPAND = 3;

export const v6Template: SerializedProject = {
  version: "1.0",
  project: {
    name: "v6_medium_pam_v3",
    outputDir: "./experiments/v6_medium_pam_v3",
    checkpointDir: "./experiments/v6_medium_pam_v3/checkpoints",
    logDir: "./experiments/v6_medium_pam_v3/logs",
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
      type: "V6Block",
      position: { x: 450, y: 150 },
      params: {
        dim: DIM, expand: EXPAND, dropout: 0.1,
        n_heads: N_HEADS, head_dim: HEAD_DIM, use_rope: true, use_gsp: true,
        fused_qkv: true, chunk_size: 0, use_reverse_assoc: false,
        max_seq_len: 2048, dt_bias_init: -4.0, layer_idx: 0,
      },
    },
    {
      id: "block_1",
      type: "V6Block",
      position: { x: 700, y: 150 },
      params: {
        dim: DIM, expand: EXPAND, dropout: 0.1,
        n_heads: N_HEADS, head_dim: HEAD_DIM, use_rope: true, use_gsp: true,
        fused_qkv: true, chunk_size: 0, use_reverse_assoc: false,
        max_seq_len: 2048, dt_bias_init: -4.0, layer_idx: 1,
      },
    },
    {
      id: "block_2",
      type: "V6Block",
      position: { x: 950, y: 150 },
      params: {
        dim: DIM, expand: EXPAND, dropout: 0.1,
        n_heads: N_HEADS, head_dim: HEAD_DIM, use_rope: true, use_gsp: true,
        fused_qkv: true, chunk_size: 0, use_reverse_assoc: false,
        max_seq_len: 2048, dt_bias_init: -4.0, layer_idx: 2,
      },
    },
    // V6 has pam_output_norm after all blocks (not present in V7)
    {
      id: "pam_output_norm",
      type: "ComplexNorm",
      position: { x: 1200, y: 200 },
      params: { dim: DIM },
    },
    {
      id: "lm_head_proj",
      type: "ComplexLinear",
      position: { x: 1400, y: 200 },
      params: { in_dim: DIM, out_dim: DIM, bias: false },
    },
    {
      id: "lm_head_norm",
      type: "ComplexNorm",
      position: { x: 1600, y: 200 },
      params: { dim: DIM },
    },
    {
      id: "head",
      type: "ComplexLMHead",
      position: { x: 1800, y: 200 },
      params: { dim: DIM, vocab_size: 50257 },
    },
  ],
  connections: [
    { source: "embed", sourcePort: "out", target: "embed_norm", targetPort: "z" },
    { source: "embed_norm", sourcePort: "out", target: "block_0", targetPort: "z" },
    { source: "block_0", sourcePort: "out", target: "block_1", targetPort: "z" },
    { source: "block_1", sourcePort: "out", target: "block_2", targetPort: "z" },
    { source: "block_2", sourcePort: "out", target: "pam_output_norm", targetPort: "z" },
    { source: "pam_output_norm", sourcePort: "out", target: "lm_head_proj", targetPort: "x" },
    { source: "lm_head_proj", sourcePort: "out", target: "lm_head_norm", targetPort: "z" },
    { source: "lm_head_norm", sourcePort: "out", target: "head", targetPort: "x" },
  ],
  training: {
    dataset: "wikitext103",
    tokenizer: "gpt2",
    seqLen: 2048,
    batchSize: 3,
    optimizer: { type: "AdamW", lr: 1e-4, betas: [0.9, 0.95], weightDecay: 0.1 },
    scheduler: { type: "cosine", warmupSteps: 1000 },
    loss: { type: "cross_entropy", auxLosses: [] },
    epochs: 10,
    gradClip: 1.0,
    amp: true,
    compile: true,
    gradAccumulation: 1,
  },
};
