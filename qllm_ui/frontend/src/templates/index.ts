import type { SerializedProject } from "@/types";
import { standardTransformerTemplate } from "./standardTransformer";
import { v5Template } from "./v5Template";
import { v6Template } from "./v6Template";
import { v7Template } from "./v7Template";
import { mambaTemplate } from "./mambaTemplate";

export interface TemplateEntry {
  id: string;
  name: string;
  description: string;
  data: SerializedProject;
}

export const templates: TemplateEntry[] = [
  {
    id: "standard_transformer",
    name: "Standard Transformer",
    description: "Encoder-only LM — double-click TransformerEncoderLayer to see norm→attn→FFN→residual internals",
    data: standardTransformerTemplate,
  },
  {
    id: "v5_algebraic",
    name: "V5 AlgebraicLM",
    description: "Complex-valued SSM backbone with PhaseAttention (V5 architecture)",
    data: v5Template,
  },
  {
    id: "v6_phasefield",
    name: "V6 Medium-PAM-v3",
    description: "Best V6 run (PPL 29.95) — double-click V6Block to drill into CGU+PAM internals",
    data: v6Template,
  },
  {
    id: "v7_lm",
    name: "V7 LM",
    description: "Hierarchical dt_bias blocks — double-click V7Block/CGU/PAM to drill into subgraphs",
    data: v7Template,
  },
  {
    id: "mamba_lm",
    name: "Mamba LM",
    description: "Selective SSM (Mamba-style) language model",
    data: mambaTemplate,
  },
];
