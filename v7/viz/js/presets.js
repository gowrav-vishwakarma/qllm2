// V7/V6 config presets, param counting, and shared helpers

export const V7_PRESETS = {
  'tiny': {
    vocab_size:50257,dim:64,n_heads:2,head_dim:32,n_layers:2,expand:2,
    dropout:0,max_seq_len:512,use_rope:true,use_gsp:true,fused_qkv:true,
    qk_norm:false,hierarchical_dt:false,dt_bias_schedule:null,
    cross_level:false,activation:'modrelu',chunk_size:256,
    multi_scale_loss:false,aux_layer_stride:3,max_aux_offset:32,use_reverse_assoc:true,
  },
  'medium': {
    vocab_size:50257,dim:384,n_heads:6,head_dim:64,n_layers:16,expand:3,
    dropout:0.1,max_seq_len:2048,use_rope:true,use_gsp:true,fused_qkv:true,
    qk_norm:false,hierarchical_dt:true,dt_bias_schedule:null,
    cross_level:false,activation:'modrelu',chunk_size:256,
    multi_scale_loss:false,aux_layer_stride:3,max_aux_offset:32,use_reverse_assoc:true,
  },
  'medium_h6': {
    vocab_size:50257,dim:512,n_heads:8,head_dim:64,n_layers:6,expand:4,
    dropout:0.1,max_seq_len:2048,use_rope:true,use_gsp:true,fused_qkv:true,
    qk_norm:false,hierarchical_dt:true,
    dt_bias_schedule:[-6.91,-5.52,-4.08,-2.64,-1.39,0.0],
    cross_level:true,activation:'modrelu',chunk_size:256,
    multi_scale_loss:false,aux_layer_stride:3,max_aux_offset:32,use_reverse_assoc:true,
  },
  'medium_h16_flat': {
    vocab_size:50257,dim:384,n_heads:6,head_dim:64,n_layers:16,expand:3,
    dropout:0.1,max_seq_len:2048,use_rope:true,use_gsp:true,fused_qkv:true,
    qk_norm:false,hierarchical_dt:false,dt_bias_schedule:null,
    cross_level:false,activation:'modrelu',chunk_size:256,
    multi_scale_loss:false,aux_layer_stride:3,max_aux_offset:32,use_reverse_assoc:true,
  },
  'medium_h16_grouped': {
    vocab_size:50257,dim:384,n_heads:6,head_dim:64,n_layers:16,expand:3,
    dropout:0.1,max_seq_len:2048,use_rope:true,use_gsp:true,fused_qkv:true,
    qk_norm:false,hierarchical_dt:true,
    dt_bias_schedule:[-6.91,-6.91,-6.91,-6.91,-5.52,-5.52,-5.52,
      -4.08,-4.08,-4.08,-2.64,-2.64,-2.64,-1.39,-1.39,0.0],
    cross_level:true,activation:'modrelu',chunk_size:256,
    multi_scale_loss:false,aux_layer_stride:3,max_aux_offset:32,use_reverse_assoc:true,
  },
};

export const V6_PRESETS = {
  'tiny': {dim:64,state_dim:128,num_layers:4,num_banks:2,bank_expand:2,single_bank:false,
    gated_state_protection:false,pam_num_heads:6,pam_head_dim:64,interleave_pam:false,
    pam_qk_norm:false,pam_rope:false,pam_fused_qkv:false,use_attention:false,attn_every:0,
    attn_num_heads:8,attn_window_size:256,attn_mode:'softmax',
    num_wm_slots:0,num_im_slots:0,num_episodic_slots:0,vocab_size:50257,max_seq_len:1024},
  'small-matched': {dim:128,state_dim:512,num_layers:12,num_banks:2,bank_expand:4,single_bank:false,
    gated_state_protection:false,pam_num_heads:6,pam_head_dim:64,interleave_pam:false,
    pam_qk_norm:false,pam_rope:false,pam_fused_qkv:false,use_attention:false,attn_every:0,
    attn_num_heads:8,attn_window_size:256,attn_mode:'softmax',
    num_wm_slots:0,num_im_slots:0,num_episodic_slots:0,vocab_size:50257,max_seq_len:1024},
  'medium-pam': {dim:384,state_dim:0,num_layers:16,num_banks:1,bank_expand:3,single_bank:true,
    gated_state_protection:true,pam_num_heads:6,pam_head_dim:64,interleave_pam:false,
    pam_qk_norm:false,pam_rope:false,pam_fused_qkv:false,use_attention:false,attn_every:0,
    attn_num_heads:8,attn_window_size:256,attn_mode:'softmax',
    num_wm_slots:0,num_im_slots:0,num_episodic_slots:0,vocab_size:50257,max_seq_len:1024},
  'medium-pam-v2': {dim:384,state_dim:0,num_layers:16,num_banks:1,bank_expand:3,single_bank:true,
    gated_state_protection:true,pam_num_heads:6,pam_head_dim:64,interleave_pam:true,
    pam_qk_norm:false,pam_rope:false,pam_fused_qkv:false,use_attention:false,attn_every:0,
    attn_num_heads:8,attn_window_size:256,attn_mode:'softmax',
    num_wm_slots:0,num_im_slots:0,num_episodic_slots:0,vocab_size:50257,max_seq_len:1024},
  'medium-pam-v3': {dim:384,state_dim:0,num_layers:16,num_banks:1,bank_expand:3,single_bank:true,
    gated_state_protection:true,pam_num_heads:6,pam_head_dim:64,interleave_pam:true,
    pam_qk_norm:false,pam_rope:true,pam_fused_qkv:true,use_attention:false,attn_every:0,
    attn_num_heads:8,attn_window_size:256,attn_mode:'softmax',
    num_wm_slots:0,num_im_slots:0,num_episodic_slots:0,vocab_size:50257,max_seq_len:1024},
  'medium-pam-v3-attn': {dim:384,state_dim:0,num_layers:16,num_banks:1,bank_expand:3,single_bank:true,
    gated_state_protection:true,pam_num_heads:6,pam_head_dim:64,interleave_pam:true,
    pam_qk_norm:false,pam_rope:true,pam_fused_qkv:true,use_attention:true,attn_every:4,
    attn_num_heads:6,attn_window_size:256,attn_mode:'softmax',
    num_wm_slots:0,num_im_slots:0,num_episodic_slots:0,vocab_size:50257,max_seq_len:1024},
  'medium-pam-v3-pia': {dim:384,state_dim:0,num_layers:16,num_banks:1,bank_expand:3,single_bank:true,
    gated_state_protection:true,pam_num_heads:6,pam_head_dim:64,interleave_pam:true,
    pam_qk_norm:false,pam_rope:true,pam_fused_qkv:true,use_attention:true,attn_every:4,
    attn_num_heads:6,attn_window_size:256,attn_mode:'interference',attn_rope:true,attn_fused_qkv:true,
    num_wm_slots:0,num_im_slots:0,num_episodic_slots:0,vocab_size:50257,max_seq_len:1024},
  'medium-rebalanced-gsp': {dim:192,state_dim:1536,num_layers:16,num_banks:1,bank_expand:3,single_bank:true,
    gated_state_protection:true,pam_num_heads:6,pam_head_dim:64,interleave_pam:false,
    pam_qk_norm:false,pam_rope:false,pam_fused_qkv:false,use_attention:false,attn_every:0,
    attn_num_heads:8,attn_window_size:256,attn_mode:'softmax',
    num_wm_slots:0,num_im_slots:0,num_episodic_slots:0,vocab_size:50257,max_seq_len:1024},
};

export function getDtSchedule(cfg) {
  const n = cfg.n_layers || cfg.num_layers;
  if (cfg.hierarchical_dt === false && !cfg.dt_bias_schedule) return Array(n).fill(-4.0);
  if (cfg.dt_bias_schedule && cfg.dt_bias_schedule.length === n) return [...cfg.dt_bias_schedule];
  const s = [];
  for (let i = 0; i < n; i++) s.push(-6.91 + 6.91 * i / Math.max(n - 1, 1));
  return s;
}

export function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

export function fmtNum(n) {
  if (n >= 1e9) return (n/1e9).toFixed(2)+'B';
  if (n >= 1e6) return (n/1e6).toFixed(2)+'M';
  if (n >= 1e3) return (n/1e3).toFixed(1)+'K';
  return n.toString();
}

export function complexLinearParams(inD, outD, bias) {
  let p = 2 * inD * outD;
  if (bias) p += 2 * outD;
  return p;
}

export function linearParams(inD, outD) { return inD * outD + outD; }

export function computeBlockParams(cfg, tog, layerIdx) {
  const dim = cfg.dim;
  const n_heads = cfg.n_heads || cfg.pam_num_heads || 6;
  const head_dim = cfg.head_dim || cfg.pam_head_dim || 64;
  const inner = n_heads * head_dim;
  const expand = cfg.expand || cfg.bank_expand || 3;
  const hidden = dim * expand;
  let cguP = 0;
  if (tog.dual_banks) {
    cguP = 2 * (complexLinearParams(dim, hidden, false) * 2 + complexLinearParams(hidden, dim, false) + hidden + dim);
    cguP += dim * 4;
  } else {
    cguP = complexLinearParams(dim, hidden, false) * 2 + complexLinearParams(hidden, dim, false);
    const act = cfg.activation || 'modrelu';
    if (act === 'modrelu') cguP += hidden;
    else if (act === 'swish') cguP += hidden * 2;
    else cguP += hidden * 4;
  }
  let pamP = 0;
  if (tog.fused_qkv) pamP += complexLinearParams(dim, 3 * inner, false);
  else pamP += 3 * complexLinearParams(dim, inner, false);
  pamP += complexLinearParams(inner, dim, false);
  pamP += linearParams(dim * 2, n_heads) + n_heads;
  if (tog.use_gsp) pamP += linearParams(dim, n_heads);
  if (tog.cross_level && layerIdx > 0) pamP += complexLinearParams(dim, inner, false);
  if (tog.use_reverse_assoc) pamP += 1;
  let attnP = 0;
  if (tog.attention) attnP = complexLinearParams(dim, dim, false) * 4 + dim;
  const normP = dim * 2, scaleP = 2;
  let memP = 0;
  if (tog.working_memory) memP += dim * 64 * 2 + dim * 2;
  if (tog.internal_memory) memP += dim * 32 * 2;
  if (tog.episodic_memory) memP += dim * 32 * 2 + dim;
  return { cgu: cguP, pam: pamP, attn: attnP, norms: normP, scales: scaleP, memory: memP,
           total: cguP + pamP + attnP + normP + scaleP + memP };
}

export function computeTotalParams(cfg, tog) {
  const dim = cfg.dim, vocab = cfg.vocab_size || 50257;
  const nLayers = cfg.n_layers || cfg.num_layers;
  let embedP = 2 * vocab * dim + dim;
  let blocksP = 0;
  for (let i = 0; i < nLayers; i++) blocksP += computeBlockParams(cfg, tog, i).total;
  let headP = complexLinearParams(dim, dim, true) + dim * 2;
  let auxP = 0;
  if (tog.multi_scale_loss) {
    const stride = cfg.aux_layer_stride || 3;
    for (let i = 0; i < nLayers; i += stride) auxP += dim;
  }
  return { embed: embedP, blocks: blocksP, head: headP, aux: auxP,
           total: embedP + blocksP + headP + auxP };
}
