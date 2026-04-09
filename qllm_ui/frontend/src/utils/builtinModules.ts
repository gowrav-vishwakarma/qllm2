import type { ModuleDef } from "@/types";

export const builtinModules: ModuleDef[] = [
  // ═══════════════════════════════════════════════════════════════════════════
  // QLLM Custom — exact code from v7/model.py
  // ═══════════════════════════════════════════════════════════════════════════

  {
    id: "ModReLU",
    name: "ModReLU",
    category: "QLLM Custom",
    code: `class ModReLU(nn.Module):
    """Phase-preserving activation: threshold on magnitude, phase untouched."""

    def __init__(self, dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.full((dim,), -0.1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mag = cabs(z)
        activated = F.relu(mag + self.bias)
        phase = z / (mag.unsqueeze(-1) + 1e-8)
        return phase * activated.unsqueeze(-1)`,
    inputs: [{ name: "z", type: "Tensor" }],
    outputs: [{ name: "out", type: "Tensor" }],
    constructorParams: [{ name: "dim", type: "int" }],
  },
  {
    id: "ModSwish",
    name: "ModSwish",
    category: "QLLM Custom",
    code: `class ModSwish(nn.Module):
    """Smooth phase-preserving activation: Swish on magnitude, phase untouched.

    Replaces hard ReLU threshold with mag * sigmoid(beta * mag + bias).
    No dead neurons, non-zero gradient below threshold, learnable sharpness.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.ones(dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mag = cabs(z)
        activated = mag * torch.sigmoid(self.beta * mag + self.bias)
        phase = z / (mag.unsqueeze(-1) + 1e-8)
        return phase * activated.unsqueeze(-1)`,
    inputs: [{ name: "z", type: "Tensor" }],
    outputs: [{ name: "out", type: "Tensor" }],
    constructorParams: [{ name: "dim", type: "int" }],
  },
  {
    id: "PhaseModulatedActivation",
    name: "PhaseModulatedActivation",
    category: "QLLM Custom",
    code: `class PhaseModulatedActivation(nn.Module):
    """Activation that couples magnitude and phase.

    Extends ModSwish with magnitude-dependent phase rotation: high-magnitude
    signals get different phase rotations than low-magnitude ones. Initialized
    with phase_alpha=0 so it starts as ModSwish and discovers coupling via
    gradient descent.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.ones(dim))
        self.phase_alpha = nn.Parameter(torch.zeros(dim))
        self.phase_beta = nn.Parameter(torch.zeros(dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mag = cabs(z)
        activated = mag * torch.sigmoid(self.beta * mag + self.bias)
        phase = z / (mag.unsqueeze(-1) + 1e-8)
        theta = self.phase_alpha * mag + self.phase_beta
        rot = torch.stack([theta.cos(), theta.sin()], dim=-1)
        phase = cmul(phase, rot)
        return phase * activated.unsqueeze(-1)`,
    inputs: [{ name: "z", type: "Tensor" }],
    outputs: [{ name: "out", type: "Tensor" }],
    constructorParams: [{ name: "dim", type: "int" }],
  },
  {
    id: "ComplexLinear",
    name: "ComplexLinear",
    category: "QLLM Custom",
    code: `class ComplexLinear(nn.Module):
    """Complex linear via split real/imag matmuls with orthogonal init.

    (a_r + i*a_i)(w_r + i*w_i) computed as four F.linear calls:
        y_r = xr @ Wr^T - xi @ Wi^T
        y_i = xr @ Wi^T + xi @ Wr^T
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        scale = (2 / (in_dim + out_dim)) ** 0.5
        self.weight_real = nn.Parameter(torch.empty(out_dim, in_dim))
        self.weight_imag = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.orthogonal_(self.weight_real, gain=scale)
        nn.init.orthogonal_(self.weight_imag, gain=scale)
        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_dim))
            self.bias_imag = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = x[..., 0], x[..., 1]
        yr = F.linear(xr, self.weight_real) - F.linear(xi, self.weight_imag)
        yi = F.linear(xr, self.weight_imag) + F.linear(xi, self.weight_real)
        if self.bias_real is not None:
            yr = yr + self.bias_real
            yi = yi + self.bias_imag
        return torch.stack([yr, yi], dim=-1)`,
    inputs: [{ name: "x", type: "Tensor" }],
    outputs: [{ name: "out", type: "Tensor" }],
    constructorParams: [
      { name: "in_dim", type: "int" },
      { name: "out_dim", type: "int" },
      { name: "bias", type: "bool", default: true },
    ],
  },
  {
    id: "ComplexNorm",
    name: "ComplexNorm",
    category: "QLLM Custom",
    code: `class ComplexNorm(nn.Module):
    """RMSNorm for complex: normalize magnitude, preserve phase."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mag = torch.sqrt(z[..., 0].square() + z[..., 1].square() + 1e-8)
        rms = torch.sqrt(mag.square().mean(dim=-1, keepdim=True) + self.eps)
        scaled = (mag / rms) * self.scale
        phase = z / (mag.unsqueeze(-1) + 1e-8)
        return phase * scaled.unsqueeze(-1)`,
    inputs: [{ name: "z", type: "Tensor" }],
    outputs: [{ name: "out", type: "Tensor" }],
    constructorParams: [
      { name: "dim", type: "int" },
      { name: "eps", type: "float", default: 1e-6 },
    ],
  },
  {
    id: "ComplexGatedUnit",
    name: "ComplexGatedUnit",
    category: "QLLM Custom",
    code: `class ComplexGatedUnit(nn.Module):
    """SwiGLU-style complex gating: magnitude gates how much, phase gates rotation.

    gate = gate_proj(z)
    up   = activation(up_proj(z))
    gated = complex_mul(normalize(gate), up) * sigmoid(|gate|)
    out  = down_proj(gated)
    """

    def __init__(self, dim: int, expand: int = 3, activation: str = 'modrelu'):
        super().__init__()
        hidden = dim * expand
        self.gate_proj = ComplexLinear(dim, hidden, bias=False)
        self.up_proj = ComplexLinear(dim, hidden, bias=False)
        self.down_proj = ComplexLinear(hidden, dim, bias=False)
        self.act = _build_activation(activation, hidden)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(z)
        up = self.act(self.up_proj(z))
        # Complex gating: sigmoid(|gate|) * cmul(phase(gate), up)
        gmag = cabs(gate)
        gate_mag = torch.sigmoid(gmag)
        phase = gate / (gmag.unsqueeze(-1) + 1e-8)
        pr, pi = phase[..., 0], phase[..., 1]
        ur, ui = up[..., 0], up[..., 1]
        out_r = (pr * ur - pi * ui) * gate_mag
        out_i = (pr * ui + pi * ur) * gate_mag
        gated = torch.stack([out_r, out_i], dim=-1)
        return self.down_proj(gated)`,
    inputs: [{ name: "z", type: "Tensor" }],
    outputs: [{ name: "out", type: "Tensor" }],
    constructorParams: [
      { name: "dim", type: "int" },
      { name: "expand", type: "int", default: 3 },
      { name: "activation", type: "str", default: "modrelu", description: "modrelu | swish | phase_mod" },
    ],
    subGraph: {
      nodes: [
        { id: "in_z", type: "__input__", label: "Input", params: {}, portName: "z" },
        { id: "gate_proj", type: "ComplexLinear", label: "gate_proj", params: { in_dim: 384, out_dim: 1152, bias: false } },
        { id: "up_proj", type: "ComplexLinear", label: "up_proj", params: { in_dim: 384, out_dim: 1152, bias: false } },
        { id: "act", type: "ModReLU", label: "activation", params: { dim: 1152 } },
        { id: "fused_gate", type: "Residual", label: "fused_cgu_gate", params: { scale: 1.0 } },
        { id: "down_proj", type: "ComplexLinear", label: "down_proj", params: { in_dim: 1152, out_dim: 384, bias: false } },
        { id: "out_z", type: "__output__", label: "Output", params: {}, portName: "out" },
      ],
      connections: [
        { source: "in_z", sourcePort: "z", target: "gate_proj", targetPort: "x" },
        { source: "in_z", sourcePort: "z", target: "up_proj", targetPort: "x" },
        { source: "up_proj", sourcePort: "out", target: "act", targetPort: "z" },
        { source: "gate_proj", sourcePort: "out", target: "fused_gate", targetPort: "residual" },
        { source: "act", sourcePort: "out", target: "fused_gate", targetPort: "x" },
        { source: "fused_gate", sourcePort: "out", target: "down_proj", targetPort: "x" },
        { source: "down_proj", sourcePort: "out", target: "out_z", targetPort: "out" },
      ],
    },
  },
  {
    id: "ComplexEmbed",
    name: "ComplexEmbed",
    category: "QLLM Custom",
    code: `class ComplexEmbed(nn.Module):
    """Embed tokens into complex space: real + imaginary components."""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.dim = dim
        self.embed_real = nn.Embedding(vocab_size, dim)
        self.embed_imag = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.embed_real.weight, std=0.02)
        nn.init.normal_(self.embed_imag.weight, std=0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.embed_real(ids), self.embed_imag(ids)], dim=-1)`,
    inputs: [{ name: "ids", type: "Tensor" }],
    outputs: [{ name: "out", type: "Tensor" }],
    constructorParams: [
      { name: "vocab_size", type: "int", default: 50257 },
      { name: "dim", type: "int" },
    ],
  },
  {
    id: "PhaseAssociativeLayer",
    name: "PhaseAssociativeLayer",
    category: "QLLM Custom",
    code: `class PhaseAssociativeLayer(nn.Module):
    r"""Matrix-state memory with complex-conjugate retrieval.

        S_t = gamma_t * S_{t-1} + V_t (x) K_t^*
        Y_t = S_t * Q_t

    Training: O(T^2) dual form (GPU-friendly matmuls, no sequential loop).
    Inference: O(1) per token recurrent form.
    Chunked dual form: O(T*C) for long sequences.
    """

    def __init__(self, dim: int, n_heads: int = 6, head_dim: int = 64,
                 use_rope: bool = True, use_gsp: bool = True,
                 fused_qkv: bool = True, qk_norm: bool = False,
                 hierarchical_dt: bool = True, dt_bias_init: float = -4.0,
                 chunk_size: int = 256, use_reverse_assoc: bool = True,
                 cross_level: bool = False, layer_idx: int = 0,
                 dropout: float = 0.1, max_seq_len: int = 2048):
        super().__init__()
        self.num_heads = n_heads
        self.head_dim = head_dim
        inner = n_heads * head_dim
        self.inner_dim = inner
        self.dim = dim
        self.fused_qkv = fused_qkv
        self.use_rope = use_rope
        self.use_gsp = use_gsp
        self.qk_norm = qk_norm

        if fused_qkv:
            self.qkv_proj = ComplexLinear(dim, 3 * inner, bias=False)
        else:
            self.q_proj = ComplexLinear(dim, inner, bias=False)
            self.k_proj = ComplexLinear(dim, inner, bias=False)
            self.v_proj = ComplexLinear(dim, inner, bias=False)
        self.o_proj = ComplexLinear(inner, dim, bias=False)

        # Hierarchical dt_bias: each layer gets its own base decay rate
        self.dt_proj = nn.Linear(dim * 2, n_heads)
        self.dt_bias = nn.Parameter(torch.zeros(n_heads) + dt_bias_init)

        if use_gsp:
            self.protect_gate = nn.Linear(dim, n_heads)
            nn.init.constant_(self.protect_gate.bias, -3.0)

        # Cross-level drift: project higher layer's PAM output into Q-space
        if cross_level and layer_idx > 0:
            self.drift_proj = ComplexLinear(dim, inner, bias=False)

        if use_rope:
            freqs = 1.0 / (10000.0 ** (torch.arange(head_dim).float() / head_dim))
            positions = torch.arange(max_seq_len).float()
            angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
            self.register_buffer('rope_cache',
                torch.stack([angles.cos(), angles.sin()], dim=-1), persistent=False)

        self.dropout = nn.Dropout(dropout)

        self.use_reverse_assoc = use_reverse_assoc
        if use_reverse_assoc:
            self.rev_scale = nn.Parameter(torch.zeros(1))

        self.chunk_size = chunk_size
        _causal_size = chunk_size if chunk_size > 0 else max_seq_len
        self.register_buffer('_causal',
            torch.tril(torch.ones(_causal_size, _causal_size)), persistent=False)

    @staticmethod
    def _dual_form_block(q_s, k, v_prime, gamma, causal_mask, rev_scale=None):
        """Dual form on a single block. q_s is pre-scaled by d^{-0.5}."""
        B, H, T = gamma.shape
        # Decay matrix from gamma
        log_gamma = torch.log(gamma + 1e-6)
        C = torch.cumsum(-log_gamma, dim=-1)
        log_D = (C.unsqueeze(-1) - C.unsqueeze(-2)).transpose(-1, -2)
        D = torch.exp(log_D.clamp(max=0.0)) * causal_mask

        qr, qi = q_s[..., 0], q_s[..., 1]
        kr, ki = k[..., 0], k[..., 1]
        wr = qr @ kr.transpose(-1, -2) + qi @ ki.transpose(-1, -2)
        wi = qi @ kr.transpose(-1, -2) - qr @ ki.transpose(-1, -2)

        ar, ai = wr * D, wi * D
        vpr, vpi = v_prime[..., 0], v_prime[..., 1]
        yr = ar @ vpr - ai @ vpi
        yi = ar @ vpi + ai @ vpr
        y = torch.stack([yr, yi], dim=-1)

        if rev_scale is not None:
            ar_rev = wr.transpose(-1, -2) * D
            ai_rev = wi.transpose(-1, -2) * D
            yr_rev = ar_rev @ vpr - ai_rev @ vpi
            yi_rev = ar_rev @ vpi + ai_rev @ vpr
            y = y + rev_scale * torch.stack([yr_rev, yi_rev], dim=-1)

        D_last = D[:, :, -1, :]
        wv_r = vpr * D_last.unsqueeze(-1)
        wv_i = vpi * D_last.unsqueeze(-1)
        sr = wv_r.transpose(-1, -2) @ kr + wv_i.transpose(-1, -2) @ ki
        si = wv_i.transpose(-1, -2) @ kr - wv_r.transpose(-1, -2) @ ki
        S_block = torch.stack([sr, si], dim=-1)
        return y, S_block

    def forward(self, x: torch.Tensor, state=None, step_offset: int = 0,
                drift_signal=None):
        B, T, _, _ = x.shape
        H, d = self.num_heads, self.head_dim

        # 1. Q/K/V projections
        if self.fused_qkv:
            qkv = self.qkv_proj(x).view(B, T, 3, H, d, 2)
            q = qkv[:, :, 0].transpose(1, 2).contiguous()
            k = qkv[:, :, 1].transpose(1, 2).contiguous()
            v = qkv[:, :, 2].transpose(1, 2).contiguous()
        else:
            q = self.q_proj(x).view(B, T, H, d, 2).transpose(1, 2)
            k = self.k_proj(x).view(B, T, H, d, 2).transpose(1, 2)
            v = self.v_proj(x).view(B, T, H, d, 2).transpose(1, 2)

        # 1b. Complex RoPE on Q, K
        if self.use_rope:
            pos = self.rope_cache[step_offset:step_offset + T].to(dtype=x.dtype)
            q = cmul(q, pos)
            k = cmul(k, pos)

        # 1c. Optional QK normalization
        if self.qk_norm:
            q = cnormalize(q)
            k = cnormalize(k)

        # 1d. Cross-level drift: bias Q toward higher layer's goal
        if drift_signal is not None and hasattr(self, 'drift_proj'):
            drift_q = self.drift_proj(drift_signal).view(B, T, H, d, 2).transpose(1, 2)
            q = q + drift_q

        # 2. Data-dependent decay + GSP
        x_flat = to_real_concat(x)
        dt = F.softplus(self.dt_proj(x_flat) + self.dt_bias).transpose(1, 2)

        if self.use_gsp:
            p = torch.sigmoid(self.protect_gate(cabs(x))).transpose(1, 2)
            gamma = torch.exp(-dt) * (1 - p) + p
            v_prime = v * (1 - p).unsqueeze(-1).unsqueeze(-1)
        else:
            gamma = torch.exp(-dt)
            v_prime = v

        # 3. Dual Form training (no sequential loop)
        _rev = self.rev_scale if self.use_reverse_assoc else None
        scale = d ** -0.5
        q_s = q * scale
        causal = self._causal[:T, :T]
        y, new_state = self._dual_form_block(q_s, k, v_prime, gamma, causal, _rev)

        # 4. Output projection + dropout
        y = y.transpose(1, 2).contiguous().view(B, T, self.inner_dim, 2)
        out = self.o_proj(y)
        if self.training:
            mask = self.dropout(torch.ones(B, T, self.dim, device=x.device))
            out = out * mask.unsqueeze(-1)
        return out, new_state`,
    inputs: [{ name: "x", type: "Tensor" }],
    outputs: [
      { name: "out", type: "Tensor" },
      { name: "state", type: "Tensor" },
    ],
    constructorParams: [
      { name: "dim", type: "int" },
      { name: "n_heads", type: "int", default: 6 },
      { name: "head_dim", type: "int", default: 64 },
      { name: "use_rope", type: "bool", default: true },
      { name: "use_gsp", type: "bool", default: true },
      { name: "fused_qkv", type: "bool", default: true },
      { name: "chunk_size", type: "int", default: 256 },
      { name: "use_reverse_assoc", type: "bool", default: true },
      { name: "dropout", type: "float", default: 0.1 },
      { name: "max_seq_len", type: "int", default: 2048 },
      { name: "dt_bias_init", type: "float", default: -4.0, description: "Hierarchical: -6.91 (global) to 0.0 (step)" },
      { name: "layer_idx", type: "int", default: 0 },
    ],
    subGraph: {
      nodes: [
        { id: "in_x", type: "__input__", label: "Input", params: {}, portName: "x" },
        { id: "qkv_proj", type: "ComplexLinear", label: "qkv_proj (fused Q,K,V)", params: { in_dim: 384, out_dim: 1152, bias: false } },
        { id: "rope", type: "ComplexLinear", label: "RoPE (complex rotation)", params: { in_dim: 384, out_dim: 384, bias: false } },
        { id: "dt_proj", type: "nn.Linear", label: "dt_proj (decay rates)", params: { in_features: 768, out_features: 6 } },
        { id: "gsp_gate", type: "nn.Linear", label: "GSP protect_gate", params: { in_features: 384, out_features: 6 } },
        { id: "dual_form", type: "ComplexLinear", label: "dual_form_block (Q@K^T * decay * V)", params: { in_dim: 384, out_dim: 384, bias: false } },
        { id: "o_proj", type: "ComplexLinear", label: "o_proj", params: { in_dim: 384, out_dim: 384, bias: false } },
        { id: "dropout", type: "Dropout", label: "dropout", params: { p: 0.1 } },
        { id: "out_y", type: "__output__", label: "Output: out", params: {}, portName: "out" },
        { id: "out_state", type: "__output__", label: "Output: state", params: {}, portName: "state" },
      ],
      connections: [
        { source: "in_x", sourcePort: "x", target: "qkv_proj", targetPort: "x" },
        { source: "qkv_proj", sourcePort: "out", target: "rope", targetPort: "x" },
        { source: "in_x", sourcePort: "x", target: "dt_proj", targetPort: "x" },
        { source: "in_x", sourcePort: "x", target: "gsp_gate", targetPort: "x" },
        { source: "rope", sourcePort: "out", target: "dual_form", targetPort: "x" },
        { source: "dt_proj", sourcePort: "out", target: "dual_form", targetPort: "x" },
        { source: "gsp_gate", sourcePort: "out", target: "dual_form", targetPort: "x" },
        { source: "dual_form", sourcePort: "out", target: "o_proj", targetPort: "x" },
        { source: "o_proj", sourcePort: "out", target: "dropout", targetPort: "x" },
        { source: "dropout", sourcePort: "out", target: "out_y", targetPort: "out" },
        { source: "dual_form", sourcePort: "out", target: "out_state", targetPort: "state" },
      ],
    },
  },
  {
    id: "V7Block",
    name: "V7Block",
    category: "QLLM Custom",
    code: `class V7Block(nn.Module):
    """Pre-norm residual: CGU for channel mixing, PAM for sequence mixing.

    Forward:
        cgu_out = CGU(norm1(x))
        x = x + cgu_out * cgu_scale       # residual 1
        pam_out, state = PAM(norm2(x))
        x = x + pam_out * pam_scale        # residual 2
    """

    def __init__(self, dim: int, expand: int = 3, activation: str = 'modrelu',
                 dropout: float = 0.1, n_heads: int = 6, head_dim: int = 64,
                 use_rope: bool = True, use_gsp: bool = True,
                 fused_qkv: bool = True, chunk_size: int = 256,
                 use_reverse_assoc: bool = True, max_seq_len: int = 2048,
                 dt_bias_init: float = -4.0, layer_idx: int = 0,
                 cross_level: bool = False):
        super().__init__()
        self.norm1 = ComplexNorm(dim)
        self.cgu = ComplexGatedUnit(dim, expand, activation=activation)
        self.cgu_scale = nn.Parameter(torch.tensor(1.0))
        self.cgu_dropout = nn.Dropout(dropout)
        self.norm2 = ComplexNorm(dim)
        self.pam = PhaseAssociativeLayer(
            dim=dim, n_heads=n_heads, head_dim=head_dim,
            use_rope=use_rope, use_gsp=use_gsp, fused_qkv=fused_qkv,
            chunk_size=chunk_size, use_reverse_assoc=use_reverse_assoc,
            max_seq_len=max_seq_len, dt_bias_init=dt_bias_init,
            layer_idx=layer_idx, cross_level=cross_level, dropout=dropout,
        )
        self.pam_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, pam_state=None, step_offset: int = 0,
                drift_signal=None):
        cgu_out = self.cgu(self.norm1(x))
        if self.training:
            drop_mask = self.cgu_dropout(
                torch.ones(cgu_out.shape[:-1], device=cgu_out.device)
            )
            cgu_out = cgu_out * drop_mask.unsqueeze(-1)
        x = x + cgu_out * self.cgu_scale
        pam_out, new_state = self.pam(
            self.norm2(x), state=pam_state, step_offset=step_offset,
            drift_signal=drift_signal,
        )
        x = x + pam_out * self.pam_scale
        return x, new_state, pam_out`,
    inputs: [{ name: "x", type: "Tensor" }],
    outputs: [
      { name: "out", type: "Tensor" },
      { name: "state", type: "Tensor" },
      { name: "pam_out", type: "Tensor" },
    ],
    constructorParams: [
      { name: "dim", type: "int" },
      { name: "expand", type: "int", default: 3 },
      { name: "activation", type: "str", default: "modrelu" },
      { name: "dropout", type: "float", default: 0.1 },
      { name: "n_heads", type: "int", default: 6 },
      { name: "head_dim", type: "int", default: 64 },
      { name: "use_rope", type: "bool", default: true },
      { name: "use_gsp", type: "bool", default: true },
      { name: "chunk_size", type: "int", default: 256 },
      { name: "use_reverse_assoc", type: "bool", default: true },
      { name: "max_seq_len", type: "int", default: 2048 },
      { name: "dt_bias_init", type: "float", default: -4.0, description: "Hierarchical dt bias for this layer" },
      { name: "layer_idx", type: "int", default: 0 },
    ],
    subGraph: {
      nodes: [
        { id: "in_x", type: "__input__", label: "Input", params: {}, portName: "x" },
        { id: "norm1", type: "ComplexNorm", label: "norm1", params: { dim: 384 } },
        { id: "cgu", type: "ComplexGatedUnit", label: "CGU", params: { dim: 384, expand: 3, activation: "modrelu" } },
        { id: "cgu_scale", type: "Residual", label: "x + cgu*scale", params: { scale: 1.0 } },
        { id: "norm2", type: "ComplexNorm", label: "norm2", params: { dim: 384 } },
        { id: "pam", type: "PhaseAssociativeLayer", label: "PAM", params: { dim: 384, n_heads: 6, head_dim: 64 } },
        { id: "pam_scale", type: "Residual", label: "x + pam*0.1", params: { scale: 0.1 } },
        { id: "out_x", type: "__output__", label: "Output: out", params: {}, portName: "out" },
        { id: "out_state", type: "__output__", label: "Output: state", params: {}, portName: "state" },
        { id: "out_pam", type: "__output__", label: "Output: pam_out", params: {}, portName: "pam_out" },
      ],
      connections: [
        { source: "in_x", sourcePort: "x", target: "norm1", targetPort: "z" },
        { source: "norm1", sourcePort: "out", target: "cgu", targetPort: "z" },
        { source: "cgu", sourcePort: "out", target: "cgu_scale", targetPort: "x" },
        { source: "in_x", sourcePort: "x", target: "cgu_scale", targetPort: "residual" },
        { source: "cgu_scale", sourcePort: "out", target: "norm2", targetPort: "z" },
        { source: "norm2", sourcePort: "out", target: "pam", targetPort: "x" },
        { source: "pam", sourcePort: "out", target: "pam_scale", targetPort: "x" },
        { source: "cgu_scale", sourcePort: "out", target: "pam_scale", targetPort: "residual" },
        { source: "pam_scale", sourcePort: "out", target: "out_x", targetPort: "out" },
        { source: "pam", sourcePort: "state", target: "out_state", targetPort: "state" },
        { source: "pam", sourcePort: "out", target: "out_pam", targetPort: "pam_out" },
      ],
    },
  },
  {
    id: "V6Block",
    name: "V6Block (interleaved)",
    category: "QLLM Custom",
    code: `class V6Block(nn.Module):
    """V6 single-bank interleaved CGU+PAM block.

    From PhaseFieldBackbone (single_bank=True, interleave_pam=True):
        residual = z
        out = cgu(norm(z))
        z = residual + out * feat_scale      # channel mixing
        residual = z
        h, state = pam(pam_norm(z))
        z = residual + h * pam_scale          # sequence mixing
    """

    def __init__(self, dim: int, expand: int = 3, dropout: float = 0.1,
                 n_heads: int = 6, head_dim: int = 64,
                 use_rope: bool = True, use_gsp: bool = True,
                 fused_qkv: bool = True, chunk_size: int = 0,
                 use_reverse_assoc: bool = False,
                 max_seq_len: int = 2048, dt_bias_init: float = -4.0,
                 layer_idx: int = 0):
        super().__init__()
        self.norm = ComplexNorm(dim)
        self.cgu = ComplexGatedUnit(dim, expand)
        self.dropout = nn.Dropout(dropout)
        self.feat_scale = nn.Parameter(torch.tensor(1.0))
        self.pam_norm = ComplexNorm(dim)
        self.pam = PhaseAssociativeLayer(
            dim=dim, n_heads=n_heads, head_dim=head_dim,
            use_rope=use_rope, use_gsp=use_gsp, fused_qkv=fused_qkv,
            chunk_size=chunk_size, use_reverse_assoc=use_reverse_assoc,
            max_seq_len=max_seq_len, dt_bias_init=dt_bias_init,
            layer_idx=layer_idx, dropout=dropout,
        )
        self.pam_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, z: torch.Tensor, pam_state=None, step_offset: int = 0):
        residual = z
        out = self.cgu(self.norm(z))
        if self.training:
            drop_mask = self.dropout(
                torch.ones(out.shape[:-1], device=out.device)
            )
            out = out * drop_mask.unsqueeze(-1)
        z = residual + out * self.feat_scale
        residual = z
        h_out, new_state = self.pam(
            self.pam_norm(z), state=pam_state, step_offset=step_offset,
        )
        z = residual + h_out * self.pam_scale
        return z, new_state`,
    inputs: [{ name: "z", type: "Tensor" }],
    outputs: [
      { name: "out", type: "Tensor" },
      { name: "state", type: "Tensor" },
    ],
    constructorParams: [
      { name: "dim", type: "int" },
      { name: "expand", type: "int", default: 3 },
      { name: "dropout", type: "float", default: 0.1 },
      { name: "n_heads", type: "int", default: 6 },
      { name: "head_dim", type: "int", default: 64 },
      { name: "use_rope", type: "bool", default: true },
      { name: "use_gsp", type: "bool", default: true },
      { name: "fused_qkv", type: "bool", default: true },
      { name: "chunk_size", type: "int", default: 0 },
      { name: "use_reverse_assoc", type: "bool", default: false },
      { name: "max_seq_len", type: "int", default: 2048 },
      { name: "dt_bias_init", type: "float", default: -4.0 },
      { name: "layer_idx", type: "int", default: 0 },
    ],
    subGraph: {
      nodes: [
        { id: "in_z", type: "__input__", label: "Input", params: {}, portName: "z" },
        { id: "norm", type: "ComplexNorm", label: "norm", params: { dim: 384 } },
        { id: "cgu", type: "ComplexGatedUnit", label: "CGU", params: { dim: 384, expand: 3 } },
        { id: "dropout", type: "Dropout", label: "dropout", params: { p: 0.1 } },
        { id: "cgu_res", type: "Residual", label: "z + cgu*feat_scale", params: { scale: 1.0 } },
        { id: "pam_norm", type: "ComplexNorm", label: "pam_norm", params: { dim: 384 } },
        { id: "pam", type: "PhaseAssociativeLayer", label: "PAM", params: { dim: 384, n_heads: 6, head_dim: 64 } },
        { id: "pam_res", type: "Residual", label: "z + pam*0.1", params: { scale: 0.1 } },
        { id: "out_z", type: "__output__", label: "Output: out", params: {}, portName: "out" },
        { id: "out_state", type: "__output__", label: "Output: state", params: {}, portName: "state" },
      ],
      connections: [
        { source: "in_z", sourcePort: "z", target: "norm", targetPort: "z" },
        { source: "norm", sourcePort: "out", target: "cgu", targetPort: "z" },
        { source: "cgu", sourcePort: "out", target: "dropout", targetPort: "x" },
        { source: "dropout", sourcePort: "out", target: "cgu_res", targetPort: "x" },
        { source: "in_z", sourcePort: "z", target: "cgu_res", targetPort: "residual" },
        { source: "cgu_res", sourcePort: "out", target: "pam_norm", targetPort: "z" },
        { source: "pam_norm", sourcePort: "out", target: "pam", targetPort: "x" },
        { source: "pam", sourcePort: "out", target: "pam_res", targetPort: "x" },
        { source: "cgu_res", sourcePort: "out", target: "pam_res", targetPort: "residual" },
        { source: "pam_res", sourcePort: "out", target: "out_z", targetPort: "out" },
        { source: "pam", sourcePort: "state", target: "out_state", targetPort: "state" },
      ],
    },
  },
  {
    id: "AuxPredHead",
    name: "AuxPredHead",
    category: "QLLM Custom",
    code: `class AuxPredHead(nn.Module):
    """Lightweight per-layer prediction head for multi-scale temporal loss.

    Ties output projection with the embedding weights (like the main LM head)
    so each head only adds ~dim params (one ComplexNorm scale vector).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = ComplexNorm(dim)

    def forward(self, z: torch.Tensor, embed_real_w: torch.Tensor,
                embed_imag_w: torch.Tensor) -> torch.Tensor:
        z = self.norm(z)
        return z[..., 0] @ embed_real_w.T + z[..., 1] @ embed_imag_w.T`,
    inputs: [
      { name: "z", type: "Tensor" },
      { name: "embed_real_w", type: "Tensor" },
      { name: "embed_imag_w", type: "Tensor" },
    ],
    outputs: [{ name: "logits", type: "Tensor" }],
    constructorParams: [{ name: "dim", type: "int" }],
  },
  {
    id: "ComplexLMHead",
    name: "Complex LM Head (Tied)",
    category: "QLLM Custom",
    code: `class ComplexLMHead(nn.Module):
    """Tied complex LM head: logits = z_r @ E_r^T + z_i @ E_i^T

    Pipeline: ComplexLinear -> ComplexNorm -> bilinear match with embeddings.
    """

    def __init__(self, dim: int, vocab_size: int = 50257):
        super().__init__()
        self.proj = ComplexLinear(dim, dim)
        self.norm = ComplexNorm(dim)
        # When used standalone (not tied), add projection to vocab
        self.out_proj = nn.Linear(dim * 2, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lm = self.norm(self.proj(x))
        flat = lm.reshape(*lm.shape[:-2], -1)
        return self.out_proj(flat)`,
    inputs: [{ name: "x", type: "Tensor" }],
    outputs: [{ name: "logits", type: "Tensor" }],
    constructorParams: [
      { name: "dim", type: "int" },
      { name: "vocab_size", type: "int", default: 50257 },
    ],
  },
  {
    id: "ComplexSSM",
    name: "ComplexSSM",
    category: "QLLM Custom",
    code: `class ComplexSSM(nn.Module):
    """Complex-valued State Space Model (V5 architecture)."""
    def __init__(self, dim: int, state_dim: int = 64, n_layers: int = 1):
        super().__init__()
        self.state_dim = state_dim
        self.proj_in = ComplexLinear(dim, state_dim, bias=False)
        self.proj_out = ComplexLinear(state_dim, dim, bias=False)
        self.A = nn.Parameter(torch.randn(state_dim) * 0.01)
        self.B = ComplexLinear(dim, state_dim, bias=False)
        self.norm = ComplexNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(self.norm(x))
        out = self.proj_out(h)
        return out`,
    inputs: [{ name: "x", type: "Tensor" }],
    outputs: [{ name: "out", type: "Tensor" }],
    constructorParams: [
      { name: "dim", type: "int" },
      { name: "state_dim", type: "int", default: 64 },
      { name: "n_layers", type: "int", default: 1 },
    ],
  },
  {
    id: "PhaseAttention",
    name: "PhaseAttention",
    category: "QLLM Custom",
    code: `class PhaseAttention(nn.Module):
    """Sparse windowed attention in complex space (V5/V6)."""
    def __init__(self, dim: int, n_heads: int = 4, window_size: int = 256):
        super().__init__()
        self.n_heads = n_heads
        self.window_size = window_size
        self.q_proj = ComplexLinear(dim, dim, bias=False)
        self.k_proj = ComplexLinear(dim, dim, bias=False)
        self.v_proj = ComplexLinear(dim, dim, bias=False)
        self.out_proj = ComplexLinear(dim, dim, bias=False)
        self.norm = ComplexNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)
        q, k, v = self.q_proj(normed), self.k_proj(normed), self.v_proj(normed)
        out = self.out_proj(v)
        return out`,
    inputs: [{ name: "x", type: "Tensor" }],
    outputs: [{ name: "out", type: "Tensor" }],
    constructorParams: [
      { name: "dim", type: "int" },
      { name: "n_heads", type: "int", default: 4 },
      { name: "window_size", type: "int", default: 256 },
    ],
  },
  {
    id: "WorkingMemory",
    name: "WorkingMemory",
    category: "QLLM Custom",
    code: `class WorkingMemory(nn.Module):
    def __init__(self, dim: int, n_slots: int = 32):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(1, n_slots, dim, 2) * 0.02)
        self.query = ComplexLinear(dim, dim, bias=False)
        self.gate = nn.Linear(dim * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        return x`,
    inputs: [{ name: "x", type: "Tensor" }],
    outputs: [{ name: "out", type: "Tensor" }],
    constructorParams: [
      { name: "dim", type: "int" },
      { name: "n_slots", type: "int", default: 32 },
    ],
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // Standard PyTorch modules
  // ═══════════════════════════════════════════════════════════════════════════

  {
    id: "nn.Linear",
    name: "nn.Linear",
    category: "Standard",
    code: `# PyTorch built-in nn.Linear
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)`,
    inputs: [{ name: "x", type: "Tensor" }],
    outputs: [{ name: "out", type: "Tensor" }],
    constructorParams: [
      { name: "in_features", type: "int" },
      { name: "out_features", type: "int" },
      { name: "bias", type: "bool", default: true },
    ],
  },
  {
    id: "nn.LayerNorm",
    name: "nn.LayerNorm",
    category: "Standard",
    code: `class LayerNormModule(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)`,
    inputs: [{ name: "x", type: "Tensor" }],
    outputs: [{ name: "out", type: "Tensor" }],
    constructorParams: [
      { name: "normalized_shape", type: "int" },
      { name: "eps", type: "float", default: 1e-5 },
    ],
  },
  {
    id: "nn.Embedding",
    name: "nn.Embedding",
    category: "Standard",
    code: `class EmbeddingModule(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids)`,
    inputs: [{ name: "input_ids", type: "Tensor" }],
    outputs: [{ name: "out", type: "Tensor" }],
    constructorParams: [
      { name: "num_embeddings", type: "int", default: 50257 },
      { name: "embedding_dim", type: "int" },
    ],
  },
  {
    id: "MultiheadAttention",
    name: "MultiheadAttention",
    category: "Standard",
    code: `class MultiheadAttentionModule(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x)
        return out`,
    inputs: [{ name: "x", type: "Tensor" }],
    outputs: [{ name: "out", type: "Tensor" }],
    constructorParams: [
      { name: "embed_dim", type: "int" },
      { name: "num_heads", type: "int", default: 8 },
      { name: "dropout", type: "float", default: 0.0 },
    ],
  },
  {
    id: "TransformerEncoderLayer",
    name: "TransformerEncoderLayer",
    category: "Standard",
    code: `class TransformerLayerModule(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)`,
    inputs: [{ name: "x", type: "Tensor" }],
    outputs: [{ name: "out", type: "Tensor" }],
    constructorParams: [
      { name: "d_model", type: "int" },
      { name: "nhead", type: "int", default: 8 },
      { name: "dim_feedforward", type: "int", default: 2048 },
      { name: "dropout", type: "float", default: 0.1 },
    ],
    subGraph: {
      nodes: [
        { id: "in_x", type: "__input__", label: "Input", params: {}, portName: "x" },
        { id: "norm1", type: "nn.LayerNorm", label: "norm1", params: { normalized_shape: 512 } },
        { id: "self_attn", type: "MultiheadAttention", label: "self_attn", params: { embed_dim: 512, num_heads: 8 } },
        { id: "dropout1", type: "Dropout", label: "dropout1", params: { p: 0.1 } },
        { id: "res1", type: "Residual", label: "x + attn", params: { scale: 1.0 } },
        { id: "norm2", type: "nn.LayerNorm", label: "norm2", params: { normalized_shape: 512 } },
        { id: "ffn", type: "FeedForward", label: "FFN", params: { dim: 512, hidden_dim: 2048, dropout: 0.1 } },
        { id: "dropout2", type: "Dropout", label: "dropout2", params: { p: 0.1 } },
        { id: "res2", type: "Residual", label: "x + ffn", params: { scale: 1.0 } },
        { id: "out_x", type: "__output__", label: "Output", params: {}, portName: "out" },
      ],
      connections: [
        { source: "in_x", sourcePort: "x", target: "norm1", targetPort: "x" },
        { source: "norm1", sourcePort: "out", target: "self_attn", targetPort: "x" },
        { source: "self_attn", sourcePort: "out", target: "dropout1", targetPort: "x" },
        { source: "dropout1", sourcePort: "out", target: "res1", targetPort: "x" },
        { source: "in_x", sourcePort: "x", target: "res1", targetPort: "residual" },
        { source: "res1", sourcePort: "out", target: "norm2", targetPort: "x" },
        { source: "norm2", sourcePort: "out", target: "ffn", targetPort: "x" },
        { source: "ffn", sourcePort: "out", target: "dropout2", targetPort: "x" },
        { source: "dropout2", sourcePort: "out", target: "res2", targetPort: "x" },
        { source: "res1", sourcePort: "out", target: "res2", targetPort: "residual" },
        { source: "res2", sourcePort: "out", target: "out_x", targetPort: "out" },
      ],
    },
  },
  {
    id: "FeedForward",
    name: "FeedForward (FFN)",
    category: "Standard",
    code: `class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.gelu(self.w1(x))))`,
    inputs: [{ name: "x", type: "Tensor" }],
    outputs: [{ name: "out", type: "Tensor" }],
    constructorParams: [
      { name: "dim", type: "int" },
      { name: "hidden_dim", type: "int", default: 0, description: "0 = dim*4" },
      { name: "dropout", type: "float", default: 0.0 },
    ],
  },
  {
    id: "SwiGLU",
    name: "SwiGLU FFN",
    category: "Standard",
    code: `class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 0):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8 / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))`,
    inputs: [{ name: "x", type: "Tensor" }],
    outputs: [{ name: "out", type: "Tensor" }],
    constructorParams: [
      { name: "dim", type: "int" },
      { name: "hidden_dim", type: "int", default: 0 },
    ],
  },
  {
    id: "RMSNorm",
    name: "RMSNorm",
    category: "Standard",
    code: `class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight`,
    inputs: [{ name: "x", type: "Tensor" }],
    outputs: [{ name: "out", type: "Tensor" }],
    constructorParams: [
      { name: "dim", type: "int" },
      { name: "eps", type: "float", default: 1e-6 },
    ],
  },
  {
    id: "LMHead",
    name: "LM Head (Projection)",
    category: "Standard",
    code: `class LMHead(nn.Module):
    def __init__(self, dim: int, vocab_size: int = 50257):
        super().__init__()
        self.proj = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)`,
    inputs: [{ name: "x", type: "Tensor" }],
    outputs: [{ name: "logits", type: "Tensor" }],
    constructorParams: [
      { name: "dim", type: "int" },
      { name: "vocab_size", type: "int", default: 50257 },
    ],
  },
  {
    id: "Residual",
    name: "Residual Add",
    category: "Standard",
    code: `class Residual(nn.Module):
    """Adds two tensors (residual connection) with learnable scale."""
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return residual + x * self.scale`,
    inputs: [
      { name: "x", type: "Tensor" },
      { name: "residual", type: "Tensor" },
    ],
    outputs: [{ name: "out", type: "Tensor" }],
    constructorParams: [
      { name: "scale", type: "float", default: 1.0 },
    ],
  },
  {
    id: "Dropout",
    name: "Dropout",
    category: "Standard",
    code: `class DropoutModule(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x)`,
    inputs: [{ name: "x", type: "Tensor" }],
    outputs: [{ name: "out", type: "Tensor" }],
    constructorParams: [
      { name: "p", type: "float", default: 0.1 },
    ],
  },

  // ═══════════════════════════════════════════════════════════════════════════
  // SSM / Mamba
  // ═══════════════════════════════════════════════════════════════════════════

  {
    id: "MambaBlock",
    name: "MambaBlock",
    category: "SSM/Mamba",
    code: `class MambaBlock(nn.Module):
    """Selective State Space Model block (Mamba-style)."""
    def __init__(self, dim: int, state_dim: int = 16, expand: int = 2):
        super().__init__()
        inner_dim = dim * expand
        self.in_proj = nn.Linear(dim, inner_dim * 2, bias=False)
        self.conv1d = nn.Conv1d(inner_dim, inner_dim, kernel_size=4, padding=3, groups=inner_dim)
        self.out_proj = nn.Linear(inner_dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)
        x_branch = self.conv1d(x_branch.transpose(1, 2))[..., :x.shape[1]].transpose(1, 2)
        x_branch = F.silu(x_branch)
        z = F.silu(z)
        return self.out_proj(x_branch * z) + residual`,
    inputs: [{ name: "x", type: "Tensor" }],
    outputs: [{ name: "out", type: "Tensor" }],
    constructorParams: [
      { name: "dim", type: "int" },
      { name: "state_dim", type: "int", default: 16 },
      { name: "expand", type: "int", default: 2 },
    ],
  },
  {
    id: "S4Block",
    name: "S4Block",
    category: "SSM/Mamba",
    code: `class S4Block(nn.Module):
    """Structured State Space (S4) block."""
    def __init__(self, dim: int, state_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.D = nn.Parameter(torch.ones(dim))
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        return self.dropout(self.proj(x) * self.D) + residual`,
    inputs: [{ name: "x", type: "Tensor" }],
    outputs: [{ name: "out", type: "Tensor" }],
    constructorParams: [
      { name: "dim", type: "int" },
      { name: "state_dim", type: "int", default: 64 },
      { name: "dropout", type: "float", default: 0.0 },
    ],
  },
];
