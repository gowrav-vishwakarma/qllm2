# SCI-1 architecture: forward + backward pseudocode

Model: PAM M2, 385.82M params, from v13_sempty/model.py at 59c3cf5 (flags in v13_sempty/tmp/sci1_chain.sh).

```python
# ===== SCI-1: 24 blocks, D=1024, 8 heads x d=128, MLP 3704; notebooks in blocks 3,7,11,15,19,23 =====
# Flags on: DELTA, CHRONO, OUT_GATE, SHARE_READS (M2), ECHO, RESET_AT_EOS.   State per notebook = 8 x 128x128 -> O(1)

def forward(tokens x[1..T]):
  h = RMSNorm(E[x])                                   # tied embedding E: vocab x 1024
  for l in 0..23:
    h = h + a_l * SwiGLU(RMSNorm(h))                  # SwiGLU(u) = W_down( silu(W_g u) * W_up u ),  a_l learned (init 1)
    if l in NOTEBOOK_LAYERS:  h = h + b_l * PAM(RMSNorm(h))   # b_l init 0.1 ("memory starts soft")
    elif l > 3:               h = h + READ(RMSNorm(h), notebook of nearest PAM layer below)   # M2 share_reads
  z  = RMSNorm(h)
  lm = RMSNorm(W_head z) + W_eo * ECHO(z, x)          # W_eo zero-init
  return logits = lm @ E^T                            # loss = CE(logits, next token), answer tokens weighted x100 in synthetic docs

def PAM(u):                                           # per head, per token t (training runs this as a chunked kernel, 256 tokens/chunk)
  q,k,v  = split(W_qkv u_t)                           # each 128-dim
  g_t    = exp(clamp(w_c . u_t, -3, 3))               # CHRONO: learned speed of time
  phi_t  = sum_{j<t} g_j * f                          # f_i = 10000^(-i/64): RoPE with a learned clock
  q,k    = rotate(q, phi_t), rotate(k, phi_t)
  k      = k / |k|
  r_t    = exp(-softplus(w_dt . u_t + bias))          # retention (forget) in (0,1); bias -4
  if x_t == EOS: r_t = 0                              # RESET_AT_EOS: new document = fresh notebook
  e_t    = min(sigmoid(w_e . u_t - 3), 0.95)          # erase gate
  w_t    = sigmoid(w_w . u_t)                         # write gate
  S_t    = r_t * S_{t-1} (I - e_t k k^T) + w_t v k^T  # DELTA RULE: fade, erase old value at key k, write new value
  y_t    = silu(w_o . u_t) * S_t q / sqrt(128)        # read, scaled by a per-head output gate
  return W_out concat_heads(y_t)

def READ(u, S):                                       # notebook-less block reads a lower block's notebook S
  q'  = rotate(W_q' u_t, phi_t of the writer)         # query lands in the writer's own time frame
  return W_o' ( silu(w_g . u_t) * S_t q' / sqrt(128) )   # W_o' zero-init -> identity at start

def ECHO(z, x):                                       # copy / induction memory just before the output head
  k_t = unit(W_k z_{t-1});  v_t = rmsnorm(E[x_t]);  q_t = W_q z_t
  S_t = r_t S_{t-1}(I - e_t k k^T) + w_t v k^T        # same delta rule, gates read from z_t
  return S_t q_t                                      # "the token that followed the last state like this one"

# ===== BACKWARD (autograd + a hand-written Triton kernel; per head, reverse time) =====
# Write A_t = r_t (I - e_t k k^T)   (symmetric), so  S_t = S_{t-1} A_t + w_t v k^T,  y_t = c_t S_t q / sqrt(d)
dS = 0
for t = T..1:
  dS     += c_t * dy_t q^T / sqrt(d)  + sum over readers of (their dy q'^T)   # M2: readers' gradients flow into the shared notebook
  dq      = c_t * S_t^T dy_t / sqrt(d)
  dc_t    = dy_t . (S_t q) / sqrt(d)
  dv      = w_t * dS k
  dw_t    = v^T dS k
  dA      = S_{t-1}^T dS
  dr_t    = <dA, I - e_t k k^T>                       # -> d(dt logit) = -dr_t * r_t * sigmoid(logit)
  de_t    = -r_t * k^T dA k
  dk      = w_t dS^T v - r_t e_t (dA + dA^T) k        # then back through unit-norm and rotation (rotation is orthogonal: R^T)
  dS      = dS A_t                                    # carry to t-1  (A_t = 0 at EOS -> gradients stop at the document edge)
# Everything else (SwiGLU, RMSNorm, linear layers, tied E, CE) is standard autograd. AdamW, LR 8e-4, warmup-stable-decay schedule.
```
