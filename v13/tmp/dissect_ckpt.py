#!/usr/bin/env python3
"""Static dissection of a V13 checkpoint: protect gate, phase, write-phase, betas.
Usage: python v13/tmp/dissect_ckpt.py <ckpt.pt>
CPU-only, no GPU contention.
"""
import math
import sys

import torch


def main(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    cfg = ck.get("config", {})
    n = cfg.get("n_layers", 16)
    print(f"ckpt step={ck.get('step', '?')}")

    def mean(f):
        return sum(f) / len(f)

    pg, gb = [], []
    for i in range(n):
        b = sd.get(f"blocks.{i}.pam.protect_gate.bias")
        if b is not None:
            with torch.no_grad():
                gb.append(float(b.mean()))
                pg.append(float(torch.sigmoid(b).mean()))
    print(f"protect_gate: bias [{min(gb):+.3f},{max(gb):+.3f}]  mean_protect [{min(pg):.3f},{max(pg):.3f}]")

    ph_w, ph_b = [], []
    wp_w, wp_b = [], []
    for i in range(n):
        w = sd.get(f"blocks.{i}.pam.phase_proj.weight")
        b = sd.get(f"blocks.{i}.pam.phase_proj.bias")
        if w is not None:
            with torch.no_grad():
                ph_w.append(float(w.norm()))
                ph_b.append(float(b.norm()))
        w = sd.get(f"blocks.{i}.pam.write_phase_proj.weight")
        b = sd.get(f"blocks.{i}.pam.write_phase_proj.bias")
        if w is not None:
            with torch.no_grad():
                wp_w.append(float(w.norm()))
                wp_b.append(float(b.norm()))
    print(f"phase_proj:      wnorm [{min(ph_w):.3f},{max(ph_w):.3f}]  bnorm [{min(ph_b):.4f},{max(ph_b):.4f}]")
    print(f"write_phase_proj: wnorm [{min(wp_w):.4f},{max(wp_w):.4f}]  bnorm [{min(wp_b):.4f},{max(wp_b):.4f}]")

    def sig(x):
        return 1 / (1 + math.exp(-x))
    for li in (0, 8, 15):
        bw = float(sd[f"blocks.{li}.pam.beta_proj.bias"].mean())
        be = float(sd[f"blocks.{li}.pam.erase_beta_proj.bias"].mean())
        print(f"L{li:2d} beta_w bias {bw:+.3f} (sig {sig(bw):.3f})  beta_e bias {be:+.3f} (sig {sig(be):.3f})")


if __name__ == "__main__":
    main(sys.argv[1])
