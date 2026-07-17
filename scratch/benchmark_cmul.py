#!/usr/bin/env python3
"""Benchmark and correctness script for standard vs Gauss 3-multiply complex multiplication.

This script compares the codebase's standard 4-multiply complex multiplication
with Gauss's 3-multiply method across eager and compiled runtimes on both CPU and GPU.
"""

import sys
import time
import argparse
from pathlib import Path
import torch

# Ensure parent directory is in sys.path to import active complex_ops
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import active cmul from codebase
from v11.complex_ops import cmul as cmul_standard

# Define Gauss's 3-multiply multiplication
def cmul_gauss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Gauss 3-multiply, 5-addition method."""
    ar, ai = a[..., 0], a[..., 1]
    br, bi = b[..., 0], b[..., 1]
    
    k1 = ar * (br + bi)
    k2 = bi * (ar + ai)
    k3 = br * (ai - ar)
    
    return torch.stack([k1 - k2, k1 + k3], dim=-1)


def check_correctness(device: torch.device, shape_a, shape_b, name: str):
    """Verify forward and backward correctness between standard and Gauss methods using double precision."""
    print(f"\n--- Running Correctness Checks for {name} on {device} ---")
    
    # Check in double precision to avoid float32 rounding accumulation limits
    a_double = torch.randn(*shape_a, device=device, dtype=torch.float64, requires_grad=True)
    b_double = torch.randn(*shape_b, device=device, dtype=torch.float64, requires_grad=True)
    a_gauss = a_double.clone().detach().requires_grad_(True)
    b_gauss = b_double.clone().detach().requires_grad_(True)
    
    out_std = cmul_standard(a_double, b_double)
    out_gauss = cmul_gauss(a_gauss, b_gauss)
    
    fwd_diff = (out_std - out_gauss).abs().max().item()
    print(f"  [Double] Forward pass max diff: {fwd_diff:.2e}")
    assert fwd_diff < 1e-11, f"Forward outputs do not match for {name} (Double)!"
    
    loss_std = (out_std ** 2).sum()
    loss_gauss = (out_gauss ** 2).sum()
    
    loss_std.backward()
    loss_gauss.backward()
    
    grad_a_diff = (a_double.grad - a_gauss.grad).abs().max().item()
    grad_b_diff = (b_double.grad - b_gauss.grad).abs().max().item()
    print(f"  [Double] Backward grad_a max diff: {grad_a_diff:.2e}")
    print(f"  [Double] Backward grad_b max diff: {grad_b_diff:.2e}")
    
    assert grad_a_diff < 1e-11, f"grad_a does not match for {name} (Double)!"
    assert grad_b_diff < 1e-11, f"grad_b does not match for {name} (Double)!"
    
    # Also log Float32 diffs for informativeness (without asserting strict thresholds)
    a_float = a_double.float().detach().requires_grad_(True)
    b_float = b_double.float().detach().requires_grad_(True)
    a_g_float = a_float.clone().detach().requires_grad_(True)
    b_g_float = b_float.clone().detach().requires_grad_(True)
    
    out_std_f = cmul_standard(a_float, b_float)
    out_gauss_f = cmul_gauss(a_g_float, b_g_float)
    fwd_diff_f = (out_std_f - out_gauss_f).abs().max().item()
    
    (out_std_f ** 2).sum().backward()
    (out_gauss_f ** 2).sum().backward()
    grad_a_diff_f = (a_float.grad - a_g_float.grad).abs().max().item()
    grad_b_diff_f = (b_float.grad - b_g_float.grad).abs().max().item()
    
    print(f"  [Float32 info] Forward diff: {fwd_diff_f:.2e} | grad_a: {grad_a_diff_f:.2e} | grad_b: {grad_b_diff_f:.2e}")
    print("  Correctness tests: PASS")


def benchmark_op(op, a, b, num_warmup=20, num_iters=100) -> float:
    """Run warmups and time a given function forward + backward pass."""
    # Warmup
    for _ in range(num_warmup):
        out = op(a, b)
        loss = (out ** 2).sum()
        loss.backward()
        a.grad.zero_()
        if b.grad is not None:
            b.grad.zero_()
            
    # Sync device
    if a.device.type == "cuda":
        torch.cuda.synchronize()
        
    start_time = time.time()
    for _ in range(num_iters):
        out = op(a, b)
        loss = (out ** 2).sum()
        loss.backward()
        a.grad.zero_()
        if b.grad is not None:
            b.grad.zero_()
            
    if a.device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()
    
    return (end_time - start_time) / num_iters * 1000  # Return mean execution time in ms


def run_benchmark_suite():
    parser = argparse.ArgumentParser(description="Benchmark Standard vs Gauss cmul.")
    parser.add_argument("--iters", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=20, help="Number of warmup iterations")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (B)")
    parser.add_argument("--heads", type=int, default=4, help="Number of heads (H)")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length (T)")
    parser.add_argument("--dim", type=int, default=32, help="Head dimension (d)")
    args = parser.parse_args()
    
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
        
    # Scaled down to prevent local machine memory overload (OOM)
    B, H, T, d = args.batch_size, args.heads, args.seq_len, args.dim
    
    scenarios = [
        {
            "name": "Standard (No Broadcast)",
            "shape_a": [B, H, T, d, 2],
            "shape_b": [B, H, T, d, 2]
        },
        {
            "name": "RoPE Broadcast",
            "shape_a": [B, H, T, d, 2],
            "shape_b": [1, 1, T, d, 2]  # pos broadcasts to B and H
        },
        {
            "name": "State Rotation Broadcast",
            "shape_a": [B, H, T, d, 2],
            "shape_b": [B, H, T, 1, 2]  # rotation broadcasts to d
        }
    ]
    
    # Run correctness checks first
    for device in devices:
        for scenario in scenarios:
            check_correctness(device, scenario["shape_a"], scenario["shape_b"], scenario["name"])
            
    # Benchmark runs
    results = []
    
    # Number of complex multiplication operations performed per pass
    ops_per_pass = B * H * T * d
    
    for device in devices:
        print(f"\n=======================================================")
        print(f"Benchmarking on {device.type.upper()}")
        print(f"=======================================================")
        
        for scenario in scenarios:
            name = scenario["name"]
            shape_a = scenario["shape_a"]
            shape_b = scenario["shape_b"]
            
            # Setup tensors
            a = torch.randn(*shape_a, device=device, requires_grad=True)
            b = torch.randn(*shape_b, device=device, requires_grad=True)
            
            # Compile versions
            compiled_std_default = torch.compile(cmul_standard, mode="default")
            compiled_gauss_default = torch.compile(cmul_gauss, mode="default")
            
            compiled_std_reduce = torch.compile(cmul_standard, mode="reduce-overhead")
            compiled_gauss_reduce = torch.compile(cmul_gauss, mode="reduce-overhead")
            
            # Run eager benchmarks
            t_std_eager = benchmark_op(cmul_standard, a, b, args.warmup, args.iters)
            t_gauss_eager = benchmark_op(cmul_gauss, a, b, args.warmup, args.iters)
            
            # Run compile default benchmarks
            t_std_comp_def = benchmark_op(compiled_std_default, a, b, args.warmup, args.iters)
            t_gauss_comp_def = benchmark_op(compiled_gauss_default, a, b, args.warmup, args.iters)
            
            # Run compile reduce-overhead benchmarks
            t_std_comp_red = benchmark_op(compiled_std_reduce, a, b, args.warmup, args.iters)
            t_gauss_comp_red = benchmark_op(compiled_gauss_reduce, a, b, args.warmup, args.iters)
            
            # Helper to calculate Giga-operations/sec (complex multiplications per sec)
            # Ops/sec = ops_per_pass / (time_in_ms / 1000)
            # G-ops/sec = ops_per_pass / (time_in_ms * 1e6)
            def to_gops(t_ms):
                return ops_per_pass / (t_ms * 1e6)
            
            results.append({
                "Device": device.type.upper(),
                "Scenario": name,
                "Runtime": "Eager",
                "Standard (ms)": t_std_eager,
                "Gauss (ms)": t_gauss_eager,
                "Std Giga-ops/s": to_gops(t_std_eager),
                "Gauss Giga-ops/s": to_gops(t_gauss_eager),
                "Speedup": (t_std_eager / t_gauss_eager - 1.0) * 100
            })
            
            results.append({
                "Device": device.type.upper(),
                "Scenario": name,
                "Runtime": "Compile (default)",
                "Standard (ms)": t_std_comp_def,
                "Gauss (ms)": t_gauss_comp_def,
                "Std Giga-ops/s": to_gops(t_std_comp_def),
                "Gauss Giga-ops/s": to_gops(t_gauss_comp_def),
                "Speedup": (t_std_comp_def / t_gauss_comp_def - 1.0) * 100
            })
            
            results.append({
                "Device": device.type.upper(),
                "Scenario": name,
                "Runtime": "Compile (reduce-overhead)",
                "Standard (ms)": t_std_comp_red,
                "Gauss (ms)": t_gauss_comp_red,
                "Std Giga-ops/s": to_gops(t_std_comp_red),
                "Gauss Giga-ops/s": to_gops(t_gauss_comp_red),
                "Speedup": (t_std_comp_red / t_gauss_comp_red - 1.0) * 100
            })

    # Print results in a markdown table
    print("\n\n" + "="*116)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*116)
    
    headers = ["Device", "Scenario", "Runtime", "Std (ms)", "Gauss (ms)", "Std G-ops/s", "Gauss G-ops/s", "Speedup (%)"]
    row_format = "| {:<6} | {:<25} | {:<27} | {:<8} | {:<10} | {:<11} | {:<13} | {:<11} |"
    
    print(row_format.format(*headers))
    print("|" + "-"*8 + "|" + "-"*27 + "|" + "-"*29 + "|" + "-"*10 + "|" + "-"*12 + "|" + "-"*13 + "|" + "-"*15 + "|" + "-"*13 + "|")
    
    for r in results:
        speedup_str = f"{r['Speedup']:+.2f}%"
        print(row_format.format(
            r["Device"],
            r["Scenario"],
            r["Runtime"],
            f"{r['Standard (ms)']:.3f}",
            f"{r['Gauss (ms)']:.3f}",
            f"{r['Std Giga-ops/s']:.2f}",
            f"{r['Gauss Giga-ops/s']:.2f}",
            speedup_str
        ))
    print("="*116)


if __name__ == "__main__":
    run_benchmark_suite()
