# Contributing to QLLM

Thank you for your interest in contributing to QLLM. This document explains
the process for contributing and the legal requirements.

---

## Contributor License Agreement (CLA)

**All contributions require signing a CLA before your pull request can be
merged.**

When you open your first pull request, a bot will automatically comment asking
you to sign. You sign by posting this comment on the PR:

> I have read the CLA Document and I hereby sign the CLA

By signing, you assign copyright of your contribution to the project owner
(Gowrav Vishwakarma) under the terms described in [CLA.md](CLA.md). You retain
a non-exclusive license to use your own contribution. This assignment covers
all future contributions to this repository.

The full CLA terms are in [CLA.md](CLA.md) -- please read it before signing.

---

## How to Contribute

1. **Fork** the repository and create a feature branch from `main`.
2. **Read the README** for the version directory you are working in (`v6/`, `v5/`, etc.).
3. Make your changes.
4. **Run a smoke test** before submitting:
   ```bash
   python -m v6.train --size tiny --epochs 5 --max_samples 1000 --seq_len 128
   ```
5. Open a **pull request** against `main` with a clear description of what you changed and why.
6. Sign the CLA when prompted by the bot (first-time contributors only).

---

## Code Standards

- Keep claims tied to logged experiments and configs -- no unverified performance numbers.
- Run the smoke test (above) to ensure nothing is broken.
- Follow existing code style and structure within each version directory.
- Do not commit large binary files, datasets, or model checkpoints.
- Keep commit messages concise and descriptive.

---

## Reporting Issues

Open a GitHub issue with:

- A clear title describing the problem or suggestion.
- Steps to reproduce (for bugs).
- Relevant logs, configs, or error messages.

---

## Intellectual Property Notice

All contributions to this repository are subject to the project's
[Contributor License Agreement](CLA.md). By contributing, you assign copyright
of your contribution to Gowrav Vishwakarma. The project owner retains full
rights to relicense, commercialize, and distribute the project under any
license.

The project is licensed under the
[PolyForm Noncommercial License 1.0.0](LICENSE). Commercial use requires
permission -- contact [gowravvishwakarma@gmail.com](mailto:gowravvishwakarma@gmail.com).
