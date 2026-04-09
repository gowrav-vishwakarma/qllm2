# QLLM Architecture Builder UI

Visual node-based editor for designing and experimenting with neural network architectures. Build standard transformers, QLLM custom architectures (V4-V7), Mamba/SSM models, or entirely new designs -- then generate runnable training code and launch experiments.

## Quick Start

### 1. Start the backend

```bash
cd qllm_ui/backend
# Uses the project venv (Python 3.9+)
../../.venv/bin/python -m uvicorn app.main:app --reload --port 8000
```

### 2. Start the frontend

```bash
cd qllm_ui/frontend
# npm ci 
npm run dev
```

Open http://localhost:3000

## Features

- **Node Editor**: Drag-and-drop modules onto a canvas, connect input/output ports
- **Module Library**: 20+ built-in modules (QLLM Complex, Standard PyTorch, SSM/Mamba)
- **Custom Modules**: Write your own `nn.Module` with Monaco code editor; ports auto-inferred from `forward()` signature
- **Custom Loss/Backward**: Tab in module editor for defining custom loss functions or backward logic
- **Architecture Templates**: One-click load of V5, V6, V7, Standard Transformer, or Mamba architectures
- **Code Generation**: Converts your node graph into a complete runnable project (model.py, train.py, config.py)
- **Run Manager**: Launch training runs, stream logs, stop/monitor from the UI
- **Save/Load**: Projects saved as JSON, portable and version-controllable
- **Composite Nodes**: Select multiple nodes → right-click → Group into Module

## Architecture

```
Frontend (React + React Flow + Monaco)
    ↕ REST API / WebSocket
Backend (FastAPI)
    ↓ Code Generation
Output: model.py + train.py + config.py + modules/
```

## Built-in Module Categories

| Category     | Modules |
|-------------|---------|
| QLLM Custom | ComplexLinear, ComplexNorm, ComplexGatedUnit, ComplexEmbed, PhaseAssociativeLayer, ComplexSSM, PhaseAttention, WorkingMemory, ComplexLMHead |
| Standard    | nn.Linear, nn.LayerNorm, nn.Embedding, MultiheadAttention, TransformerEncoderLayer, FeedForward, SwiGLU, RMSNorm, RoPE, LM Head, Residual, Dropout |
| SSM/Mamba   | MambaBlock, S4Block |

## Templates

- **Standard Transformer**: Embedding → 3x TransformerEncoderLayer → LayerNorm → LM Head
- **V5 AlgebraicLM**: ComplexEmbed → ComplexSSM stack → PhaseAttention → ComplexLMHead
- **V6 PhaseFieldLM**: ComplexEmbed → CGU + PAM + WorkingMemory → ComplexLMHead
- **V7 LM**: ComplexEmbed → interleaved CGU + PAM blocks → ComplexLMHead
- **Mamba LM**: Embedding → 6x MambaBlock → LayerNorm → LM Head

## Project Structure

```
qllm_ui/
├── frontend/          # React + TypeScript + Vite
│   └── src/
│       ├── components/  # NodeEditor, LibraryPanel, PropertiesPanel, ModuleEditor, RunPanel, TopBar
│       ├── store/       # Zustand state management
│       ├── types/       # TypeScript types
│       ├── utils/       # API client, serialization, built-in modules
│       └── templates/   # Pre-built architecture templates
├── backend/           # Python FastAPI
│   └── app/
│       ├── routers/     # REST endpoints (projects, modules, codegen, runs)
│       └── services/    # Code generation, port parsing, run management
└── README.md
```
