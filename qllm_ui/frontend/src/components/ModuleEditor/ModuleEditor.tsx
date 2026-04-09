import { useState, useCallback, useEffect, useMemo } from "react";
import Editor from "@monaco-editor/react";
import { useStore } from "@/store/useStore";
import { api } from "@/utils/api";
import type { ModuleDef, PortDef, ConstructorParam } from "@/types";

const DEFAULT_CODE = `import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModule(nn.Module):
    def __init__(self, dim: int, expand: int = 4):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * expand)
        self.w2 = nn.Linear(dim * expand, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))
`;

const DEFAULT_BACKWARD_CODE = `# Optional: Define custom backward logic or loss computation.
# This code will be included in the generated training script.
# Available variables: logits, labels, model, optimizer
#
# Example custom loss:
# def custom_loss(logits, labels, model):
#     ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
#     # Add regularization
#     reg_loss = sum(p.norm() for p in model.parameters()) * 1e-5
#     return ce_loss + reg_loss
`;

type EditorTab = "module" | "backward";

export default function ModuleEditor() {
  const {
    editorOpen,
    editorModuleId,
    closeEditor,
    moduleRegistry,
    addCustomModule,
    updateCustomModule,
    nodes,
  } = useStore();

  const existingModule = useMemo(
    () => moduleRegistry.find((m: ModuleDef) => m.id === editorModuleId),
    [moduleRegistry, editorModuleId]
  );

  const [code, setCode] = useState(existingModule?.code || DEFAULT_CODE);
  const [backwardCode, setBackwardCode] = useState(DEFAULT_BACKWARD_CODE);
  const [activeTab, setActiveTab] = useState<EditorTab>("module");
  const [moduleName, setModuleName] = useState(existingModule?.name || "MyModule");
  const [inputs, setInputs] = useState<PortDef[]>(existingModule?.inputs || []);
  const [outputs, setOutputs] = useState<PortDef[]>(existingModule?.outputs || []);
  const [ctorParams, setCtorParams] = useState<ConstructorParam[]>(
    existingModule?.constructorParams || []
  );
  const [inferring, setInferring] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (existingModule) {
      setCode(existingModule.code);
      setModuleName(existingModule.name);
      setInputs(existingModule.inputs);
      setOutputs(existingModule.outputs);
      setCtorParams(existingModule.constructorParams);
    } else {
      setCode(DEFAULT_CODE);
      setModuleName("MyModule");
      setInputs([]);
      setOutputs([]);
      setCtorParams([]);
    }
  }, [existingModule, editorModuleId]);

  const inferPorts = useCallback(async () => {
    setInferring(true);
    setError(null);
    try {
      const result = await api.inferPorts(code);
      setInputs(result.inputs);
      setOutputs(result.outputs);
      setCtorParams(result.constructorParams);
    } catch (e: any) {
      setError(e.message);
      const fwdMatch = code.match(/def forward\(self,?\s*([^)]*)\)/);
      if (fwdMatch) {
        const params = fwdMatch[1]
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean)
          .map((s) => {
            const [name] = s.split(":");
            return { name: name.trim(), type: "Tensor" };
          });
        setInputs(params);
      }
      const returnMatch = code.match(/return\s+(.+)/);
      if (returnMatch) {
        const ret = returnMatch[1].trim();
        if (ret.includes(",")) {
          const parts = ret.split(",").map((s, i) => ({
            name: `out_${i}`,
            type: "Tensor",
          }));
          setOutputs(parts);
        } else {
          setOutputs([{ name: "out", type: "Tensor" }]);
        }
      }
    } finally {
      setInferring(false);
    }
  }, [code]);

  const handleSave = useCallback(() => {
    const finalInputs =
      inputs.length > 0 ? inputs : [{ name: "x", type: "Tensor" }];
    const finalOutputs =
      outputs.length > 0 ? outputs : [{ name: "out", type: "Tensor" }];

    if (existingModule && existingModule.isCustom) {
      updateCustomModule(existingModule.id, {
        code,
        name: moduleName,
        inputs: finalInputs,
        outputs: finalOutputs,
        constructorParams: ctorParams,
      });
    } else if (!existingModule || !existingModule.isCustom) {
      const newMod: ModuleDef = {
        id: moduleName.replace(/\s+/g, ""),
        name: moduleName,
        category: "My Modules",
        code,
        inputs: finalInputs,
        outputs: finalOutputs,
        constructorParams: ctorParams,
        isCustom: true,
      };
      addCustomModule(newMod);
    }
    closeEditor();
  }, [
    code,
    moduleName,
    inputs,
    outputs,
    ctorParams,
    existingModule,
    updateCustomModule,
    addCustomModule,
    closeEditor,
  ]);

  if (!editorOpen) return null;

  const isReadOnly = existingModule && !existingModule.isCustom;

  return (
    <div className="fixed inset-0 z-50 bg-black/70 flex items-center justify-center">
      <div className="bg-[#1e1e2e] rounded-lg shadow-2xl border border-white/10 w-[80vw] h-[85vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center gap-4 px-4 py-3 border-b border-white/10">
          <h2 className="text-sm font-semibold text-white">
            {isReadOnly ? "View Module" : "Edit Module"}
          </h2>
          {!isReadOnly && (
            <input
              type="text"
              value={moduleName}
              onChange={(e) => setModuleName(e.target.value)}
              className="px-2 py-1 text-sm bg-[#0f0f1a] border border-white/10 rounded text-gray-200 focus:outline-none focus:border-indigo-500"
              placeholder="Module name"
            />
          )}
          <div className="flex-1" />
          <button
            onClick={inferPorts}
            disabled={inferring}
            className="px-3 py-1.5 text-xs font-medium bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white rounded transition-colors"
          >
            {inferring ? "Inferring..." : "Infer Ports"}
          </button>
          {!isReadOnly && (
            <button
              onClick={handleSave}
              className="px-3 py-1.5 text-xs font-medium bg-green-600 hover:bg-green-700 text-white rounded transition-colors"
            >
              Save Module
            </button>
          )}
          <button
            onClick={closeEditor}
            className="px-3 py-1.5 text-xs font-medium bg-gray-600 hover:bg-gray-700 text-white rounded transition-colors"
          >
            Close
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-white/10">
          <button
            onClick={() => setActiveTab("module")}
            className={`px-4 py-2 text-xs font-medium transition-colors ${
              activeTab === "module"
                ? "text-white border-b-2 border-indigo-500"
                : "text-gray-500 hover:text-gray-300"
            }`}
          >
            Module Code
          </button>
          <button
            onClick={() => setActiveTab("backward")}
            className={`px-4 py-2 text-xs font-medium transition-colors ${
              activeTab === "backward"
                ? "text-white border-b-2 border-orange-500"
                : "text-gray-500 hover:text-gray-300"
            }`}
          >
            Custom Loss / Backward
          </button>
        </div>

        {/* Editor */}
        <div className="flex-1 min-h-0">
          <Editor
            language="python"
            theme="vs-dark"
            value={activeTab === "module" ? code : backwardCode}
            onChange={(v) => {
              if (activeTab === "module") setCode(v || "");
              else setBackwardCode(v || "");
            }}
            options={{
              readOnly: activeTab === "module" && !!isReadOnly,
              fontSize: 13,
              minimap: { enabled: false },
              scrollBeyondLastLine: false,
              padding: { top: 12 },
              lineNumbers: "on",
            }}
          />
        </div>

        {/* Port display */}
        <div className="px-4 py-3 border-t border-white/10 flex gap-8 text-xs">
          {error && (
            <div className="text-red-400 text-xs mb-1">
              Backend unavailable, using regex fallback: {error}
            </div>
          )}
          <div>
            <span className="text-gray-400 font-semibold uppercase">
              Inputs:{" "}
            </span>
            {inputs.length === 0 ? (
              <span className="text-gray-600">none detected</span>
            ) : (
              inputs.map((p) => (
                <span key={p.name} className="text-blue-400 mr-3">
                  {p.name}: {p.type}
                  {p.optional ? "?" : ""}
                </span>
              ))
            )}
          </div>
          <div>
            <span className="text-gray-400 font-semibold uppercase">
              Outputs:{" "}
            </span>
            {outputs.length === 0 ? (
              <span className="text-gray-600">none detected</span>
            ) : (
              outputs.map((p) => (
                <span key={p.name} className="text-orange-400 mr-3">
                  {p.name}: {p.type}
                </span>
              ))
            )}
          </div>
          <div>
            <span className="text-gray-400 font-semibold uppercase">
              Constructor:{" "}
            </span>
            {ctorParams.length === 0 ? (
              <span className="text-gray-600">none</span>
            ) : (
              ctorParams.map((p) => (
                <span key={p.name} className="text-green-400 mr-3">
                  {p.name}: {p.type}
                  {p.default !== undefined ? `=${String(p.default)}` : ""}
                </span>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
