import { useStore } from "@/store/useStore";
import { useCallback, useMemo } from "react";
import type { ModuleDef } from "@/types";

function ParamInput({
  name,
  type,
  value,
  defaultVal,
  onChange,
}: {
  name: string;
  type: string;
  value: unknown;
  defaultVal?: unknown;
  onChange: (val: unknown) => void;
}) {
  const strVal = value !== undefined ? String(value) : "";

  if (type === "bool") {
    return (
      <label className="flex items-center gap-2 cursor-pointer">
        <input
          type="checkbox"
          checked={Boolean(value)}
          onChange={(e) => onChange(e.target.checked)}
          className="accent-indigo-500"
        />
        <span className="text-sm text-gray-300">{name}</span>
      </label>
    );
  }

  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs text-gray-400">
        {name}
        {defaultVal !== undefined && (
          <span className="text-gray-600 ml-1">(default: {String(defaultVal)})</span>
        )}
      </label>
      <input
        type={type === "int" || type === "float" ? "number" : "text"}
        step={type === "float" ? "any" : "1"}
        value={strVal}
        onChange={(e) => {
          const raw = e.target.value;
          if (type === "int") onChange(parseInt(raw) || 0);
          else if (type === "float") onChange(parseFloat(raw) || 0);
          else onChange(raw);
        }}
        className="px-2 py-1 text-sm bg-[#0f0f1a] border border-white/10 rounded text-gray-200 focus:outline-none focus:border-indigo-500"
      />
    </div>
  );
}

export default function PropertiesPanel() {
  const { selectedNodeId, nodes, updateNodeParams, openEditor, moduleRegistry, training, setTraining, project, setProject } = useStore();

  const selectedNode = useMemo(
    () => nodes.find((n) => n.id === selectedNodeId),
    [nodes, selectedNodeId]
  );

  const moduleDef = useMemo(() => {
    if (!selectedNode) return null;
    return moduleRegistry.find((m: ModuleDef) => m.id === selectedNode.data.moduleId) || null;
  }, [selectedNode, moduleRegistry]);

  const handleParamChange = useCallback(
    (name: string, value: unknown) => {
      if (selectedNodeId) updateNodeParams(selectedNodeId, { [name]: value });
    },
    [selectedNodeId, updateNodeParams]
  );

  if (!selectedNode || !moduleDef) {
    return (
      <div className="w-72 bg-[#1a1a2e] border-l border-white/10 flex flex-col h-full overflow-y-auto">
        <div className="p-3 border-b border-white/10">
          <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
            Properties
          </h2>
        </div>

        {/* Project config when no node selected */}
        <div className="p-3 space-y-3">
          <h3 className="text-xs text-gray-400 font-semibold uppercase">Project</h3>
          {(["name", "outputDir", "checkpointDir", "logDir"] as const).map((field) => (
            <div key={field} className="flex flex-col gap-1">
              <label className="text-xs text-gray-400">{field}</label>
              <input
                type="text"
                value={project[field]}
                onChange={(e) => setProject({ [field]: e.target.value })}
                className="px-2 py-1 text-sm bg-[#0f0f1a] border border-white/10 rounded text-gray-200 focus:outline-none focus:border-indigo-500"
              />
            </div>
          ))}

          <h3 className="text-xs text-gray-400 font-semibold uppercase mt-4">Training</h3>
          <ParamInput name="dataset" type="str" value={training.dataset} onChange={(v) => setTraining({ dataset: String(v) })} />
          <ParamInput name="tokenizer" type="str" value={training.tokenizer} onChange={(v) => setTraining({ tokenizer: String(v) })} />
          <ParamInput name="seqLen" type="int" value={training.seqLen} onChange={(v) => setTraining({ seqLen: Number(v) })} />
          <ParamInput name="batchSize" type="int" value={training.batchSize} onChange={(v) => setTraining({ batchSize: Number(v) })} />
          <ParamInput name="epochs" type="int" value={training.epochs} onChange={(v) => setTraining({ epochs: Number(v) })} />
          <ParamInput name="lr" type="float" value={training.optimizer.lr} onChange={(v) => setTraining({ optimizer: { ...training.optimizer, lr: Number(v) } })} />
          <ParamInput name="weightDecay" type="float" value={training.optimizer.weightDecay} onChange={(v) => setTraining({ optimizer: { ...training.optimizer, weightDecay: Number(v) } })} />
          <ParamInput name="gradClip" type="float" value={training.gradClip} onChange={(v) => setTraining({ gradClip: Number(v) })} />
          <ParamInput name="warmupSteps" type="int" value={training.scheduler.warmupSteps} onChange={(v) => setTraining({ scheduler: { ...training.scheduler, warmupSteps: Number(v) } })} />
          <ParamInput name="gradAccumulation" type="int" value={training.gradAccumulation} onChange={(v) => setTraining({ gradAccumulation: Number(v) })} />

          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" checked={training.amp} onChange={(e) => setTraining({ amp: e.target.checked })} className="accent-indigo-500" />
            <span className="text-sm text-gray-300">AMP (Mixed Precision)</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" checked={training.compile} onChange={(e) => setTraining({ compile: e.target.checked })} className="accent-indigo-500" />
            <span className="text-sm text-gray-300">torch.compile</span>
          </label>
        </div>
      </div>
    );
  }

  return (
    <div className="w-72 bg-[#1a1a2e] border-l border-white/10 flex flex-col h-full overflow-y-auto">
      <div className="p-3 border-b border-white/10">
        <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
          Properties
        </h2>
      </div>

      <div className="p-3 space-y-1">
        <div className="text-sm font-semibold text-white">{moduleDef.name}</div>
        <div className="text-xs text-gray-500">{moduleDef.category}</div>
        <div className="text-[10px] text-gray-600 font-mono">{selectedNode.id}</div>
      </div>

      <div className="px-3 pb-2 border-b border-white/10">
        <button
          onClick={() => openEditor(moduleDef.id)}
          className="text-xs text-indigo-400 hover:text-indigo-300 transition-colors"
        >
          View / Edit Code →
        </button>
      </div>

      <div className="p-3 space-y-3">
        <h3 className="text-xs text-gray-400 font-semibold uppercase">
          Constructor Params
        </h3>
        {moduleDef.constructorParams.map((p) => (
          <ParamInput
            key={p.name}
            name={p.name}
            type={p.type}
            value={selectedNode.data.params[p.name]}
            defaultVal={p.default}
            onChange={(v) => handleParamChange(p.name, v)}
          />
        ))}
      </div>

      <div className="p-3 space-y-1 border-t border-white/10">
        <h3 className="text-xs text-gray-400 font-semibold uppercase">Ports</h3>
        <div className="text-xs text-gray-500">
          <div className="font-semibold text-gray-400 mt-1">Inputs:</div>
          {moduleDef.inputs.map((p) => (
            <div key={p.name} className="ml-2">
              {p.name}: <span className="text-blue-400">{p.type}</span>
              {p.optional && <span className="text-gray-600"> (optional)</span>}
            </div>
          ))}
          <div className="font-semibold text-gray-400 mt-1">Outputs:</div>
          {moduleDef.outputs.map((p) => (
            <div key={p.name} className="ml-2">
              {p.name}: <span className="text-orange-400">{p.type}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
