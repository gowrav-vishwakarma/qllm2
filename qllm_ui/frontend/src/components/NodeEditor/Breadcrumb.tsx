import { useStore } from "@/store/useStore";

export default function Breadcrumb() {
  const viewStack = useStore((s) => s.viewStack);
  const drillOut = useStore((s) => s.drillOut);
  const drillOutTo = useStore((s) => s.drillOutTo);

  if (viewStack.length === 0) return null;

  return (
    <div className="h-8 bg-[#151525] border-b border-white/10 flex items-center px-4 gap-1 text-xs shrink-0">
      <button
        onClick={() => drillOutTo(0)}
        className="text-indigo-400 hover:text-indigo-300 transition-colors font-medium"
      >
        Root
      </button>
      {viewStack.map((entry, i) => (
        <span key={i} className="flex items-center gap-1">
          <span className="text-gray-600">/</span>
          {i < viewStack.length - 1 ? (
            <button
              onClick={() => drillOutTo(i + 1)}
              className="text-indigo-400 hover:text-indigo-300 transition-colors font-medium"
            >
              {entry.parentLabel}
            </button>
          ) : (
            <span className="text-white font-semibold">{entry.parentLabel}</span>
          )}
        </span>
      ))}
      <div className="flex-1" />
      <button
        onClick={drillOut}
        className="px-2 py-0.5 text-[10px] text-gray-400 hover:text-white hover:bg-white/10 rounded transition-colors"
      >
        Back
      </button>
    </div>
  );
}
