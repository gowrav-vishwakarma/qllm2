import { useEffect } from "react";
import NodeEditor from "@/components/NodeEditor/NodeEditor";
import LibraryPanel from "@/components/LibraryPanel/LibraryPanel";
import PropertiesPanel from "@/components/PropertiesPanel/PropertiesPanel";
import ModuleEditor from "@/components/ModuleEditor/ModuleEditor";
import RunPanel from "@/components/RunPanel/RunPanel";
import TopBar from "@/components/TopBar/TopBar";
import { useStore } from "@/store/useStore";
import { builtinModules } from "@/utils/builtinModules";

export default function App() {
  const setModuleRegistry = useStore((s) => s.setModuleRegistry);

  useEffect(() => {
    setModuleRegistry(builtinModules);
  }, [setModuleRegistry]);

  return (
    <div className="h-screen w-screen flex flex-col bg-[#0f0f1a] text-white overflow-hidden">
      <TopBar />
      <div className="flex flex-1 min-h-0">
        <LibraryPanel />
        <div className="flex flex-col flex-1 min-w-0">
          <NodeEditor />
          <RunPanel />
        </div>
        <PropertiesPanel />
      </div>
      <ModuleEditor />
    </div>
  );
}
