using System.IO;
using UnrealBuildTool;

public class SKACRuntimeBeta : ModuleRules
{
    public SKACRuntimeBeta(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
        PublicDependencyModuleNames.Add("Core");

        string RepositoryRoot = Path.GetFullPath(Path.Combine(PluginDirectory, "../../.."));
        PublicIncludePaths.Add(Path.Combine(RepositoryRoot, "native", "include"));
        PrivateDefinitions.Add("SKAC_RUNTIME_STATIC=1");
    }
}
