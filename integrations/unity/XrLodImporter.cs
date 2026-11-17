// Unity editor script: drop into Assets/Editor/. When a model whose meshes end in _LOD0/_LOD1/_LOD2 is
// imported, Unity already builds a LODGroup from the suffix; this postprocessor adds the project's XR rules:
// screen-height thresholds per LOD, static batching flags, and a warning if LOD0 is over the triangle budget.
// NOTE: written for Unity 2022 LTS editor APIs; not executed in this repo's CI (no Unity there).
using UnityEditor;
using UnityEngine;

public class XrLodImporter : AssetPostprocessor
{
    const int Lod0TriangleBudget = 500000;
    static readonly float[] ScreenHeights = { 0.5f, 0.2f, 0.05f };

    void OnPostprocessModel(GameObject root)
    {
        var group = root.GetComponentInChildren<LODGroup>();
        if (group == null) return;
        var lods = group.GetLODs();
        for (int i = 0; i < lods.Length && i < ScreenHeights.Length; i++)
            lods[i].screenRelativeTransitionHeight = ScreenHeights[i];
        group.SetLODs(lods);

        int tris = 0;
        foreach (var r in lods[0].renderers)
        {
            var mf = r.GetComponent<MeshFilter>();
            if (mf != null && mf.sharedMesh != null) tris += mf.sharedMesh.triangles.Length / 3;
            GameObjectUtility.SetStaticEditorFlags(r.gameObject, StaticEditorFlags.BatchingStatic);
        }
        if (tris > Lod0TriangleBudget)
            Debug.LogWarning($"{assetPath}: LOD0 has {tris} triangles, over the XR budget of {Lod0TriangleBudget}");
    }
}
