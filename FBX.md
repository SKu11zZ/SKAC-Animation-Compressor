# Experimental FBX adapter

FBX support is an optional adapter around the public Codec and retargeter. The `.skac`
container continues to store animation and its source skeleton descriptor; it does not
store meshes, materials, textures, skin weights, or complete character scenes.

The adapter currently uses Blender in background mode. Blender must provide its FBX
import/export operators and the BVH import/export add-on. Pass the executable explicitly
when it is not available as `blender` on `PATH`.

## Workflow

Extract a source or target FBX armature animation to BVH:

```text
python -m skac_codec fbx-extract character.fbx -o character.bvh --blender BLENDER
```

Use the normal Codec and frozen-profile commands:

```text
python -m skac_codec encode source.bvh -o motion.skac --quality high
python -m skac_codec profile motion.skac target.bvh -o target.skac-profile.json
python -m skac_codec decode motion.skac --target target.bvh \
  --profile target.skac-profile.json -o target-animation.bvh
```

Bake the target BVH animation into the original target FBX scene:

```text
python -m skac_codec fbx-inject target.fbx target-animation.bvh \
  -o animated-target.fbx --blender BLENDER
python -m skac_codec fbx-validate animated-target.fbx --blender BLENDER
```

Use `--armature NAME` when an FBX scene contains multiple armatures. Otherwise the
adapter selects the animated armature, falling back to the armature with the most bones.

## Preservation and validation contract

The injection bridge imports the target FBX, imports the already retargeted BVH as a
driver, matches bones by namespace-independent names, copies pose transforms, and bakes
the result onto the target armature. The driver is removed before FBX export. The target
scene's meshes, materials, skinning, and non-driver objects stay in the scene.

The bridge then opens the newly exported FBX in a clean Blender scene and rejects it
when:

- no armature is present;
- a source bone was lost;
- a source mesh or material was lost;
- no animation keyframes are present;
- an external image dependency is missing.

Outputs are written to a temporary sibling first and replace the requested destination
only after the bridge reports success. Blender is invoked with an argument array,
factory startup, no shell, an explicit timeout, and a temporary JSON report.

## Current validation status

The adapter command construction, UTF-8 process handling, temporary-output behavior,
failure handling, and Blender bridge syntax are covered by unit tests. Blender 4.5.12
LTS successfully extracted all 38 files in the checked public local run: 23 Mixamo and
15 Manny animation FBX files. One neutral sample per family also completed
`FBX -> BVH -> .skac -> BVH -> FBX`, retained its armature animation and keyframes, and
reported no missing external dependency.

Those 38 inputs are animation-only files with no Mesh or material. The run therefore
does not validate skinned-Mesh deformation, material/texture preservation, multiple
takes, constraints, or layered FBX animation. The backend remains experimental rather
than production-validated. The portable Blender binary, source FBX files, extracted
BVH cache, and round-trip outputs remain local and are not part of this repository.

FBX can express bind rotations, pivots, animation layers, multiple takes, constraints,
and scaling that BVH cannot. The current bridge deliberately bakes evaluated pose
transforms into one action. A future production backend must add a public FBX fixture
matrix and verify coordinate systems, bind pose, mesh deformation, material retention,
root motion, and animation timing across supported Blender/FBX versions.
