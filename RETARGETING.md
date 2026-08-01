# Frozen multi-target retargeting

The public retargeter decodes one source-bound `.skac` animation and applies a frozen
source-to-target profile. A target character is supplied as a BVH template; its
skeleton, channel order, rest offsets, and end sites are preserved in the output.

```text
python -m skac_codec profile motion.skac target.bvh -o target.skac-profile.json
python -m skac_codec decode motion.skac --target target.bvh \
  --profile target.skac-profile.json -o target-animation.bvh
```

Create the profile once for a source/target skeleton pair and reuse it for every clip
with the same source skeleton signature. The decoder rejects either skeleton when its
SHA-256 signature differs from the frozen profile.

## Profile construction

The builder performs these deterministic steps:

1. remove namespaces and match exact public joint names;
2. match common Mixamo, engine-humanoid, and SMPL-style semantic aliases;
3. pair different-length spine chains by normalized chain position;
4. preserve identity basis transfer when source and target joint names match;
5. otherwise derive a rest-direction basis change from the two skeletons;
6. compute root-translation scale from the public SAN-style head and leg chain length;
7. record every mapping, basis matrix, option, name, and skeleton signature in JSON;
8. hash the canonical profile so later edits cannot be mistaken for the frozen setup.

Target-only joints stay at their target rest rotation and inherit their animated parent
transform. This is appropriate for structural and attachment joints that have no source
counterpart; it does not synthesize independent cloth, hair, facial, or accessory
motion.

## Runtime transfer

Mapped source rotations are first accumulated globally. Each is expressed in the
frozen source/target rest basis, then converted back into the target hierarchy's local
rotation. Root motion is scaled once and written only to the target root, avoiding a
double rotation when an engine skeleton has both Root and Pelvis joints.

The optional `--contact-lock` profile flag enables an experimental root correction from
detected foot contacts. It is off by default because the current correction does not
yet pass the no-regression gate on every public target. It must not be enabled in a
published automatic-track result without reporting the option in the frozen profile.

## Current boundary

This milestone supports deterministic BVH targets, semantic mapping, rest-basis
alignment, hierarchy conversion, body-scale compensation, and hash-pinned profiles.
Production-quality twist distribution, end-effector IK, constraint limits, robust foot
locking, and FBX target assets remain later work. No clip-specific adjustment is
allowed in the automatic track.
