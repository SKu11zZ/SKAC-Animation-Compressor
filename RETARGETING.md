# Frozen multi-target Codec playback

Same-character playback does not use this system: it decodes the `.skac` animation
directly back onto its hash-pinned source skeleton. Profiles are only compiled for the
different-character route, so their setup and per-frame cost never appear in the
same-character quality report.

The public runtime decodes one source-bound `.skac` animation and applies a frozen,
precompiled source-to-target profile. A target character is supplied as a BVH template;
its skeleton, channel order, rest offsets, and end sites are preserved in the output.

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
2. identify public naming conventions before resolving ambiguous shoulder names;
3. match common Mixamo, engine-humanoid, and SMPL-style semantic aliases;
4. pair different-length spine chains by normalized chain position;
5. preserve identity basis transfer when source and target joint names match;
6. otherwise derive a rest-direction basis change from the two skeletons;
7. compute root-translation scale from the public SAN-style head and leg chain length;
8. compile source/target evaluation orders and basis quaternions for playback;
9. record core-joint coverage and hash the canonical profile.

Target-only joints stay at their target rest rotation and inherit their animated parent
transform. This is appropriate for structural and attachment joints that have no source
counterpart; it does not synthesize independent cloth, hair, facial, or accessory
motion.

## Runtime playback

Profile 2.0 is a small execution plan. Mapping, naming decisions, ancestor discovery,
rest-basis conversion, and target parent lookup are completed once by the Builder.
Playback uses quaternion products directly, evaluates only required hierarchy joints,
reuses scratch buffers, and writes into caller-owned output arrays. It does not build
dictionaries, convert every joint to matrices, run IK, or allocate a new pose per frame.

The same compiled runtime is reused for every animation with the pinned source skeleton.
Its scratch state is intentionally owned by one playback instance; applications use a
separate runtime instance per concurrently evaluated character.

The slower matrix path remains only as a quality-gate oracle. Version 1 profiles can
still be loaded and compiled, while newly built profiles use schema 2.0.

The optional legacy contact correction is not accepted by the real-time compiled plan.
It needs frame history and global-position passes, so it conflicts with this milestone's
hot-path budget. A future stateful implementation must pass the same performance gate
before it can enter the playback path.

## Current boundary

This milestone favors predictable compressed playback over an expensive offline solve.
It supports deterministic BVH targets, naming-aware semantic mapping, rest-basis
alignment, hierarchy conversion, body-scale compensation, hash-pinned plans, and a
zero-extra-allocation frame API. It deliberately does not add per-frame IK or general
constraint solving. No clip-specific adjustment is allowed in the automatic track.
