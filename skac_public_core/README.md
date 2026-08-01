# SKAC public cross-skeleton core

This package is an independently written, public-safe retargeting core. It contains no
product assets, private configuration, character-specific constants, or external path
references.

The API separates static skeleton setup from motion evaluation:

```python
from skac_public_core import Motion, Skeleton, build_profile, retarget_motion

profile = build_profile(source_skeleton, target_skeleton)
target_motion = retarget_motion(profile, source_motion)
```

`build_profile` creates an immutable semantic-name mapping once per source-target
skeleton pair. `retarget_motion` copies mapped local rotations, initializes unmatched
target joints to identity, and scales root translation by the frozen target-to-source
height ratio. Source and target skeletons may have different joint counts and topology.

The current implementation is deliberately minimal and deterministic. It is the public
pipeline baseline, not a claim of feature parity with any non-public implementation.
