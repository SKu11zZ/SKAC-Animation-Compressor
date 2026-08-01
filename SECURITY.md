# Public-safety boundary

This subtree is public-only and must remain independent from product engineering.

Allowed source families are Mixamo, Manny, SMPL, MetaHuman, published papers,
official public research code, and newly created benchmark code/configuration. Every
manifest entry must declare its public provenance and license or access note.

The following rules are mandatory:

1. Do not import, inspect, summarize, infer from, or reference product-private assets,
   configurations, experiments, identifiers, paths, screenshots, or derived values.
2. Do not scan parent directories. Release auditing is scoped to this subtree only.
3. Store data outside version control and refer to it only through relative paths below
   a declared public-data root.
   Generated `.skac` files are asset data: they can contain source joint names and must
   not be committed without a separate provenance and redistribution review.
   Generated retarget profiles also contain source and target joint names and follow the
   same rule.
4. Automatic-track configuration is frozen once per skeleton. Per-animation tuning is
   rejected.
5. Artist Gold is a separate track and can never be aggregated with automatic results.
6. Retargeting and codec round-trip quality are separate layers and can never be
   aggregated together.
7. Before publication, run the release audit with the private denylist supplied out of
   band through `SKAC_PRIVATE_DENYLIST` or `--denylist-file`.

If a requested step needs private information, stop that step and report that the
public-safety boundary blocks the request.
