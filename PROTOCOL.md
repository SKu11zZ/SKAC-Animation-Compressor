# Evaluation protocol

## 1. Tracks and layers

The automatic academic track compares SKAC, SAN, and R2ET. Engine and DCC baselines are reserved for a later industrial track. No clip-specific manual correction is allowed in the automatic track: one configuration is created per skeleton, hashed, and frozen before evaluation.

Each record belongs to exactly one quality layer:

- `retargeting`: prediction versus paired target motion;
- `codec_roundtrip`: decoded motion versus the pre-codec motion from the same retargeter;
- `artist_gold`: explicitly manual, reported separately.

Retargeting tables must never substitute codec error for retargeting error. The evaluator rejects a report containing multiple layers.

## 2. Primary R2ET-compatible protocol

The primary dataset is Mixamo with exactly 22 evaluated joints. A benchmark release must publish the immutable joint order, root index, coordinate convention, frame rate, unit conversion, character list, motion list, and all content hashes.

R2ET reports 1,952 non-overlapping training sequences from seven characters and 800 test sequences from 11 characters, with 60 sampled training frames and 120 frames per test sequence. A local reproduction must record any availability-driven deviation instead of presenting it as exact reproduction.

Every test sample is assigned by two independent booleans:

- target character seen or unseen during training;
- motion identity seen or unseen during training.

This yields `seen_character_seen_motion`, `seen_character_unseen_motion`, `unseen_character_seen_motion`, and `unseen_character_unseen_motion`. Membership is derived from frozen training ID sets, never inferred from result quality.

### Skeleton metrics

Let predicted and ground-truth global joint positions be `P` and `G` with shape `(T, 22, 3)`, target character height be `h > 0`, and root index be `r`.

- Global MSE raw: `mean((P - G)^2) / h^2` over frames, joints, and XYZ coordinates.
- Global MSE display: raw value multiplied by 1,000.
- Local MSE: translate every predicted frame so its root equals the ground-truth root, then apply the same MSE definition.

Both raw and `x1e3` fields are emitted so scale is never implicit.

### Geometry metrics

- Penetration `%`: for each frame, divide penetrated limb vertices by evaluated limb vertices, then average frame ratios and multiply by 100.
- Contact distance `cm`: mean hand-vertex distance to the body surface in centimeters.

The geometry backend, mesh version, limb/hand vertex sets, distance sign convention, resolution, and tolerances must be published. Until those are frozen, geometry values are marked unavailable rather than replaced by proxy metrics.

## 3. SAN compatibility

SAN has two intentionally separate reconstruction-error fields:

- `san_paper_position_error_x1e3`: mean Euclidean joint-position distance, divided by character height and multiplied by 1,000, matching the paper description of average error over joints and motions.
- `san_official_code_mse_raw`: mean squared XYZ-coordinate difference divided by height squared, matching the official evaluation implementation. Its `x1e3` display value is also emitted.

These values are not interchangeable. Reports must name the chosen field and may show both.

## 4. Aggregation

Metrics are computed per sample. Group tables use the arithmetic mean across samples and include sample counts. Under the exact 120-frame, 22-joint primary protocol this is equivalent to element-weighted aggregation. Overall results include all four groups but never cross quality layers or tracks.

## 5. Reproducibility checklist

A result is publishable only when it includes:

- data protocol version and source/access notes;
- sample manifest and hashes;
- frozen split IDs;
- 22-joint order and root index;
- skeleton configuration hashes proving one-time setup;
- method version, checkpoint hash, environment lock, and seed;
- geometry backend declaration when geometry metrics are reported;
- separate retargeting, codec, and Artist Gold reports;
- successful public-safety release audit.
