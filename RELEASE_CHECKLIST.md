# Release checklist

This checklist applies to every public release. Run all commands from the repository
root and review their output before committing or pushing.

## Repository metadata

- [ ] Confirm the `repository-code` URL in `CITATION.cff` matches the release repository.
- [ ] Confirm the project version agrees in `pyproject.toml` and `CITATION.cff`.
- [ ] Confirm Apache-2.0 is the intended license for original repository code.
- [ ] Update `THIRD_PARTY_NOTICES.md` when an upstream dependency or adapter changes.

## Content boundary

- [ ] Include only original source, tests, documentation, templates, and aggregate reports.
- [ ] Exclude datasets, weights, character assets, motions, generated motions, archives, binaries, and rendered media. Static aggregate-report SVG is allowed.
- [ ] Exclude machine-local paths, environment dumps, command histories, and raw logs.
- [ ] Confirm every report contains aggregate public results only.
- [ ] Confirm every committed quality-gate JSON has a matching static SVG and passes its frozen thresholds.
- [ ] Supply the private release denylist out of band; never commit the denylist itself.

## Verification

```text
python -m unittest discover -s tests -v
python tools/audit_release.py . --require-private-denylist
python tools/hash_tree.py . --output ../skac-release-files.json
```

`skac-release-files.json` is a local review artifact outside the repository. Delete it
after comparing the final candidate tree.

For a standalone Git repository, also inspect exactly what Git will publish:

```text
git status --short
git diff --cached --stat
git diff --cached
git ls-files
```

Do not use a broad add command from a parent engineering repository. Stage the files
from this standalone repository explicitly, inspect the staged diff, and push only
after all checks pass.
