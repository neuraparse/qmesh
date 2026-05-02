<!--
Thanks for sending a PR to qmesh!

Please keep the description short and concrete. The "why" lives in
PLAN.md / ARCHITECTURE.md; this template is just enough for a reviewer
to know what changed and how to validate it.
-->

## Summary

<!-- One or two sentences: what does this PR change? -->

## Surface(s) touched

<!-- Tick all that apply. Delete the rest. -->

- [ ] IR (`qmesh/ir/**`)
- [ ] Compiler pass (TKET / BQSKit / MQT wrapping)
- [ ] Backend (`qmesh/backends/**`)
- [ ] FT scheduler (`MCMRegion`, placement)
- [ ] QEC + decoders (Stim / PyMatching / learned)
- [ ] AI copilot
- [ ] Provenance / manifest / replay / diff
- [ ] CLI / REST / `deploy/`
- [ ] Tests / examples / docs only

## α-then-β

<!-- Per CONTRIBUTING.md: α and β slices ship separately. -->

- [ ] This is an **α** slice (working skeleton + real numbers + a centerpiece demo).
- [ ] This is a **β** slice (depth, integrations, scale on top of an existing α).
- [ ] N/A — pure bugfix / docs / tooling.

## Manifest impact

<!-- If this PR adds or changes a manifest field, list the keys and whether the change is back-compatible for `replay`. -->

- Manifest fields added / changed: `…`
- `replay` compatibility: backward-compatible / requires version bump / N/A

## How was this tested?

<!--
A new feature is "done" when there is a small, deterministic test suite
exercising the headline claim. Anything touching randomness must accept
a `seed` and the test must pass it.
-->

- [ ] `pytest` is green locally on Python 3.11.
- [ ] `ruff check .` is clean.
- [ ] `mypy --strict qmesh` is clean.
- [ ] New tests added (paths): `…`

## Related issues / refs

<!-- e.g. `Refs: #123` or a paper / vendor announcement -->

## Checklist

- [ ] I read `CONTRIBUTING.md` and agree to the DCO (`git commit -s`).
- [ ] I followed the "compose, don't replace" rule — wrapping upstream
      libraries instead of reimplementing.
- [ ] No emojis added to source unless the user-facing string requires
      one.
- [ ] No `sign=False` outside tests.
