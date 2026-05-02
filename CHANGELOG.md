# Changelog

All notable changes to **qmesh** are documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Pre-1.0 minor bumps may include breaking changes; we will call them out
explicitly in the relevant section.

## [Unreleased]

### Added
- _placeholder for in-flight work; see open PRs._

### Changed
- _nothing yet._

### Fixed
- _nothing yet._

---

## [0.1.0] — 2026-05-02

First public release of `qmesh`. Phases 1–6 ship in α/β/γ/δ slices per
`PLAN.md`. Cuts a research-backed surface broad enough to demo the
2026-native story end-to-end: modality-agnostic IR → AI-augmented
compiler → FT-mode scheduling → reproducibility-as-runtime.

### Added

#### Core IR & API
- Modality-agnostic IR: `Module` / `Function` / `Region` / `Op` with
  `Modality` tags for gate, neutral-atom analog, photonic-CV and
  pulse-level programs.
- Builder front-end: `qmesh.ir.builder.circuit` for ergonomic
  construction, plus `qmesh.api.{submit, replay, diff}` as the single
  user-facing entry point.
- `Backend` / `Capabilities` / `RunResult` abstractions; backends are
  composed (we wrap upstream libraries; we do not reimplement them).

#### Compiler & FT
- AI-augmented passes feeding TKET / BQSKit / MQT, with deterministic
  fallbacks when the AI copilot is unavailable.
- Stim + PyMatching pipeline for surface and color codes; hooks for
  AlphaQubit-style learned decoders behind a clean trait.
- Magic-state cultivation accounted for in resource estimates by default
  (no explicit distillation-factory step).

#### Scheduling
- FT-mode scheduler with `MCMRegion` semantics that admit both
  feedforward and measurement-free logical compute as peers.
- Modality-aware placement covering gate-based, neutral-atom analog,
  photonic-CV and pulse-level backends.

#### Provenance / reproducibility-as-runtime
- Signed `Manifest` on every executable surface (CLI, REST, scheduler
  nodes, AI flows, FT runs). `sign=True` is the default; `sign=False`
  is reserved for tests.
- `replay` and `diff` operate over manifests; bit-for-bit reproducibility
  for deterministic seeds.

#### Tooling & ops
- `qmesh` CLI (`qmesh.cli.__main__:app`) wired through `pyproject`
  scripts.
- `examples/` scripts and single-shot smoke commands for each phase.
- `deploy/` and `docker-compose.yml` for the service-mode REST surface.
- 157 passing tests + 1 skipped (intentional, environment-gated).

#### Project / community files
- `LICENSE` (Apache-2.0), `NOTICE`, `README.md`, `ARCHITECTURE.md`,
  `PLAN.md`, `CONTRIBUTING.md`.
- `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), `SECURITY.md`,
  `CITATION.cff`, this `CHANGELOG.md`, and `.github/` issue + PR
  templates.

### Notes

- `qmesh` is pre-1.0 and the IR / manifest schema may still shift.
  Manifests carry an explicit version field; `replay` will refuse
  cross-version manifests rather than silently degrade.
- There is no CI yet; reviewers run the test, lint and type suites
  locally before merging (see `CONTRIBUTING.md`).

---

[Unreleased]: https://github.com/neuraparse/qmesh/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/neuraparse/qmesh/releases/tag/v0.1.0
