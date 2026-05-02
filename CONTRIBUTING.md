# Contributing to qmesh

Thanks for your interest in contributing to **qmesh** — a [Neuraparse](https://neuraparse.com)
open-source project. This document is the short version of how we work; the
longer "why" lives in [`PLAN.md`](./PLAN.md) and [`ARCHITECTURE.md`](./ARCHITECTURE.md).

By participating, you agree to abide by our
[Code of Conduct](./CODE_OF_CONDUCT.md).

---

## Table of contents

1. [Ground rules](#ground-rules)
2. [Setting up a dev environment](#setting-up-a-dev-environment)
3. [Running tests, lint, and types](#running-tests-lint-and-types)
4. [Project conventions](#project-conventions)
5. [Submitting a change](#submitting-a-change)
6. [Reporting bugs and security issues](#reporting-bugs-and-security-issues)
7. [License of contributions (DCO)](#license-of-contributions-dco)

---

## Ground rules

- **Compose, don't replace.** qmesh wraps best-of-breed libraries (TKET,
  BQSKit, MQT, Mitiq, Stim, Aer, Pulser, SF, …). When you need a feature that
  an upstream library already implements well, wrap it — don't reimplement it.
- **Provenance is non-negotiable.** Every executable surface (CLI, REST,
  scheduler nodes, AI flows, FT runs) must emit a signed manifest by default.
  `sign=False` is for tests only.
- **α-then-β cadence.** Each phase ships in two slices: α (working skeleton +
  real numbers + a centerpiece demo) before β (depth, integrations, scale).
  Don't bundle a β slice into an unrelated α merge.
- **Tests track centerpieces.** A new feature is "done" when there is a small
  (≈5–15) test suite that exercises the headline claim deterministically.
  Anything that touches randomness must accept a `seed` argument and the test
  must pass it.

## Setting up a dev environment

Requirements:

- Python **3.11 or 3.12** (the host's interpreter must be `python3` — qmesh
  doesn't assume a `python` symlink).
- A POSIX shell (Linux or macOS). Windows users should develop inside WSL2.

```bash
git clone https://github.com/neuraparse/qmesh.git
cd qmesh
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[all,dev]'
```

The `[all]` extra pulls every optional backend (Qiskit/Aer, Cirq, PennyLane,
Pulser, SF, Braket, TKET, BQSKit, MQT, Mitiq, Torch, OpenAI/Anthropic). The
`[dev]` extra adds `pytest`, `ruff`, `mypy`, `hypothesis`. If you only want a
slice, the extras are independently installable (`pip install -e '.[qiskit]'`,
`pip install -e '.[ai,service]'`, …).

If your contribution doesn't touch a particular backend, you don't need its
extra installed — the test suite skips backend-specific tests when the
underlying library is unavailable.

## Running tests, lint, and types

The full single-shot smoke is:

```bash
PYTHONPATH=. python3 -m pytest tests/ -q       # 137 passing in ~9.5s
PYTHONPATH=. python3 -m ruff check qmesh tests
PYTHONPATH=. python3 -m mypy qmesh
```

Run all three before opening a PR. (We will wire CI on `main` later — for
now correctness is enforced by reviewers, not by the build system.)

For phase-specific demos see [`README.md` › Quick start](./README.md#quick-start).
Demos must keep working — if your change breaks one, fix the demo or call it
out in the PR description.

## Project conventions

- **Line length:** 100 (`ruff`). Long lines are allowed for tabular data /
  doctest blocks (`E501` is intentionally ignored).
- **Type hints:** `mypy --strict`. Public functions get full annotations;
  internal helpers may use `from __future__ import annotations` postponed
  evaluation.
- **Docstrings:** keep them short and one-paragraph. The "why" lives in
  PLAN.md / ARCHITECTURE.md, not in every file header.
- **Commits:** present-tense, imperative ("Add BB-code stabilizer generator"),
  ≤72 chars on the subject line. Reference issues with `Refs: #123` when
  relevant.
- **Provenance:** any new executable surface must emit a signed `Manifest`
  by default. New code paths that produce results without a manifest are
  rejected in review.
- **No emojis** in committed source unless the user-facing string explicitly
  requires one.

## Submitting a change

1. Open an issue first for anything non-trivial — saves rework.
2. Fork, branch, write code, write tests, run the three commands above.
3. Open a PR against `main`. Fill in the PR template; describe what changed,
   how it was tested, and which manifest fields are touched.
4. A maintainer will review. We aim for first response within five working
   days.

There is no CI yet — reviewers run the test/lint/types suite locally before
merging. Please make sure your branch is green at HEAD on Python 3.11 (and
3.12 if your change touches typing or syntax).

## Reporting bugs and security issues

- **Functional bugs / feature requests:** open a [GitHub issue](https://github.com/neuraparse/qmesh/issues)
  using the appropriate template.
- **Security vulnerabilities:** **do not** open a public issue. Email
  [security@neuraparse.com](mailto:security@neuraparse.com) — see
  [`SECURITY.md`](./SECURITY.md) for our disclosure policy.

## License of contributions (DCO)

qmesh is licensed under [Apache 2.0](./LICENSE). By submitting a contribution,
you certify that you have the right to submit it under that license, per the
[Developer Certificate of Origin (DCO)](https://developercertificate.org/).
We do **not** require a separate CLA.

You can sign off your commits with `git commit -s`, which appends:

```
Signed-off-by: Your Name <you@example.com>
```

Sign-off is encouraged but not currently enforced by CI; we may turn on the
DCO bot later as the contributor base grows.

---

Questions? Reach us at <open-source@neuraparse.com> or open a discussion on
the repository.
