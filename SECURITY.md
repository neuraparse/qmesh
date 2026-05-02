# Security Policy

`qmesh` is a research-stage quantum operating layer maintained by
[Neuraparse](https://neuraparse.com). We take security issues seriously,
especially anything that touches the **provenance / signed-manifest**
surface — that subsystem is the trust root for reproducible runs.

## Supported versions

`qmesh` is pre-1.0. Only the latest tagged release on `main` receives
security fixes. Older `0.x` tags are best-effort.

| Version  | Status                       |
| -------- | ---------------------------- |
| `0.1.x`  | supported (current)          |
| `< 0.1`  | not supported                |

## Reporting a vulnerability

**Please do not open a public GitHub issue for security problems.**

Email <security@neuraparse.com> with:

- a short description of the issue and its impact,
- a minimal reproduction (commit hash, command, or script),
- the affected `qmesh` version (`python -c "import qmesh; print(qmesh.__version__)"`),
- whether you would like credit in the advisory, and under what name.

If your report concerns the manifest signer, the AI copilot tool surface,
the FT scheduler, or the REST/CLI submit path, mark the subject line
`[qmesh-trust]` so it is routed to a maintainer immediately.

We will acknowledge receipt within **5 working days** and aim to send a
first triage assessment within **10 working days**. For valid issues we
target a fix or mitigation within **30 days**, longer if upstream library
coordination is required (TKET, BQSKit, MQT, Stim, Pulser, SF, …).

## Disclosure

We follow coordinated disclosure. Once a fix is available we publish a
GitHub Security Advisory and a `CHANGELOG.md` entry with the CVE (if
assigned) and credit to the reporter. Please give us a reasonable window
before any public write-up — typically the earlier of (a) the patched
release, or (b) **90 days** after first contact.

## Scope

In scope:

- Code in this repository (`qmesh/`, `tests/`, `examples/`, `scripts/`,
  `deploy/`).
- Manifests, signatures, and replay outputs produced by `qmesh.api.submit`
  / `qmesh.api.replay` / the `qmesh` CLI.
- Containers and compose files under `deploy/` and `docker-compose.yml`.

Out of scope:

- Vulnerabilities in third-party quantum SDKs themselves (please report
  upstream); we will, however, ship mitigations or version pins where
  appropriate.
- Issues that require a malicious local user already having root on the
  machine running `qmesh`.
- Denial-of-service that requires submitting circuits with absurd
  resource budgets the scheduler is documented to refuse.

## Safe-harbor

Good-faith research that follows this policy will not be pursued under
the Apache-2.0 license terms, the CFAA, or any equivalent local law. We
will not ask GitHub to take down a researcher's PoC repository as long
as it is not weaponized and does not contain stolen credentials.

Thank you for helping keep `qmesh` and its users safe.
