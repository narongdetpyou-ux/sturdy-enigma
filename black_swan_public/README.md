# Black Swan — public-safe mirror

This directory contains a deliberately reduced copy of the Black Swan v8 research prototype from the private `back_swan` repository.

## Included

- Core v8 Python package: `src/black_swan/`
- v7 reference math required by v8: `src/black_swan_v7_engine.py`
- Packaging metadata: `pyproject.toml`
- License

The copied source was screened before export for common secret patterns such as API keys, access tokens, private keys, password assignments, email addresses, and absolute user-home paths.

## Not included

To keep this public repository safe and minimal, this mirror intentionally excludes:

- raw/processed datasets and external data
- experiment outputs, logs, benchmark result artifacts, and local run manifests
- reports containing detailed internal results
- archives, ZIP backups, recovered historical files, and release bundles
- incoming/private working files
- storage-policy and private-location mappings
- credential/security-incident tracking documents
- notebooks that depend on private archives
- repository history and any secret or credential material

## Status

Black Swan v8 is a research prototype for deterministic anomaly triage and decision support. It is not a trained LLM and is not production-certified. The private source repository records Phase 1 as complete, Phase 2 criteria as ready but not executed, and Phase 3 as not started.

## Python

Python 3.12+ is expected. The package has no runtime dependencies declared in `pyproject.toml`.

This mirror is intentionally incomplete. Do not treat absence of private validation artifacts here as evidence that those artifacts do not exist in the source repository.
