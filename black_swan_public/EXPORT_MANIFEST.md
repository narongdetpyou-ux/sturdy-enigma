# Export Manifest

Export target: `narongdetpyou-ux/sturdy-enigma/black_swan_public`

Source: private `narongdetpyou-ux/back_swan`

Export policy: public-safe, minimum necessary copy.

## Included exact source copies

- `src/black_swan/__init__.py`
- `src/black_swan/engine.py`
- `src/black_swan/runtime.py`
- `src/black_swan_v7_engine.py`
- `pyproject.toml`
- `LICENSE`

## Explicit exclusions

- `data/**`
- `experiments/**`
- `reports/**`
- `archive/**`
- `backups/**`
- `incoming/**`
- private or local configuration
- credential/security-status material
- repository history
- private ZIPs and recovered source bundles
- any secrets, credentials, personal data, or machine-specific paths

The purpose of this export is to provide the executable logic needed to understand or reuse the core implementation without copying private evidence stores or unnecessary sensitive project material.
